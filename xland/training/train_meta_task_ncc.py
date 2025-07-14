# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import os
import shutil
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional, Dict

import jax
import jax.experimental
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np 
import optax
import orbax
import pyrallis
from pyrallis import field
import wandb
import xminigrid
import chex
from flax import core,struct
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState as BaseTrainState
from nn import ActorCriticRNN
from utils import calculate_gae, ppo_update_networks, rollout, save_params
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import GymAutoResetWrapper

from jaxued.level_sampler import LevelSampler as BaseLevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from ncc_utils import scale_y_by_ti_ada, ScaleByTiAdaState, ti_ada, projection_simplex_truncated

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)

class LevelSampler(BaseLevelSampler):

    def level_weights(self, sampler, *args,**kwargs):
        return sampler["scores"]
    
    def initialize(self, levels, level_extras):
        sampler = {
                "levels": levels,
                "scores": jnp.full(self.capacity, 1 / self.capacity, dtype=jnp.float32),
                "timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
                "size": self.capacity,
                "episode_count": 0,
        }
        if level_extras is not None:
            sampler["levels_extra"] = level_extras
        return sampler

@dataclass 
class PrioritizationParams:
    temperature: float = 1.0
    k: int = 1

@dataclass
class TrainConfig:
    project: str = "minigrid"
    mode: str = "online"
    group: str = "medium-13-ncc"
    env_id: str = "XLand-MiniGrid-R4-13x13"
    benchmark_id: str = "high-3m"
    img_obs: bool = False
    # agent
    meta_trunc: float = 1e-5
    meta_lr: float = 1e-3
    meta_reg: float = 0.0

    action_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    # training
    replace_iters: int = 1
    num_envs: int = 8192
    num_steps_per_env: int = 4096
    num_steps_per_update: int = 32
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 5.7e9
    learning_rate: float = 0.0001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    #eval
    eval_num_envs: int = 512
    eval_num_episodes: int = 10
    num_eval_steps: int = 500
    eval_seed: int = 42
    train_seed: int = 42
    checkpoint_path: Optional[str] = "checkpoints"
    #ued
    static_buffer: bool = False
    exploratory_grad_updates: bool = True
    ued_score_function: str = "MaxMC"
    replay_prob: float = 0.95
    buffer_capacity: int = 4000
    staleness_coeff: float = 0.3
    minimum_fill_ratio: float = 1.0
    prioritization: str = "rank"
    prioritization_params: PrioritizationParams = field(default_factory=PrioritizationParams)
    duplicate_check: bool = False
    sfl_buffer_refresh_freq: int = 1
    #logging
    log_num_images: int = 20  # number of images to log
    log_images_count: int = 16 # number of times to log images during training

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # splitting computation across all available devices
        assert num_devices == 1, "Only single device training is supported."
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_num_envs_per_device = self.eval_num_envs // num_devices
        self.lr = self.learning_rate
        assert self.num_envs % num_devices == 0
        self.num_meta_updates = round(
            self.total_timesteps_per_device / (self.num_envs_per_device * self.num_steps_per_env)
        )
        self.log_images_update = self.num_meta_updates // self.log_images_count
        print('logging images every', self.log_images_update)
        self.num_inner_updates = self.num_steps_per_env // self.num_steps_per_update
        assert self.num_steps_per_env % self.num_steps_per_update == 0
        print(f"Num devices: {num_devices}, Num meta updates: {self.num_meta_updates}")

class TrainState(BaseTrainState):
    # === UED ===
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)

    # === NCC Adversary ===
    y: chex.Array = struct.field(pytree_node=True)
    y_tx = struct.field(pytree_node=False)
    y_opt_state: ScaleByTiAdaState = struct.field(pytree_node=True)
    
    # === Below is used for logging ===
    num_updates: int
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, sampler, y, y_tx, num_updates, **kwargs):
        orig_ts = super().create(
            apply_fn = apply_fn,
            params=params,
            tx = tx
        )

        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx = tx,
            opt_state=orig_ts.opt_state,
            sampler=sampler,
            y=y,
            y_tx = y_tx,
            y_opt_state = y_tx.init(y),
            num_updates=num_updates
        )

def rollout_nsteps_score(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_steps: int = 500
):
    """ Rollout for `num_steps` environment steps """

    def _env_step(carry, unused):
        rng, prev_timestep, prev_action, prev_reward, hstate = carry

        rng, _rng = jax.random.split(rng)
        dist, value, hstate = train_state.apply_fn(
            train_state.params,
            {
                "observation": prev_timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action, log_prob = dist.sample_and_log_prob(seed=_rng)
        action, value, log_prob = action.squeeze(), value.squeeze(1), log_prob.squeeze(1)
        timestep = env.step(env_params, prev_timestep, action)
        carry = (rng, timestep, action, timestep.reward, hstate)
        return carry, Transition(
            done=jnp.zeros_like(timestep.last()),
            ep_done=timestep.last(),
            action=action,
            value=value,
            reward=timestep.reward,
            log_prob=log_prob,
            obs=prev_timestep.observation,
            prev_action=prev_action,
            prev_reward=prev_reward,
        )

    timestep = env.reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, timestep, prev_action, prev_reward, init_hstate)

    final_carry, transitions = jax.lax.scan(_env_step, init_carry, None, length=num_steps)

     # CALCULATE ADVANTAGE
    rng, timestep, prev_action, prev_reward, hstate = final_carry
    # calculate value of the last step for bootstrapping
    _, last_val, _ = train_state.apply_fn(
        train_state.params,
        {
                "observation": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
        },
        hstate,
    )

    return transitions, last_val
    
    
class Transition(struct.PyTreeNode):
    done: jax.Array
    ep_done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    # for rnn policy
    prev_action: jax.Array
    prev_reward: jax.Array
    
class UEDTrajBatch(struct.PyTreeNode):
    # for calculating UED score
    ep_done: jax.Array
    value: jax.Array
    reward: jax.Array
    advantage: jax.Array

def compute_score(score_fn, dones, values, max_returns, advantages):
    if score_fn == "MaxMC":
        return max_mc(dones, values, max_returns, 0.0)
    elif score_fn == "pvl":
        return positive_value_loss(dones, advantages, 0.0)
    else:
        raise ValueError(f"Unknown score function: {score_fn}")

def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.
    
    Args:
        train_state (TrainState): 
        level_sampler (LevelSampler): 

    Returns:
        dict: 
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    return {
        "log":{
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
    }

def make_states(config: TrainConfig):
    # for learning rate scheduling
    def linear_schedule(count):
        total_inner_updates = config.num_minibatches * config.update_epochs * config.num_inner_updates
        frac = 1.0 - (count // total_inner_updates) / config.num_meta_updates
        return config.lr * frac

    # setup environment
    if "XLand" not in config.env_id:
        raise ValueError("Only meta-task environments are supported.")

    env, env_params = xminigrid.make(config.env_id)
    env = GymAutoResetWrapper(env)

    # enabling image observations if needed
    if config.img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    # loading benchmark
    benchmark = xminigrid.load_benchmark(config.benchmark_id)

    # set up training state
    rng = jax.random.key(config.train_seed)
    rng, _rng = jax.random.split(rng)

    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
    )
    # [batch_size, seq_len, ...]
    init_obs = {
        "observation": jnp.zeros((config.num_envs_per_device, 1, *env.observation_shape(env_params))),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(ti_ada)(vy0 = jnp.zeros(config.buffer_capacity), eta=linear_schedule),  # eps=1e-5
    )
    
    # set up level sampler for UED
    prioritization_params = {"temperature": config.prioritization_params.temperature, "k": config.prioritization_params.k}
    level_sampler = LevelSampler(
        capacity=config.buffer_capacity,
        replay_prob=config.replay_prob,
        staleness_coeff=config.staleness_coeff,
        minimum_fill_ratio=config.minimum_fill_ratio,
        prioritization=config.prioritization,
        prioritization_params=prioritization_params,
        duplicate_check=config.duplicate_check,
    )
    rng, _rng = jax.random.split(rng)
    pholder_level = benchmark.sample_ruleset(_rng)
    sampler = None # level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
    
    init_y = jnp.full((config.buffer_capacity,), 1 / config.buffer_capacity, dtype=jnp.float32)
    y_ti_ada = scale_y_by_ti_ada(eta=config.meta_lr)
    y_opt_state = y_ti_ada.init(init_y)
    train_state = TrainState.create(
        apply_fn=network.apply, 
        params=network_params, 
        tx=tx, 
        sampler=sampler, 
        num_updates=0,
        y=init_y,
        y_tx=y_ti_ada,
        y_opt_state=y_opt_state
    )

    return rng, env, env_params, benchmark, level_sampler, init_hstate, train_state


def make_train(
    env: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    level_sampler: LevelSampler,
    config: TrainConfig,
):
    
    def log_levels(sampler, env_params, step):
        
        sorted_scores = jnp.argsort(sampler["scores"])[-config.log_num_images:]
        rulesets = jax.tree.map(lambda x: x[sorted_scores], sampler["levels"])
        # rulesets_to_log = jax.tree.map(lambda x: x[:config.log_num_images], )
        l_env_params = env_params.replace(ruleset=rulesets)
        prng = jax.random.PRNGKey(step)
        init_timestep = jax.vmap(env.reset, in_axes=(0, 0))(l_env_params, jax.random.split(prng, num=config.log_num_images))
            

        log_dict = {}
        for i in range(rulesets.rules.shape[0]):
            r = jax.tree.map(lambda x: x[i], rulesets)
            env_params = env_params.replace(ruleset=r)
            t = jax.tree.map(lambda x: x[i], init_timestep)
            img = env.render(env_params, t)
            log_dict.update({f"images/{i}_level": wandb.Image(np.array(img))})
        print('step', step)
        wandb.log(log_dict, step=step)
        
    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
        init_hstate: jax.Array,
    ):

        def score_fn(rng, rulesets, num_envs, train_state):
            
            if config.ued_score_function == "MaxMC" or config.ued_score_function == "pvl":
                
                eval_env_params = env_params.replace(ruleset=rulesets)

                eval_reset_rng = jax.random.split(rng, num=num_envs)

                eval_trajs, last_vals = jax.vmap(rollout_nsteps_score, in_axes=(0, None, 0, None, None, None))(
                    eval_reset_rng,
                    env,
                    eval_env_params,
                    train_state,
                    jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                    config.num_eval_steps
                )

                transitions = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1).squeeze(), eval_trajs)
                advantages, targets = calculate_gae(transitions, last_vals.squeeze(), config.gamma, config.gae_lambda)
                max_returns = compute_max_returns(transitions.ep_done, transitions.reward)
                scores = compute_score(config.ued_score_function, transitions.ep_done, transitions.value, max_returns, advantages)

                return scores, max_returns
    
            else:
                eval_env_params = env_params.replace(ruleset=rulesets)
                def rollout_fn(rng):

                    eval_reset_rng = jax.random.split(rng, num=num_envs)

                    eval_stats = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None))(
                        eval_reset_rng,
                        env,
                        eval_env_params,
                        train_state,
                        # TODO: make this a static method?
                        jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                        1, # num_consecutive_episodes
                    )

                    return eval_stats.success, eval_stats.reward

                rng, _rng = jax.random.split(rng)
                sucesses, returns = jax.vmap(rollout_fn)(jax.random.split(_rng, num=config.eval_num_episodes))

                p = sucesses.mean(axis=0)
                scores = p * (1 - p)

                return scores, returns.max(axis=0)

        def replace_fn(rng, train_state, old_level_scores):
            # NOTE: scores here are the actual UED scores, NOT the probabilities induced by the projection

            # Sample new levels
            rng, _rng = jax.random.split(rng)
            ruleset_rng = jax.random.split(rng, num=config.eval_num_envs_per_device)
            new_levels = jax.vmap(benchmark.sample_ruleset)(ruleset_rng)

            rng, _rng = jax.random.split(rng)
            new_level_scores, max_returns = score_fn(_rng, new_levels, config.eval_num_envs_per_device, train_state)

            idxs = jnp.flipud(jnp.argsort(new_level_scores))

            new_levels = jax.tree_util.tree_map(
                lambda x: x[idxs], new_levels
            )
            new_level_scores = new_level_scores[idxs]

            update_sampler = {**train_state.sampler,"scores": old_level_scores}

            sampler, _ = level_sampler.insert_batch(update_sampler, new_levels, new_level_scores, {"max_return": max_returns})
        
            return sampler

        # META TRAIN LOOP
        def _meta_step(meta_state, update_idx):
            rng, train_state = meta_state

            # Get y's gradient
            rng, _rng, _rng3 = jax.random.split(rng, 3)
            scores, _ = score_fn(_rng, train_state.sampler["levels"], config.buffer_capacity, train_state)

            rng, _rng = jax.random.split(rng)
            new_sampler  = {**train_state.sampler, "scores": scores} if config.static_buffer else replace_fn(_rng, train_state, scores)

            y_obj_fn = lambda y: y.T @ new_sampler["scores"] - config.meta_reg * jnp.log(y).T @ y
            grad = jax.grad(y_obj_fn)(train_state.y)
            grad, y_opt_state = train_state.y_tx.update(grad, train_state.y_opt_state)
            new_y = projection_simplex_truncated(train_state.y + grad, config.meta_trunc)

            # Collect trajectories on sampled levels
            # NOTE: we sample from the last iteration's y distribution for theoretical reasons
            #       ^although this may not perform as well emppirically, and there is active research
            #       ^on the properties of alternating TTSGDA
            rng, rng_levels = jax.random.split(rng)
            _, (_, rulesets) = level_sampler.sample_replay_levels({**train_state.sampler, "scores":train_state.y}, rng_levels, config.num_envs_per_device)

            meta_env_params = env_params.replace(ruleset=rulesets)

            reset_rng = jax.random.split(_rng3, num=config.num_envs_per_device)
            timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_rng)
            prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
            prev_reward = jnp.zeros(config.num_envs_per_device)

            outcomes = jnp.zeros((config.num_envs_per_device, 2))
            
            # INNER TRAIN LOOP
            def _update_step(runner_state, _):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, _):
                    rng, train_state, prev_timestep, prev_action, prev_reward, outcomes, prev_hstate = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    dist, value, hstate = train_state.apply_fn(
                        train_state.params,
                        {
                            # [batch_size, seq_len=1, ...]
                            "observation": prev_timestep.observation[:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                        },
                        prev_hstate,
                    )
                    action, log_prob = dist.sample_and_log_prob(seed=_rng)
                    # squeeze seq_len where possible
                    action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

                    # STEP ENV
                    timestep = jax.vmap(env.step, in_axes=0)(meta_env_params, prev_timestep, action)
                    success = timestep.discount == 0.0
                    outcomes = outcomes.at[:, 0].add(jnp.where(timestep.last(), 1, 0))
                    outcomes = outcomes.at[:, 1].add(jnp.where(success, 1, 0))
                    
                    transition = Transition(
                        # ATTENTION: done is always false, as we optimize for entire meta-rollout
                        done=jnp.zeros_like(timestep.last()),
                        ep_done=timestep.last(),
                        action=action,
                        value=value,
                        reward=timestep.reward,
                        log_prob=log_prob,
                        obs=prev_timestep.observation,
                        prev_action=prev_action,
                        prev_reward=prev_reward,
                    )
                    runner_state = (rng, train_state, timestep, action, timestep.reward, outcomes, hstate)
                    return runner_state, transition

                initial_hstate = runner_state[-1]
                # transitions: [seq_len, batch_size, ...]
                runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps_per_update)

                # CALCULATE ADVANTAGE
                rng, train_state, timestep, prev_action, prev_reward, outcomes, hstate = runner_state
                # calculate value of the last step for bootstrapping
                _, last_val, _ = train_state.apply_fn(
                    train_state.params,
                    {
                        "observation": timestep.observation[:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    hstate,
                )
                advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)

                # UPDATE NETWORK
                def _update_epoch(update_state, _):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, transitions, advantages, targets = batch_info
                        new_train_state, update_info = ppo_update_networks(
                            train_state=train_state,
                            transitions=transitions,
                            init_hstate=init_hstate.squeeze(1),
                            advantages=advantages,
                            targets=targets,
                            clip_eps=config.clip_eps,
                            vf_coef=config.vf_coef,
                            ent_coef=config.ent_coef,
                        )
                        return new_train_state, update_info

                    rng, train_state, init_hstate, transitions, advantages, targets = update_state

                    # MINIBATCHES PREPARATION
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                    # [seq_len, batch_size, ...]
                    batch = (init_hstate, transitions, advantages, targets)
                    # [batch_size, seq_len, ...], as our model assumes
                    batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                    shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                    # [num_minibatches, minibatch_size, ...]
                    minibatches = jtu.tree_map(
                        lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                    )
                    train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                    update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
                    return update_state, update_info

                # hstate shape: [seq_len=None, batch_size, num_layers, hidden_dim]
                update_state = (rng, train_state, initial_hstate[None, :], transitions, advantages, targets)
                update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
                # WARN: do not forget to get updated params
                rng, train_state = update_state[:2]

                # averaging over minibatches then over epochs
                loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
                ued_traj_batch = UEDTrajBatch(
                    ep_done=transitions.ep_done,
                    value=transitions.value,
                    reward=transitions.reward,
                    advantage=advantages,
                )
                runner_state = (rng, train_state, timestep, prev_action, prev_reward, outcomes, hstate)
                return runner_state, (loss_info, ued_traj_batch)

            # on each meta-update we reset rnn hidden to init_hstate
            runner_state = (rng, train_state, timestep, prev_action, prev_reward, outcomes, init_hstate)
            runner_state, (loss_info, transitions) = jax.lax.scan(_update_step, runner_state, None, config.num_inner_updates)
            rng, train_state = runner_state[:2]
            # WARN: do not forget to get updated params

            train_state = train_state.replace(
                sampler = new_sampler,
                y = new_y,
                y_opt_state = y_opt_state,
                num_updates=train_state.num_updates + 1,
            )
            
            outcomes = runner_state[-2]
            success_rate = outcomes.at[:, 1].get() / outcomes.at[:, 0].get()
            # EVALUATE AGENT
            eval_ruleset_rng, eval_reset_rng = jax.random.split(jax.random.key(config.eval_seed))
            eval_ruleset_rng = jax.random.split(eval_ruleset_rng, num=config.eval_num_envs_per_device)
            eval_reset_rng = jax.random.split(eval_reset_rng, num=config.eval_num_envs_per_device)

            eval_ruleset = jax.vmap(benchmark.sample_ruleset)(eval_ruleset_rng)
            eval_env_params = env_params.replace(ruleset=eval_ruleset)

            eval_stats = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None))(
                eval_reset_rng,
                env,
                eval_env_params,
                train_state,
                # TODO: make this a static method?
                jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                config.eval_num_episodes,
            )
            eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")

            ruleset_mean_num_rules = jnp.mean(jnp.sum(jnp.where(sampler["levels"].rules.at[:,0].get() > 0, 1, 0), axis=1))

            jax.lax.cond(
                update_idx % config.log_images_update == 0,
                lambda *_: jax.experimental.io_callback(log_levels, None, sampler, env_params, update_idx),
                lambda *_: None,
            )


            # averaging over inner updates, adding evaluation metrics
            loss_info = jtu.tree_map(lambda x: x.mean(-1), loss_info)
            loss_info.update(
                {
                    "eval/returns_mean": eval_stats.reward.mean(0),
                    "eval/returns_median": jnp.median(eval_stats.reward),
                    "eval/lengths": eval_stats.length.mean(0),
                    "eval/success_rate_mean": jnp.mean(eval_stats.success/eval_stats.episodes),
                    "eval/lengths_20percentile": jnp.percentile(eval_stats.length, q=20),
                    "eval/returns_20percentile": jnp.percentile(eval_stats.reward, q=20),
                    "ruleset_mean_num_rules": ruleset_mean_num_rules,
                    "outcomes": success_rate,
                    "num_env_steps": update_idx * config.num_inner_updates * config.num_steps_per_update * config.num_envs,
                    "update_step": update_idx,
                    **train_state_to_log_dict(train_state, level_sampler)
                }
            )
            
            def _callback(info):
                wandb.log(
                    info,
                    step=info["update_step"]
                )
            
            jax.experimental.io_callback(_callback, None, loss_info)
            
            meta_state = (rng, train_state)
            return meta_state, loss_info

        rng, _rng = jax.random.split(rng)
        ruleset_rng = jax.random.split(_rng, num=config.buffer_capacity)
        levels = jax.vmap(benchmark.sample_ruleset)(ruleset_rng)
        sampler = level_sampler.initialize(levels, {"max_return": jnp.full(config.buffer_capacity, -jnp.inf)})
        meta_state = (rng, train_state.replace(sampler = sampler))
        meta_state, loss_info = jax.lax.scan(_meta_step, meta_state, jnp.arange(config.num_meta_updates), config.num_meta_updates)
        return {"state": meta_state[1], "loss_info": loss_info}

    return train


@pyrallis.wrap()
def train(config: TrainConfig):
    # logging to wandb
    run = wandb.init(
        project=config.project,
        group=config.group,
        config=asdict(config),
        save_code=True,
        mode=config.mode,
    )
    # removing existing checkpoints if any
    if config.checkpoint_path is not None and os.path.exists(config.checkpoint_path):
        shutil.rmtree(config.checkpoint_path)

    rng, env, env_params, benchmark, level_sampler, init_hstate, train_state = make_states(config)
    # replicating args across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())

    print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, benchmark, level_sampler, config)
    train_fn = train_fn.lower(rng, train_state, init_hstate).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng, train_state, init_hstate, y_opt_state))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Logginig...")
    loss_info = unreplicate(train_info["loss_info"])

    run.summary["training_time"] = elapsed_time
    run.summary["steps_per_second"] = (config.total_timesteps_per_device * jax.local_device_count()) / elapsed_time

    if config.checkpoint_path is not None:
        params = train_info["state"].params
        save_dir = os.path.join(config.checkpoint_path, run.name)
        
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact   
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()
        checkpoint = {"config": asdict(config), "params": unreplicate(train_info)["state"].params}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(checkpoint)
        orbax_checkpointer.save(config.checkpoint_path, checkpoint, save_args=save_args)

    print("Final return: ", float(loss_info["eval/returns_mean"][-1]))
    run.finish()


if __name__ == "__main__":
    train()