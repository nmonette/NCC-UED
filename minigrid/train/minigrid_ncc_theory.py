"""
Nonconvex-Concave Regret Optimization for UED
"""

import json
import time
from functools import partial
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import os
import orbax.checkpoint as ocp
import wandb
import chex
from enum import IntEnum
import hydra
from omegaconf import OmegaConf

import operator

from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeSolved, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.level_sampler import LevelSampler as BaseLevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper

from sfl.util.jaxued.jaxued_utils import l1_value_loss
from sfl.train.train_utils import save_params
from sfl.train.minigrid_plr import (
    compute_gae,
    # TrainState,
    evaluate_rnn,
    # update_actor_critic_rnn,
    # ActorCritic,
    setup_checkpointing,
    compute_score
)
from sfl.util.ncc_utils import scale_y_by_ti_ada, ScaleByTiAdaState, ti_ada, projection_simplex_truncated

@struct.dataclass
class SolvedEnvParams:
    max_steps_in_episode: int = 250

def tree_sum(tree) -> chex.Numeric:
  """Compute the sum of all the elements in a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  sums = jax.tree_util.tree_map(jnp.sum, tree)
  return jax.tree_util.tree_reduce(operator.add, sums, initializer=0)

def _square(leaf):
  return jnp.square(leaf.real) + jnp.square(leaf.imag)

def tree_l2_norm(tree, squared: bool = False) -> chex.Numeric:
  """Compute the l2 norm of a pytree.

  Args:
    tree: pytree.
    squared: whether the norm should be returned squared or not.

  Returns:
    a scalar value.
  """
  squared_tree = jax.tree_util.tree_map(_square, tree)
  sqnorm = tree_sum(squared_tree)
  if squared:
    return sqnorm
  else:
    return jnp.sqrt(sqnorm)

from flax.core import FrozenDict
from typing import Callable, Any

class ActorCritic(nn.Module):
    """This is an actor critic class that uses an LSTM
    """
    action_dim: Sequence[int]
    is_recurrent: bool = True
    
    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.selu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.selu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.selu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)

class BestIterateState(TrainState):
    """TrainState that also tracks the parameters with the lowest gradient norm."""

    step: int 
    apply_fn: Callable = struct.field(pytree_node=False)
    params: FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    best_params: FrozenDict[str, Any] = struct.field(pytree_node=True)
    best_grad_norm: jax.Array = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        grad_norm = jnp.inf  # Initialize to infinity so the first update always sets it
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            best_params=params,
            best_grad_norm=grad_norm,
            **kwargs,
        )

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        # Compute the L2 norm of gradients
        grad_norm = tree_l2_norm(grads)

        # Update best params if this grad norm is smaller
        is_better = grad_norm < self.best_grad_norm
        new_best_params = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(is_better, x, y), self.params, self.best_params
        )
        new_best_grad_norm = jax.lax.select(is_better, grad_norm, self.best_grad_norm)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            best_params=new_best_params,
            best_grad_norm=new_best_grad_norm,
            **kwargs,
        )
    
def safe_norm(
    x,
    min_norm,
    ord = None, 
    axis = None,
    keepdims = False,
):
    """
    Taken from Optax utils
    """
    norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
    norm = jnp.squeeze(norm, axis=axis) if not keepdims else norm
    masked_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return jnp.where(norm <= min_norm, min_norm, masked_norm) 

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch
            
            def loss_fn(params):
                _, pi, _ = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                policy_loss = -(log_probs_pred * targets).sum(0).mean() 
                loss = policy_loss - entropy_coeff * entropy

                return loss, {"policy_loss": policy_loss, "entropy":entropy}

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (total_loss, losses), grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)

                # clip into (large) box constraints to ensure ipschitzness/stability as per https://openreview.net/pdf?id=Hygxb2CqKm
                train_state = train_state.replace(
                    params=jax.tree_util.tree_map(
                        lambda x: jnp.clip(x, min=-1000, max=1000), train_state.params
                    )
                )
                # NOTE: if you would like you can also divide x by safe_norm(x, 1e-6, axis=0, keepdims=True) if you would like 
                #       to improve stability
                
            return train_state, {**losses, "total_loss": total_loss, "grad_norm": tree_l2_norm(grads)}

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
    gamma: float = 1.0
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_hstate (chex.ArrayTree): This is the init RNN hidden state, has to have shape (NUM_ENVS, ...)
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, disc_return, disc_factor, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        disc_factor *= gamma
        carry = (rng, train_state, hstate, next_obs, env_state, disc_return + disc_factor * reward, disc_factor, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, disc_return, _, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=float),
            1.0,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0), disc_return), traj

def train_state_to_log_dict(train_state, level_sampler) -> dict:
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
        "info": {
            "num_dr_updates": 0.0,
            "num_replay_updates": 0.0,
            "num_mutation_updates": 0.0,
        }
    }

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

@hydra.main(version_base=None, config_path="config", config_name="minigrid-ncc-theory")
def main(config):

    config = OmegaConf.to_container(config)

    d = {}
    for k, v in config.items():
        if isinstance(v, dict):
            d = d | v
        else:
            d[k] = v
    config = d
    config["TOTAL_TIMESTEPS"] = int(config["TOTAL_TIMESTEPS"])

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    run = wandb.init(
        group=config.get('GROUP_NAME'),
        entity=config["ENTITY"],
        project=config["PROJECT"],
        config=config,
        mode=config["WANDB_MODE"],
        reinit=True
    )

    def log_eval(stats, train_state_info):
        print(f"Logging update")
        
        # generic stats
        log_dict = {}
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        best_eval_returns = stats["best_eval_returns"] 
        best_solve_rates = stats["best_eval_solve_rates"]
        log_dict.update({"solve_rate/best_iterate_mean": best_solve_rates.mean()})
        log_dict.update({"return/best_iterate_mean": best_eval_returns.mean()})
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["EVAL_LEVELS"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["EVAL_LEVELS"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        log_dict.update({"return/sampled_mean": stats['sampled_returns'][-1]})
        
        # level sampler
        log_dict.update(train_state_info["log"])

        log_dict.update({
            "policy_loss":stats["policy_loss"].mean(), "entropy":stats["entropy"].mean(), "total_loss": stats["total_loss"].mean(), "grad_norm": stats["grad_norm"].mean()
        })

        log_dict.update({
            "meta_entropy": stats["meta_entropy"].mean(),
            "meta_loss": stats["meta_loss"].mean(),
            "best_grad_norm": stats["best_grad_norm"][-1],
            "adv_grad_norm": stats["adv_grad_norm"].mean(),
        })

        # images
        log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest scoring level")})
        log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="Highest weighted level")})

        for s in ['dr', 'replay', 'mutation']:
            if train_state_info['info'][f'num_{s}_updates'] > 0:
                log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        # animations
        for i, level_name in enumerate(config["EVAL_LEVELS"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)


    NUM_WALLS = 13
    # Setup the environment
    env = Maze(max_height=NUM_WALLS, max_width=NUM_WALLS, agent_view_size=config["AGENT_VIEW_SIZE"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["N_WALLS"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params

    # And the level sampler    
    # TODO: maybe need to subclass the LevelSampler to get rid of staleness and such
    level_sampler = LevelSampler(
        capacity=config["PLR_PARAMS"]["capacity"],
        replay_prob=config["PLR_PARAMS"]["replay_prob"],
        staleness_coeff=config["PLR_PARAMS"]["staleness_coeff"],
        minimum_fill_ratio=config["PLR_PARAMS"]["minimum_fill_ratio"],
        prioritization=config["PLR_PARAMS"]["prioritization"],
        prioritization_params=config["PLR_PARAMS"]["prioritization_params"],
        duplicate_check=config["PLR_PARAMS"]['duplicate_check'],
    )

    def get_optimal_return(rng, levels, gamma, num_levels):
        env = MazeSolved(max_height=NUM_WALLS, max_width=NUM_WALLS, agent_view_size=config["AGENT_VIEW_SIZE"], normalize_obs=True)
        rng = jax.random.split(rng, num_levels)
        env_params = SolvedEnvParams()
        _, init_state = jax.vmap(env.reset_env_to_level, in_axes=(0, 0, None))(rng, levels, env_params)
        return jax.vmap(env.optimal_value, in_axes=(0, None, None))(init_state, gamma, env_params)

    @partial(jax.jit, static_argnums=(2, ))
    def learnability_fn(rng, levels, num_envs, train_state):
        def rollout_fn(rng):

            # Get the scores of the levels
            rng, _rng = jax.random.split(rng)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, num_envs), levels, env_params)
            # Rollout
            (
                (rng, _, hstate, last_obs, last_env_state, last_value, disc_return),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((num_envs,)),
                init_obs,
                init_env_state,
                num_envs,
                config["NUM_STEPS"],
                config["GAMMA"]
            )
            return disc_return
        
        rng, _rng = jax.random.split(rng)
        returns = jax.vmap(rollout_fn)(jax.random.split(_rng, config["EVAL_NUM_ATTEMPTS"]))

        rng, _rng = jax.random.split(rng)
        opt = get_optimal_return(_rng, levels, config["GAMMA"], num_envs)
        return opt - returns.mean(axis=0), returns.max(axis=0)

    def replace_fn(rng, train_state, old_level_scores):
        # NOTE: scores here are the actual UED scores, NOT the probabilities induced by the projection

        # Sample new levels
        rng, _rng = jax.random.split(rng)
        new_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["NUM_ENVS"]))

        rng, _rng = jax.random.split(rng)
        new_level_scores, max_returns = learnability_fn(_rng, new_levels, config["NUM_ENVS"], train_state)

        idxs = jnp.flipud(jnp.argsort(new_level_scores))

        new_levels = jax.tree_util.tree_map(
            lambda x: x[idxs], new_levels
        )
        new_level_scores = new_level_scores[idxs]

        update_sampler = {**train_state.sampler,"scores": old_level_scores}

        sampler, _ = level_sampler.insert_batch(update_sampler, new_levels, new_level_scores, {"max_return": max_returns})
        
        return sampler

    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config["LR"] * frac
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["NUM_ENVS"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["NUM_ENVS"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["NUM_ENVS"],)))

        tx = optax.chain(
                # optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                # optax.adagrad(learning_rate=linear_schedule)
                # optax.scale(-config["LR"])
                optax.scale_by_schedule(lambda t: -linear_schedule(t)),
        )
        rng, _rng = jax.random.split(rng)
        init_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["PLR_PARAMS"]["capacity"]))
        sampler = level_sampler.initialize(init_levels, {"max_return": jnp.full(config["PLR_PARAMS"]["capacity"], -jnp.inf)})
        return BestIterateState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
        )
    
    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):

        rng, train_state, xhat = carry

        # Use the old y so that x and y's updates are simultaneous
        new_score = xhat

        # Update the level sampler
        levels = train_state.sampler["levels"]
        rng, _rng = jax.random.split(rng)
        scores, _ = learnability_fn(_rng, levels, config["PLR_PARAMS"]["capacity"], train_state)

        def grad_fn(y):
            return y.T @ scores - config["META_REG"] * y.T @ jnp.log(y) # jnp.square(optax.safe_norm(x, 0, 2))
        
        grad = jax.grad(grad_fn)(xhat)
        xhat = projection_simplex_truncated(xhat + grad * config["META_LR"], config["META_TRUNC"])

        sampler = {**train_state.sampler, "scores": new_score}
        # Collect trajectories on replay levels
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["NUM_ENVS"])
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["NUM_ENVS"]), levels, env_params)
        (
            (rng, train_state, hstate, last_obs, last_env_state, last_value, disc_return),
            (obs, actions, rewards, dones, log_probs, values, info),
        ) = sample_trajectories_rnn(
            rng,
            env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((config["NUM_ENVS"],)),
            init_obs,
            init_env_state,
            config["NUM_ENVS"],
            config["NUM_STEPS"],
            config["GAMMA"]
        )
        advantages, _ = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
        
        def compute_returns(rewards, dones):

            def loop(returns, data):
                reward, done = data

                returns = reward + returns * config["GAMMA"] * (1 - done)
                return returns, returns

            return jax.lax.scan(loop, jnp.zeros_like(rewards[0]), (rewards, dones), reverse=True)[1]

        targets = compute_returns(rewards, dones)
        # Update the policy using trajectories collected from replay levels
        (rng, train_state), losses = update_actor_critic_rnn(
            rng,
            train_state,
            ActorCritic.initialize_carry((config["NUM_ENVS"],)),
            (obs, actions, dones, log_probs, values, targets, advantages),
            config["NUM_ENVS"],
            config["NUM_STEPS"],
            config["NUM_MINIBATCHES"],
            config["UPDATE_EPOCHS"],
            config["CLIP_EPS"],
            config["ENT_COEF"],
            config["VF_COEF"],
            update_grad=True,
        )

        losses = jax.tree_map(lambda x: x.mean(), losses)
        metrics = {
            **losses,
            "mean_num_blocks": levels.wall_map.sum() / config["NUM_ENVS"],
            "meta_entropy": -jnp.dot(sampler["scores"], jnp.log(sampler["scores"] + 1e-6)),
            "meta_loss": grad_fn(new_score),
            "best_grad_norm": train_state.best_grad_norm,
            "adv_grad_norm": jnp.linalg.norm(new_score - projection_simplex_truncated(new_score + config["META_LR"] * jax.grad(grad_fn)(new_score), config["META_TRUNC"]))
        }

        return (rng, train_state, xhat), metrics
    
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["EVAL_LEVELS"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["EVAL_LEVELS"])
        num_levels = len(config["EVAL_LEVELS"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    def eval_sampled_levels(rng: chex.PRNGKey, train_state: TrainState):
        num_levels = config["NUM_ENVS"]
        rng, _rng, rng_reset = jax.random.split(rng, 3)
        levels = jax.vmap(sample_random_level)(jax.random.split(_rng, num_levels))
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
            This function runs the train_step for a certain number of iterations, and then evaluates the policy.
            It returns the updated train state, and a dictionary of metrics.
        """
        # Train
        (rng, train_state, xhat), metrics = jax.lax.scan(train_step, runner_state, None, config["EVAL_FREQ"])

        # Eval (BEST)
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["EVAL_NUM_ATTEMPTS"]), train_state.replace(params = train_state.best_params))
        
        # Collect Metrics
        best_eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        best_eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)

        # Eval 
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["EVAL_NUM_ATTEMPTS"]), train_state)
        
        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["best_eval_returns"] = best_eval_returns
        metrics["best_eval_solve_rates"] = best_eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        # metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
        
        # Eval (Random Levels)
        rng, rng_eval = jax.random.split(rng)
        _, cum_rewards, _ = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["EVAL_NUM_ATTEMPTS"]), train_state)
        metrics["sampled_returns"] = cum_rewards.mean(axis=0)

        highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax())
        
        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)
        
        return (rng, train_state, xhat), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["EVAL_NUM_ATTEMPTS"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['EVAL_LEVELS'])
        return states, cum_rewards, episode_lengths
    
    # Set up the train states
    rng = jax.random.PRNGKey(config["SEED"])
    rng_init, rng_train = jax.random.split(rng)
    train_state = create_train_state(rng_init)

    xhat = grad = jnp.full_like(train_state.sampler["scores"], 1 / len(train_state.sampler["scores"]), dtype=jnp.float32)
    runner_state = (rng_train, train_state, xhat)
    
    # And run the train_eval_sep function for the specified number of updates
    if config["CHECKPOINT_SAVE_INTERVAL"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["NUM_UPDATES"] // config["EVAL_FREQ"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["CHECKPOINT_SAVE_INTERVAL"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    if config["SAVE_PATH"] is not None:
        params = runner_state[1].params
        
        save_dir = os.path.join(config["SAVE_PATH"], wandb.run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact   
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()

    return runner_state[1]

if __name__=="__main__":
    main()