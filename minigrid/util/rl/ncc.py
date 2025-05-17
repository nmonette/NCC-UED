"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import struct
import chex
import numpy as np

from .ued_scores import UEDScore
from ..ncc_utils import projection_simplex_truncated


class PLRBuffer(struct.PyTreeNode):
	levels: chex.Array
	scores: chex.Array
	ages: chex.Array
	max_returns: chex.Array # for MaxMC
	filled: chex.Array
	filled_count: chex.Array
	n_mutations: chex.Array

	ued_score: int = struct.field(pytree_node=False, default=UEDScore.L1_VALUE_LOSS.value)
	replay_prob: float = struct.field(pytree_node=False, default=0.5)
	meta_lr: float = struct.field(pytree_node=False, default=1e-4)
	meta_trunc: float = struct.field(pytree_node=False, default=1e-6)
	buffer_size: int = struct.field(pytree_node=False, default=100)
	staleness_coef: float = struct.field(pytree_node=False, default=0.3)
	temp: float = struct.field(pytree_node=False, default=1.0)
	use_score_ranks: bool = struct.field(pytree_node=False, default=True)
	min_fill_ratio: float = struct.field(pytree_node=False, default=0.5)
	use_robust_plr: bool = struct.field(pytree_node=False, default=False)
	use_parallel_eval: bool = struct.field(pytree_node=False, default=False)


class PLRManager:
	def __init__(
		self,
		example_level, # Example env instance
		ued_score,
		replay_prob=0.5,
		buffer_size=100,
		staleness_coef=0.3,
		temp=1.0,
		min_fill_ratio=0.5,
		use_score_ranks=True,
		use_robust_plr=False,
		use_parallel_eval=False,
		comparator_fn=None,
		n_devices=1):

		assert not (ued_score == UEDScore.MAX_MC and not use_score_ranks), \
			'Cannot use proportional normalization with MaxMC, which can produce negative scores.'

		self.ued_score = ued_score
		self.replay_prob = replay_prob
		self.buffer_size = buffer_size
		self.staleness_coef = staleness_coef
		self.temp = temp
		self.min_fill_ratio = min_fill_ratio
		self.use_score_ranks = use_score_ranks
		self.use_robust_plr = use_robust_plr
		self.use_parallel_eval = use_parallel_eval
		self.comparator_fn = comparator_fn

		self.n_devices = n_devices

		example_level = jax.tree_map(lambda x: jnp.array(x), example_level)
		self.levels = jax.tree_map(
			lambda x: (
				jnp.tile(jnp.zeros_like(x), (buffer_size,) + (1,)*(len(x.shape)-1))).reshape(buffer_size, *x.shape),
			example_level)

		self.scores = jnp.full(buffer_size, -jnp.inf)
		self.max_returns = jnp.full(buffer_size, -jnp.inf)
		self.ages = jnp.zeros(buffer_size, dtype=jnp.uint32)
		self.filled = jnp.zeros(buffer_size, dtype=jnp.bool_)
		self.filled_count = jnp.zeros((1,), dtype=jnp.int32)
		self.n_mutations = jnp.zeros(buffer_size, dtype=jnp.uint32)


	partial(jax.jit, static_argnums=(0,))
	def reset(self):
		return PLRBuffer(
			ued_score=self.ued_score.value,
			replay_prob=self.replay_prob,
			buffer_size=self.buffer_size,
			staleness_coef=self.staleness_coef,
			temp=self.temp,
			min_fill_ratio=self.min_fill_ratio,
			use_robust_plr=self.use_robust_plr,
			use_parallel_eval=self.use_parallel_eval,
			levels=self.levels,
			scores=self.scores,
			max_returns=self.max_returns,
			ages=self.ages,
			filled=self.filled,
			filled_count=self.filled_count,
			n_mutations=self.n_mutations)

	partial(jax.jit, static_argnums=(0,))
	def _get_replay_dist(self, scores, ages, filled):
		# Score dist
		return scores

	partial(jax.jit, static_argnums=(0,))
	def _get_next_insert_idx(self, plr_buffer):
		return jax.lax.cond(
			jnp.greater(plr_buffer.buffer_size, plr_buffer.filled_count[0]), 
			lambda *_: plr_buffer.filled_count[0], 
			lambda *_: jnp.argmin(self._get_replay_dist(plr_buffer.scores, plr_buffer.ages, plr_buffer.filled))
		)
 
	@partial(jax.jit, static_argnums=(0,3))
	def _sample_replay_levels(self, rng, plr_buffer, n):
		def _sample_replay_level(carry, step):
			ages = carry
			subrng = step
			replay_dist = self._get_replay_dist(plr_buffer.scores, ages, plr_buffer.filled)
			replay_idx = jax.random.choice(subrng, np.arange(self.buffer_size), shape=(), p=replay_dist)
			replay_level = jax.tree_map(lambda x: x.take(replay_idx, axis=0), plr_buffer.levels)

			ages = ((ages + 1)*(plr_buffer.filled)).at[replay_idx].set(0)

			return ages, (replay_level, replay_idx)

		rng, *subrngs = jax.random.split(rng, n+1)
		next_ages, (replay_levels, replay_idxs) = jax.lax.scan(
			_sample_replay_level,
			plr_buffer.ages,
			jnp.array(subrngs)
		)

		next_plr_buffer = plr_buffer.replace(
			ages=next_ages
		)

		return replay_levels, replay_idxs, next_plr_buffer

	def _sample_buffer_uniform(self, rng, plr_buffer, n):
		rand_idxs = jax.random.choice(rng, np.arange(self.buffer_size), shape=(n,), p=plr_buffer.filled)
		levels = jax.tree_map(lambda x: x.take(rand_idxs, axis=0), plr_buffer.levels)

		return levels, rand_idxs, plr_buffer

	# Levels must be sampled sequentially, to account for staleness
	@partial(jax.jit, static_argnums=(0,4,5))
	def sample(self, rng, plr_buffer: PLRBuffer, new_levels, n: int, random=False):
		"""Sample levels from the plr buffer

		Args:
			rng (_type_): _description_
			plr_buffer (PLRBuffer): current plr buffer
			new_levels (_type_): _description_
			n (int): number of levels to sample
			random (bool, optional): _description_. Defaults to False.

		Returns:
			_type_: _description_
		"""
		rng, replay_rng, sample_rng = jax.random.split(rng, 3)
		levels, level_idxs, next_plr_buffer = self._sample_replay_levels(sample_rng, plr_buffer, n=n)

		return levels, level_idxs, True, next_plr_buffer

	@partial(jax.jit, static_argnums=(0,))
	def dedupe_levels(self, plr_buffer, levels, level_idxs):
		if self.comparator_fn is not None and level_idxs.shape[-1] > 2:
			def _check_equal(carry, step):
				match_idxs, other_levels, is_self  = carry
				batch_idx, level = step

				matches = jax.vmap(self.comparator_fn, in_axes=(0,None))(other_levels, level)

				top2match, top2match_idxs = jax.lax.top_k(matches,2)

				is_self_dupe = jnp.logical_and(is_self, top2match[1]) # More than 1 match
				is_dedupe_idx = jnp.logical_and(is_self_dupe, jnp.greater(batch_idx, top2match_idxs[0]))
				self_match_idx = top2match_idxs[0]*is_dedupe_idx - (~is_dedupe_idx)

				_match_idx = jnp.where(
					is_self,
					self_match_idx, # only first
					top2match_idxs[0], # use first matching index in buffer
				)

				match_idxs = jnp.where(
					matches.any(),
					match_idxs.at[batch_idx].set(_match_idx),
					match_idxs
				)

				return (match_idxs, other_levels, is_self), None

			# dedupe among batch levels
			batch_dupe_idxs = jnp.full_like(level_idxs, -1)
			(batch_dupe_idxs, _, _), _ = jax.lax.scan(
				_check_equal,
				(batch_dupe_idxs, levels, True),
				(np.arange(level_idxs.shape[-1]), levels)
			)
			batch_dupe_mask = jnp.greater(batch_dupe_idxs, -1)

			# dedupe against PLR buffer levels
			(level_idxs, _, _), _ = jax.lax.scan(
				_check_equal,
				(level_idxs, plr_buffer.levels, False),
				(np.arange(level_idxs.shape[-1]), levels)
			)

			return level_idxs, batch_dupe_mask
		else:
			return level_idxs, jnp.zeros_like(level_idxs, dtype=jnp.bool_)

	partial(jax.jit, static_argnums=(0,7))	
	def update(self, plr_buffer, levels, level_idxs, ued_scores, dupe_mask=None, info=None, ignore_val=-jnp.inf, parent_idxs=None):
		# Note: parent_idxs are only used for mutated levels
		next_plr_buffer = plr_buffer.replace(
			scores = projection_simplex_truncated(plr_buffer.scores + plr_buffer.meta_lr*ued_scores, plr_buffer.meta_trunc)
        )

		return next_plr_buffer

	@partial(jax.jit, static_argnums=(0,))
	def get_metrics(self, plr_buffer):
		replay_dist = self._get_replay_dist(plr_buffer.scores, plr_buffer.ages, plr_buffer.filled)
		weighted_n_mutations = (plr_buffer.n_mutations*replay_dist).sum()
		scores = plr_buffer.scores
		weighted_ued_score = (scores*replay_dist).sum()

		weighted_age = (plr_buffer.ages*replay_dist).sum()

		return dict(
			weighted_n_mutations=weighted_n_mutations,
			weighted_ued_score=weighted_ued_score,
			weighted_age=weighted_age
		)
	
	@partial(jax.jit, static_argnums=(0, 2))
	def get_score_extremes(self, plr_buffer: PLRBuffer, num_samples: int=10):
		sorted_idxs = jnp.argsort(plr_buffer.scores)
  
		# Get top and bottom scores
		top_idxs = sorted_idxs.at[-num_samples:].get()
		bottom_idxs = sorted_idxs.at[:num_samples].get()
		idxs = jnp.concatenate([top_idxs, bottom_idxs])
		
		info = {
			"levels": jax.tree_map(lambda x: x.at[idxs].get(), plr_buffer.levels),
			"scores": plr_buffer.scores.at[idxs].get(),
			"ages": plr_buffer.ages.at[idxs].get(),
		}
		return info

	@partial(jax.jit, static_argnums=(0,))
	def get_buffer_distribution(self, plr_buffer: PLRBuffer):
     
		# Score dist
		scores = plr_buffer.scores

		if self.use_score_ranks:
			sorted_idx = jnp.argsort(-scores) # Top first
			scores = jnp.zeros(self.buffer_size, dtype=jnp.int32)\
					.at[sorted_idx]\
					.set(1/jnp.arange(self.buffer_size))
			
		scores = scores*plr_buffer.filled
		score_dist = scores/self.temp
		z = score_dist.sum()
		z = jnp.where(jnp.equal(z,0),1,z)
		score_dist = jax.lax.select(
			jnp.greater(z, 0),
			score_dist/z,
			plr_buffer.filled*1. # Assign equal weight to all present levels
		)

		# Staleness dist
		staleness_scores = plr_buffer.ages*plr_buffer.filled
		_z = staleness_scores.sum()
		z = jnp.where(jnp.equal(_z,0),1,_z)
		staleness_dist = jax.lax.select(
			jnp.greater(_z, 0),
			staleness_scores/z,
			score_dist # If no solutions are stale, do not sample from staleness dist
		)

		# Replay dist
		replay_dist = (1-self.staleness_coef)*score_dist \
			+ self.staleness_coef*staleness_dist
     
		order = jnp.argsort(plr_buffer.scores)
		replay_dist = self._get_replay_dist(plr_buffer.scores, plr_buffer.ages, plr_buffer.filled)
		return {
			"order": order,
			"replay_dist": replay_dist.at[order].get(),
			"scores": plr_buffer.scores.at[order].get(),
			"ages": plr_buffer.ages.at[order].get(),
			"score_dist": score_dist.at[order].get(),
			"staleness_dist": staleness_dist.at[order].get(),
		}
  
	

     


class PopPLRManager(PLRManager):
	def __init__(self, *, n_agents, **kwargs):
		super().__init__(**kwargs)

		self.n_agents = n_agents

	@partial(jax.jit, static_argnums=(0,1))
	def reset(self, n):
		sup = super()
		return jax.vmap(lambda *_: sup.reset())(np.arange(n))

	partial(jax.jit, static_argnums=(0,4,5))
	def sample(self, rng, plr_buffer, new_levels, n, random=False):
		sup = super()
		
		rng, *vrngs = jax.random.split(rng, self.n_agents+1)

		return jax.vmap(sup.sample, in_axes=(0,0,0,None,None))(
			jnp.array(vrngs), plr_buffer, new_levels, n, random
		)

	@partial(jax.jit, static_argnums=(0,))
	def dedupe_levels(self, plr_buffer, levels, level_idxs):
		sup = super()
		return jax.vmap(sup.dedupe_levels)(plr_buffer, levels, level_idxs)

	partial(jax.jit, static_argnums=(0,7))
	def update(self, plr_buffer, levels, level_idxs, ued_scores, dupe_mask=None, info=None, ignore_val=-jnp.inf, parent_idxs=None):
		sup = super()
		return jax.vmap(sup.update, in_axes=(0,0,0,0,0,0,None,0))(
			plr_buffer, levels, level_idxs, ued_scores, dupe_mask, info, ignore_val, parent_idxs
		)

	partial(jax.jit, static_argnums=(0,))
	def get_metrics(self, plr_buffer):
		sup = super()
		return jax.vmap(sup.get_metrics)(plr_buffer)
