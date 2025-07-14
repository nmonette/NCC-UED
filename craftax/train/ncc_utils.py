import jax
import jax.numpy as jnp
from flax.struct import dataclass
import optax
import chex
from optax._src import numerics


from functools import partial
import operator

"""THE FOLLOWING IS JUST COPIED FROM OPTAX TREE UTILS"""

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

def tree_update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_util.tree_map(
      lambda g, t: (
          (1 - decay) * (g**order) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )

def tree_update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g**order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return numerics.abs_sq(g) ** half_order

  return jax.tree_util.tree_map(
      lambda g, t: (
          (1 - decay) * orderth_norm(g) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )



@dataclass
class ScaleByTiAdaState:
    vx: float | None
    vy: float

    # if doing Adam + ScaleByTiAdaState
    prev_grad: dict | jnp.ndarray = None
    exp_b1: float | None = 1.0
    exp_b2: float | None = 1.0

    # for amssgrad version
    nu_max: dict  = None


def scale_x_by_ti_ada(
    vx0: float = 0.1,
    vy0: float = 0.1, # just pass in a zeros_like y_params
    eta: float = 1e-4,
    alpha: float = 0.6,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-5,
    amsgrad: bool = False
):
    """
    https://openreview.net/pdf?id=zClyiZ5V6sL 
    assumes we are doing the adam version
    """
    def init_fn(params):
        vx = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        prev_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

        nu_max = None
        if amsgrad:
            nu_max = prev_grad

        return ScaleByTiAdaState(vx, vy0, prev_grad = prev_grad, nu_max = nu_max)
    
    def update_fn(x_updates, state, params=None):

        grad = tree_update_moment(x_updates, state.prev_grad, b1, 1)
        vx = tree_update_moment_per_elem_norm(x_updates, state.vx, b2, 2)
        
        exp_b1 = state.exp_b1 * b1
        exp_b2 = state.exp_b2 * b2

        total_sum_vx = tree_sum(vx)
        total_sum_vy = state.vy.sum()

        ratio = jax.lax.pow(total_sum_vx, alpha) / jax.lax.pow(jax.lax.max(total_sum_vx, total_sum_vy), alpha)

        coeff = jax.tree_util.tree_map(
            lambda v: eta / (jax.lax.pow(v / jnp.sqrt(1 - exp_b2), alpha) + eps), vx
        )

        nu_max = None
        if amsgrad:
            nu_max = jax.tree_util.tree_map(
                lambda state_nu, nu: jnp.maximum(state_nu, nu / (1 - exp_b2)), state.nu_max, vx
            )
            coeff = jax.tree_util.tree_map(
                lambda v: eta / (jax.lax.pow(v, alpha) + eps), nu_max# $ lambda v: eta / (jax.lax.pow(v, alpha) / jnp.sqrt(1 - exp_b2) + eps), vx
            )

        bias_corrected_grad = jax.tree_util.tree_map(lambda m: m / (1 - exp_b1), grad)
    
        x_grad = jax.tree_util.tree_map(
            lambda m, c: ratio * c * m, bias_corrected_grad, coeff
        )

        new_state = ScaleByTiAdaState(
            vx, 
            state.vy, 
            prev_grad=grad,
            exp_b1=exp_b1,
            exp_b2=exp_b2,
            nu_max = nu_max
        )

        return x_grad, new_state

    return optax.GradientTransformation(init_fn, update_fn)

def ti_ada(
    vx0: float = 0.1,
    vy0: float = 0.1, # just pass in a zeros_like y_params
    eta: float = 1e-4,
    alpha: float = 0.6,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-5, 
    amsgrad: bool = False
):
    return optax.chain(
        scale_x_by_ti_ada(vx0, vy0, 1.0, alpha, b1, b2, eps, amsgrad),
        optax.scale(-eta) if isinstance(eta, float) else optax.scale_by_schedule(lambda t: -eta(t)) 
    )

    
def projection_simplex_truncated(x: jnp.ndarray, eps: float) -> jnp.ndarray: 
    """
    Code adapted from 
    https://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html
    To represent truncated simplex projection. Assumes 1D vector. 
    """
    ones = jnp.ones_like(x)
    lambdas = jnp.concatenate((ones * eps - x, ones - x), axis=-1)
    idx = jnp.argsort(lambdas)
    lambdas = jnp.take_along_axis(lambdas, idx, -1)
    active = jnp.cumsum((jnp.float32(idx < x.shape[-1])) * 2 - 1, axis=-1)[..., :-1]
    diffs = jnp.diff(lambdas, n=1, axis=-1)
    left = (ones * eps).sum(axis=-1)
    left = left.reshape(*left.shape, 1)
    totals = left + jnp.cumsum(active*diffs, axis=-1)

    def generate_vmap(counter, func):
        if counter == 0:
            return func
        else:
            return generate_vmap(counter - 1, jax.vmap(func))
                
    i = jnp.expand_dims(generate_vmap(len(totals.shape) - 1, partial(jnp.searchsorted, v=1))(totals), -1)
    lam = (1 - jnp.take_along_axis(totals, i, -1)) / jnp.take_along_axis(active, i, -1) + jnp.take_along_axis(lambdas, i+1, -1)
    return jnp.clip(x + lam, eps, 1)

@dataclass
class ScaleByRssState:
    """State holding the sum of gradient squares to date."""

    sum_of_squares: dict


def abs_sq(x: chex.Array) -> chex.Array:
    return (x.conj() * x).real


def scale_by_rss(initial_accumulator_value: float = 0.1, eps: float = 1e-7): 
    def init_fn(params):
        return ScaleByRssState(
            sum_of_squares=jax.tree_util.tree_map(lambda x: jnp.full_like(x, initial_accumulator_value), params)
        )

    def update_fn(updates, state, params=None):
        del params
        sum_of_squares = jax.tree_util.tree_map(
            lambda g, t: abs_sq(g) + t, updates, state.sum_of_squares
        )
        inv_sqrt_g_square = jax.tree_util.tree_map(
            lambda t: jnp.where(t > 0, jax.lax.pow(t + eps, -0.4), 0.0), sum_of_squares
        )
        updates = jax.tree_util.tree_map(
            lambda x,y: x * y, inv_sqrt_g_square, updates
        )
        return updates, ScaleByRssState(sum_of_squares=sum_of_squares)

    return optax.GradientTransformation(init_fn, update_fn)

def scale_y_by_ti_ada(
    learning_rate: float = 1e-2
):
    return optax.chain(
        scale_by_rss(),
        optax.scale(-learning_rate) if isinstance(learning_rate, float) else optax.scale_by_schedule(lambda t: -learning_rate(t))
    )