from collections import namedtuple
from typing import Any, Callable, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jax import Array, device_put, grad, jit, vmap
from jax.tree_util import tree_flatten, tree_map, tree_unflatten, Partial
from jax.nn import relu
from jax.typing import ArrayLike


from utils import progress_bar_scan, tree_random_normal_like

jax.config.update("jax_log_compiles", False)


LangevinState = namedtuple("LangevinState", ["key", "x"])
PrimalDualState = namedtuple("PrimalDualState", ["key", "x", "lmbda", "nu"])
AugmentedLagrangianState = namedtuple("AugmentedLagrangianState", ["key", "x", "lmbda", "nu", "c"])
ConstraintFn = Callable[[ArrayLike | Any], Array | Any]
PotentialFn = Callable[[ArrayLike | Any], float]
ProjectionFn = Callable[[ArrayLike | Any], Array | Any]


def projlmc_step(
    state: LangevinState, grad_f, proj, step_size
) -> Tuple[LangevinState, LangevinState]:
    key, x = state
    key, gauss = jr.split(key, 2)
    g = grad_f(x)
    x = tree_map(
        lambda p, g: p - step_size * g + jnp.sqrt(2 * step_size) * jr.normal(gauss, p.shape),
        x,
        g,
    )
    x = proj(x)
    new_state = device_put(LangevinState(key, x))

    return new_state, new_state


def lmc_run_chain(
    initial_key: jax.Array,
    f: PotentialFn,
    k: int,
    step_size: float,
    initial_point: ArrayLike | Any,
) -> LangevinState:

    grad_f = jit(grad(f))
    step = Partial(projlmc_step, grad_f=grad_f, proj=lambda x: x, step_size=step_size)

    init_state = device_put(LangevinState(initial_key, initial_point))
    _, traj = lax.scan(lambda x, _: step(x), init_state, None, k)

    return traj


def projlmc_run_chain(
    initial_key: jax.Array,
    f: PotentialFn,
    proj: ProjectionFn,
    k: int,
    step_size: float,
    initial_point: ArrayLike | Any,
) -> LangevinState:

    grad_f = jit(grad(f))
    step = Partial(projlmc_step, grad_f=grad_f, proj=proj, step_size=step_size)

    init_state = device_put(LangevinState(initial_key, initial_point))
    _, traj = lax.scan(lambda x, _: step(x), init_state, None, k)

    return traj


def langevin_step(
    state: LangevinState, grad_u, proj, step_size
) -> Tuple[LangevinState, LangevinState]:
    key, x = state

    key, normals = tree_random_normal_like(key, x)

    g = grad_u(x)
    x = tree_map(
        lambda p, g, r: p - step_size * g + jnp.sqrt(2 * step_size) * r,
        x,
        g,
        normals,
    )
    x = proj(x)
    new_state = LangevinState(key, x)
    return new_state, new_state


def create_langevin_step(grad_u, proj, step_size):
    def langevin_step_wrapper(state: LangevinState, _) -> Tuple[LangevinState, LangevinState]:
        return langevin_step(state, grad_u, proj, step_size)
    return langevin_step_wrapper


def pdlmc_run_chain(
    initial_key: jax.Array,
    f: PotentialFn,
    g: ConstraintFn,
    h: ConstraintFn,
    iterations: int,
    lmc_steps: int,
    burnin: int,
    step_size_x: float,
    step_size_lmbda: float,
    step_size_nu: float,
    initial_x: ArrayLike | Any,
    initial_lmbda: ArrayLike | Any,
    initial_nu: ArrayLike | Any,
    proj: ProjectionFn = lambda x: x,
) -> Tuple[jax.Array, PrimalDualState]:

    u = jit(lambda x, lmbda, nu: f(x) + jnp.inner(lmbda, g(x)) + jnp.inner(nu, h(x)))
    grad_u = jit(grad(u))

    @progress_bar_scan(iterations)
    def pd_step(state: PrimalDualState, iter_num) -> Tuple[PrimalDualState, PrimalDualState]:
        key, x, lmbda, nu = state
        grad_u_lang = Partial(grad_u, lmbda=lmbda, nu=nu)

        step = create_langevin_step(grad_u_lang, proj, step_size_x)

        key, lang_key = jr.split(key, 2)

        init_state = device_put(LangevinState(lang_key, x))
        _, pdlmc_traj = lax.scan(step, init_state, None, lmc_steps + burnin)

        flat, tree = tree_flatten(pdlmc_traj.x)
        after_burnin = tree_map(lambda x: x[burnin:], flat)
        after_burnin_tr = tree_unflatten(tree, after_burnin)
        last_iterate = tree_map(lambda x: x[-1], flat)
        last_iterate_tr = tree_unflatten(tree, last_iterate)

        qlmbda = vmap(g)(after_burnin_tr).mean(axis=0)
        qnu = vmap(h)(after_burnin_tr).mean(axis=0)

        new_state = device_put(
            PrimalDualState(
                key,
                last_iterate_tr,
                relu(lmbda + step_size_lmbda * qlmbda),
                nu + step_size_nu * qnu,
            )
        )
        return new_state, new_state

    init_state = device_put(PrimalDualState(initial_key, initial_x, initial_lmbda, initial_nu))
    final_state, pd_traj = lax.scan(pd_step, init_state, jnp.arange(iterations), iterations)

    return final_state.key, pd_traj


def allmclc_run_chain(
    initial_key: jax.Array,
    f: PotentialFn,
    g: ConstraintFn,
    h: ConstraintFn,
    iterations: int,
    lmc_steps: int,
    burnin: int,
    step_size_x: float,
    step_size_lmbda: float,
    step_size_nu: float,
    tracker_rate: float,
    penalty_rho: float,
    initial_x: ArrayLike | Any,
    initial_lmbda: ArrayLike | Any,
    initial_nu: ArrayLike | Any,
    initial_c: ArrayLike | Any,
    proj: ProjectionFn = lambda x: x,
) -> Tuple[jax.Array, AugmentedLagrangianState]:
    """
    Augmented Lagrangian Langevin Monte Carlo (AL-LMC) sampling.
    
    Args:
        initial_key: Random key for sampling
        f: Potential function (negative log posterior)
        g: Inequality constraint function
        h: Equality constraint function  
        iterations: Number of AL-LMC iterations
        lmc_steps: Number of Langevin steps per iteration
        burnin: Burnin steps for each Langevin chain
        step_size_x: Step size for primal variables
        step_size_lmbda: Step size for inequality dual variables
        step_size_nu: Step size for equality dual variables
        tracker_rate: Exponential averaging rate for constraint tracking (β_k)
        penalty_rho: Penalty parameter for augmented Lagrangian
        initial_x: Initial primal variables
        initial_lmbda: Initial inequality dual variables
        initial_nu: Initial equality dual variables
        initial_c: Initial constraint tracker
        proj: Projection function for primal variables
        
    Returns:
        Tuple of (final_key, trajectory)
    """

    # Augmented Lagrangian potential: f(x) + λᵀg(x) + νᵀh(x) + (ρ/2)||g(x)||² + (ρ/2)||h(x)||²
    u_aug = jit(lambda x, lmbda, nu: (
        f(x) + jnp.inner(lmbda, g(x)) + jnp.inner(nu, h(x)) +
        (penalty_rho / 2) * (jnp.sum(g(x)**2) + jnp.sum(h(x)**2))
    ))
    grad_u_aug = jit(grad(u_aug))

    @progress_bar_scan(iterations)
    def al_step(state: AugmentedLagrangianState, iter_num) -> Tuple[AugmentedLagrangianState, AugmentedLagrangianState]:
        key, x, lmbda, nu, c = state
        grad_u_lang = Partial(grad_u_aug, lmbda=lmbda, nu=nu)

        step = create_langevin_step(grad_u_lang, proj, step_size_x)

        key, lang_key = jr.split(key, 2)

        init_state = device_put(LangevinState(lang_key, x))
        _, allmclc_traj = lax.scan(step, init_state, None, lmc_steps + burnin)

        flat, tree = tree_flatten(allmclc_traj.x)
        after_burnin = tree_map(lambda x: x[burnin:], flat)
        after_burnin_tr = tree_unflatten(tree, after_burnin)
        last_iterate = tree_map(lambda x: x[-1], flat)
        last_iterate_tr = tree_unflatten(tree, last_iterate)

        # Compute constraint values (empirical average over chain samples)
        g_bar = vmap(g)(after_burnin_tr).mean(axis=0)
        h_bar = vmap(h)(after_burnin_tr).mean(axis=0)

        # Update constraint tracker: c_{k+1} = (1-β_k)c_k + β_k * g_bar
        new_c_g = (1 - tracker_rate) * c[:len(g_bar)] + tracker_rate * g_bar
        new_c_h = (1 - tracker_rate) * c[len(g_bar):] + tracker_rate * h_bar
        new_c = jnp.concatenate([new_c_g, new_c_h])

        # Dual variable updates using g_bar (not c)
        new_state = device_put(
            AugmentedLagrangianState(
                key,
                last_iterate_tr,
                relu(lmbda + step_size_lmbda * g_bar),
                nu + step_size_nu * h_bar,
                new_c,
            )
        )
        return new_state, new_state

    init_state = device_put(AugmentedLagrangianState(initial_key, initial_x, initial_lmbda, initial_nu, initial_c))
    final_state, al_traj = lax.scan(al_step, init_state, jnp.arange(iterations), iterations)

    return final_state.key, al_traj
