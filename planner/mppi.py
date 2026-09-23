import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit,static_argnums=(0, 1))
def mppi(args, cost_fn, x0, goal_loc, seed=42, bound=False, input_bound=0.0):
    """Model Predictive Path Integral (MPPI) planning function. Implementation is based on the paper "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving" by Williams et al. (2018)
    args: arguments for the planner
    cost_fn: function to compute the cost of a trajectory
    x0: initial states of all robots, shape (N_robots, state_dim)
    goal_loc: goal locations of all robots, shape (N_robots, state_dim)
    seed: random seed for sampling
    bound: whether to bound the control inputs
    input_bound: the bound for control inputs if bound is True
    """
    rng = jax.random.PRNGKey(seed)

    # Nominal control sequence
    U = jnp.zeros((args.N, args.T, args.dim))

    def mppi_step(carry, _):
        rng, U = carry

        rng, sample_rng = jax.random.split(rng)

        # perturbations
        eps = jax.random.normal(sample_rng, (args.Nsample_cent, args.N, args.T, args.dim))* args.sigma_mppi
        # candidate trajectories
        U_samples = U + eps

        U_samples = jnp.where(bound, jnp.clip(U_samples, -input_bound, input_bound), U_samples)

        # costs = - costs of the sampled control sequences, shape (Nsample_cent,)
        costs = -jax.vmap(cost_fn, in_axes=(0, None, None))(U_samples, x0, goal_loc)

        cost_std = jnp.where(costs.std() < 1e-4, 1.0, costs.std())
        # MPPI weights 
        weights = jax.nn.softmax(-(costs - jnp.min(costs)) / (cost_std * args.mppi_temp_sample))

        delta_U = jnp.einsum("i,intd->ntd", weights, eps)

        U_new = U + delta_U
        return (rng, U_new), U_new

    (_, U_final), U_hist = jax.lax.scan(
        mppi_step,
        (rng, U),
        None,
        length=args.Ndiffuse,
    )

    return U_hist




