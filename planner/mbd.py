import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit,static_argnums=(0, 1))
def central_mbd(args,cost_fn, x0, goal_loc, seed = 42, bound=False, input_bound = 0.0):
    """Model-Based Diffusion (MBD) planning function. Implementation is based on the paper "Model-Based Diffusion for Trajectory Optimization" by Pan et al. (2024)
    args: arguments for the planner
    cost_fn: function to compute the cost of a trajectory
    x0: initial states of all robots, shape (N_robots, state_dim)
    goal_loc: goal locations of all robots, shape (N_robots, state_dim)
    seed: random seed for sampling
    bound: whether to bound the control inputs
    input_bound: the bound for control inputs if bound is True
    """

    rng = jax.random.PRNGKey(seed=seed)

    YN = jnp.zeros((args.N, args.T, args.dim))

    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)


    def reverse_once(carry):
        i, rng, Yi = carry
        rng, Y0s_rng = jax.random.split(rng)

        samples = jax.random.normal(Y0s_rng, (args.Nsample_cent, args.N, args.T, args.dim))
        Y0s = samples * jnp.sqrt(1.0 / alphas_bar[i] - 1.0) + Yi/jnp.sqrt(alphas_bar[i])

        # Better without clipping, but can be added if desired
        Y0s = jnp.where(bound, jnp.clip(Y0s, -input_bound, input_bound), Y0s)

        rewss = jax.vmap(cost_fn, in_axes=(0, None, None))(Y0s, x0, goal_loc)
        rew_mean = rewss.mean(axis=-1)
        rew_std = rewss.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        logp0 = (rewss - rew_mean) / rew_std / args.temp_sample
   
        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("i,intd->ntd", weights, Y0s)    # Y0s is (Nsample_cent, N, T, dim) and weights is (Nsample_cent,) -> Ybar is (N, T, dim)

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        return (i - 1, rng, Yim1)


    # run reverse
    def reverse(YN, rng):
        indices = jnp.arange(args.Ndiffuse - 1, 0, -1)

        def scan_body(carry, i):
            rng, Yi = carry

            carry_once = (i, rng, Yi)
            _, rng, Yi = reverse_once(carry_once)

            return (rng, Yi), Yi

        (_, _), Ybars = jax.lax.scan(scan_body,(rng, YN), indices)
        return Ybars
    
    rng_exp, rng = jax.random.split(rng)
    Yi = reverse(YN, rng_exp)
    return Yi





