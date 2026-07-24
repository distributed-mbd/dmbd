import jax 
import jax.numpy as jnp
from functools import partial


@partial(jax.jit,static_argnums=(0, 1))
def d4orm(args, cost_fn, x0, goal_loc, seed = 42, bound=False, input_bound = 0.0):
    """D4orm planning function. Implementation is based on the paper "D4orm: Multi-Robot Trajectories with Dynamics-Aware Diffusion Denoised Deformations" by Zhang et al. (2025)
    args: arguments for the planner
    cost_fn: function to compute the cost of a trajectory
    x0: initial states of all robots, shape (N_robots, state_dim)
    goal_loc: goal locations of all robots, shape (N_robots, state_dim)
    seed: random seed for sampling
    bound: whether to bound the control inputs
    input_bound: the bound for control inputs if bound is True
    """

    u = jnp.zeros((args.N, args.T, args.dim))
    YN  = jnp.zeros((args.N, args.T, args.dim))
    rng = jax.random.PRNGKey(seed=seed)
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)

    def reverse_once(carry):
        i, rng, Yi, current_u = carry
        rng, Y0s_rng = jax.random.split(rng)

        samples = jax.random.normal(Y0s_rng, (args.Nsample_cent, args.N, args.T, args.dim))
        Y0s = samples * jnp.sqrt(1.0 / alphas_bar[i] - 1.0) + Yi/jnp.sqrt(alphas_bar[i])
        new_Y0s = current_u + Y0s

        new_Y0s = jnp.where(bound, jnp.clip(new_Y0s, -input_bound, input_bound), new_Y0s)

        rewss = jax.vmap(cost_fn, in_axes=(0, None, None))(new_Y0s, x0, goal_loc)
        rew_mean = rewss.mean(axis=-1)
        rew_std = rewss.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        logp0 = (rewss - rew_mean) / rew_std / args.temp_sample
   
        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("i,intd->ntd", weights, Y0s)    

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        return (i - 1, rng, Yim1, current_u)


    def iterative_denoising(YN, u0, rng):
        diffuse_indices = jnp.arange(args.Ndiffuse-1, 0, -1)
        def diffuse_step(carry, i):
            rng, Yi, u = carry

            carry_once = (i, rng, Yi, u)
            _, rng, Yi, u = reverse_once(carry_once)

            return (rng, Yi, u), Yi

        def outer_step(carry, _):
            rng, u = carry
            Yi0 = YN
            (rng, Yi, u), _ = jax.lax.scan(diffuse_step, (rng, Yi0, u), diffuse_indices,)
            u = u + Yi
            return (rng, u), u
        (rng, u_final), ubars = jax.lax.scan(outer_step, (rng, u0), None, length=args.Niter,)
        return ubars


    rng_exp, rng = jax.random.split(rng)
    u = iterative_denoising(YN, u, rng_exp)
    return u












