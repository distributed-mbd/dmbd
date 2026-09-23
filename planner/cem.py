import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit,static_argnums=(0, 1))
def cem(args,cost_fn, x0, goal_loc, seed = 42, bound=False, input_bound = 0.0):
    """Cross Entropy Method (CEM) planning function. Implementation is based on the paper "The Cross-Entropy Method for Combinatorial and Continous Optimization" by Rubinstein (1999)
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

    sigma = (
        jnp.ones((args.N, args.T, args.dim))
    )
    Nelite = int(args.Nsample_cent * args.elite_frac)
    
    def reverse_once(carry):
        rng, mu, sigma = carry
        rng, Y0s_rng = jax.random.split(rng)

        samples = jax.random.normal(Y0s_rng, (args.Nsample_cent, args.N, args.T, args.dim))
        Y0s = mu + sigma * samples

        Y0s = jnp.where(bound, jnp.clip(Y0s, -input_bound, input_bound), Y0s)
        # returns the (-costs) of the sampled control sequences, shape (Nsample_cent, N)
        rewss = jax.vmap(cost_fn, in_axes=(0, None, None))(Y0s, x0, goal_loc) 

        # Select the elite samples and update the mean and std
        elite_idx = jnp.argsort(rewss)[-Nelite:]
        new_Y0s = Y0s[elite_idx]
        mu = jnp.mean(new_Y0s, axis=0)
        sigma = jnp.maximum(jnp.std(new_Y0s, axis=0), 1e-5)

        return (rng, mu, sigma)


    # run reverse
    def reverse(YN, rng, sigma):
        indices = jnp.arange(args.Ndiffuse - 1, 0, -1)
        def scan_body(carry, i):
            rng, Yi, sigma = carry
            carry_once = (rng, Yi, sigma)
            rng, Yi, sigma = reverse_once(carry_once)

            return (rng, Yi, sigma), Yi

        (_, _, _), Ybars = jax.lax.scan(scan_body,(rng, YN, sigma), indices)
        return Ybars
    
    rng_exp, rng = jax.random.split(rng)
    Yi = reverse(YN, rng_exp, sigma)
    return Yi





