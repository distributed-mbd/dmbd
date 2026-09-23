import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit,static_argnums=(0, 1, 2))
def distributed_mbd(args, cost_fn, rollout_fn, x0, goal_loc, seed = 42, bound=False, input_bound = 0.0):
    """Distributed Model-Based Diffusion (DMBD) planning function.
    args: arguments for the planner
    cost_fn: function to compute the cost of a trajectory
    rollout_fn: function to rollout the trajectory given initial state and control inputs
    x0: initial states of all robots, shape (N_robots, state_dim)
    goal_loc: goal locations of all robots, shape (N_robots, state_dim)
    seed: random seed for sampling
    bound: whether to bound the control inputs
    input_bound: the bound for control inputs if bound is True
    """
    rng = jax.random.PRNGKey(seed=seed)
    master_key, init_key = jax.random.split(rng)

    # Initialize the control inputs for all robots (In our paper, this is done by each robot and then sent to the server)
    YN = jax.random.normal(init_key, (args.N, args.T, args.dim))
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)

    def reverse_each_robot_once(i, Yki, trajectories, Y0s_rng, index):
        """Reverse the diffusion process for a single robot.
        i: current diffusion step index
        Yki: current control inputs for robot k, shape (T, dim)
        trajectories: the trajectories of all robots, shape (N_robots, T, state_dim)
        Y0s_rng: random key for sampling
        index: index of the robot k
        """

        # Sample control inputs for robot k at diffusion step i
        samples = jax.random.normal(Y0s_rng, (args.Nsample, args.T, args.dim))
        local_sampled_Y = samples * jnp.sqrt(1.0 / alphas_bar[i] - 1.0) + Yki / jnp.sqrt(alphas_bar[i])

        # Better without clipping, but can be added if desired
        local_sampled_Y = jnp.where(bound, jnp.clip(local_sampled_Y, -input_bound, input_bound), local_sampled_Y)

        # Compute the costs for the sampled control inputs using the cost function. 
        costs = jax.vmap(cost_fn, in_axes=(0, None, None, None, None))(
            local_sampled_Y, x0[index], trajectories, index, goal_loc[index])
        
        # Use costs as log-prob (team_cost already returns negative total cost)
        logp0 = costs / args.temp_sample
        logp0 = logp0 - jnp.max(logp0)

        # Compute the weights for the sampled control inputs using softmax of the log-probabilities.
        weights = jax.nn.softmax(logp0)

        Ybar = jnp.einsum("i,iTd->Td", weights, local_sampled_Y)

        # Compute the local conditional score and update the control inputs for robot k.
        score = (1.0 / (1.0 - alphas_bar[i])) * (
            -Yki + jnp.sqrt(alphas_bar[i]) * Ybar
        )
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yki + (1.0 - alphas_bar[i]) * score)

        return Yim1

    def reverse(YN, master_key):
        """Run the reverse diffusion process for all robots."""
        indices = jnp.arange(args.Ndiffuse - 1, 0, -1)

        def body(carry, idx):
            Yi, master_key = carry
            master_key, subkey = jax.random.split(master_key)
            robot_keys = jax.random.split(subkey, args.N)
            
            trajectories = rollout_fn(x0, Yi)
            # Each robot recevies the state trajectories of all robots, and then perform local conditional denoising to update its own control inputs. 
            # This is done in parallel for all robots using vmap. Then it sends its state trajectories to the server which aggregates and sends back to the robots.
            Yi = jax.vmap(reverse_each_robot_once, in_axes=(None, 0, None, 0, 0))(jnp.array(idx), Yi, trajectories, robot_keys, jnp.arange(args.N))
            return (Yi, master_key), Yi

        # initial carry: Yi = YN[0]
        init_carry = (YN, master_key)
        (final_carry, Ys) = jax.lax.scan(body, init_carry, indices)
        # Ys has shape (len(indices), ...) collect and return last
        return Ys

    Ybars = reverse(YN, master_key)  # shape (Ndiffuse-1, N_robots, Hsample, dim_u)
    return Ybars
