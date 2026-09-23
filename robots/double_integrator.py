import jax 
import jax.numpy as jnp


@jax.jit
def double_integrator_rollout(x0, u, dt=0.1, amax=1.0, vmax=5.0):
    """Roll out double-integrator dynamics for batched robots.

    Args:
        x0: array of shape (N, dim_state) containing initial states [px, py, vx, vy].
        u: array of shape (N, T, dim_u) containing accelerations for each robot over T steps.
        dt: time-step for integration.
        amax: maximum acceleration.
        vmax: maximum velocity.

    Returns:
        traj: array of shape (N, T, dim_state) containing the rolled-out states.
    """

    @jax.jit
    def rollout_single(x0_i, u_i):
        @jax.jit
        def step_fn(x, ui):
            px, py, vx, vy = x
            ax, ay = ui

            ax, ay = jnp.clip(ax, -amax, amax), jnp.clip(ay, -amax, amax)
            vx, vy = jnp.clip(vx, -vmax, vmax), jnp.clip(vy, -vmax, vmax)

            px_next = px + vx * dt + 1/2*ax*dt**2
            py_next = py + vy * dt + 1/2*ay*dt**2
            vx_next = vx + ax * dt
            vy_next = vy + ay * dt
            x_next = jnp.array([px_next, py_next, vx_next, vy_next])
            return x_next, x_next
        
        _, traj = jax.lax.scan(step_fn, x0_i, u_i)
        return traj

    return jax.vmap(rollout_single)(x0, u)