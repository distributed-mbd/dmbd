import jax
import jax.numpy as jnp

@jax.jit
def kinematic_bicycle_rollout(x0, u, dt=0.25, L=3.0, vmax=5.0, amax=1.0, steer_max=0.25):
    """Roll out a kinematic bicycle model for batched robots.

    State: [x, y, theta, v]
    Control per step: [a, delta]

    Args:
        x0: array of shape (N, 4) containing initial states [x,y,theta,v]
        u: array of shape (N, T, 2) containing controls [a, delta] per timestep
        dt: timestep duration
        L: wheelbase length (meters)

    Returns:
        traj: array of shape (N, T, 4) containing states at each timestep

    Notes:
        - Integration uses simple forward Euler (sufficient for small dt).
        - The function is jitted and vectorized over the robot batch.
    """

    def rollout_single(x0_i, u_i):
        def step_fn(x, ui):
            x_pos, y_pos, theta, v = x
            a, delta = ui
            a = jnp.clip(a, -amax, amax)
            delta = jnp.clip(delta, -steer_max, steer_max)
            v = jnp.clip(v, -vmax, vmax)
            # Kinematic bicycle continuous-time dynamics
            dx = v * jnp.cos(theta)
            dy = v * jnp.sin(theta)
            dtheta = v / L * jnp.tan(delta)
            dv = a

            x_pos_next = x_pos + dx * dt
            y_pos_next = y_pos + dy * dt
            theta_next = theta + dtheta * dt
            
            v_next = v + dv * dt

            next_state = jnp.array([x_pos_next, y_pos_next, theta_next, v_next])
            return next_state, next_state

        _, traj = jax.lax.scan(step_fn, x0_i, u_i)
        return traj

    return jax.vmap(rollout_single)(x0, u)


@jax.jit
def one_D_kinematic_bicycle_rollout(x0, u, dt=0.25, vmax=5.0, amax=1.0):
    """Roll out 1D kinematic bicycle model for batched robots.


    State: [x, y, theta, v]
    Control per step: [a]

    The robots can only move forward or backward based on the direction 
    determined by their initial theta angle.

    Args:
        x0: array of shape (N, 4) containing initial states [x,y,theta,v]
        u: array of shape (N, T, 1) containing controls [a] per timestep
        dt: timestep duration
        L: wheelbase length (meters)

    Returns:
        traj: array of shape (N, T, 4) containing states at each timestep

    Notes:
        - Integration uses simple forward Euler (sufficient for small dt).
        - The function is jitted and vectorized over the robot batch.
    """

    def rollout_single(x0_i, u_i):
        def step_fn(x, a):
            x_pos, y_pos, theta, v = x
            a = jnp.clip(a[0], -amax, amax)
            v = jnp.clip(v, -vmax, vmax)

            # Kinematic bicycle continuous-time dynamics
            dx = v * jnp.cos(theta)
            dy = v * jnp.sin(theta)

            x_pos_next = x_pos + dx * dt
            y_pos_next = y_pos + dy * dt
            
            v_next = v + a * dt

            next_state = jnp.array([x_pos_next, y_pos_next, theta, v_next])
            return next_state, next_state

        _, traj = jax.lax.scan(step_fn, x0_i, u_i)
        return traj

    return jax.vmap(rollout_single)(x0, u)


@jax.jit
def buffered_kinematic_bicycle_rollout(initial_states, control_sequences, dt=0.25, L = 4.2, vmax=5.0, amax=1.0, steer_max=0.25):
    """ Rollout function that can handle different car dynamics.
        robot 0 uses elevator kinematics, other robots use standard bicycle kinematics.
        initial_states: (N, state_dim) - Initial states for all robots
        control_sequences: (N, T, control_dim) - Control sequences for all robots
    """
    kin_traj_4d = kinematic_bicycle_rollout(initial_states[:, [0,1,3,4]], control_sequences, dt=dt, L=L, vmax=vmax, amax=amax, steer_max=steer_max)  # (N, T, 4)
    z_init = initial_states[:, 2]  # (N,)
    z_traj = jnp.repeat(z_init[:, None], control_sequences.shape[1], axis=1)  # (N, T)
    trajectories = jnp.concatenate([kin_traj_4d[:, :, :2], z_traj[:, :, None], kin_traj_4d[:, :, 2:]], axis=-1)
    return trajectories


@jax.jit
def elevator_kinematic_bicycle_rollout(
    x0,
    u,
    dt=0.25,
    L=3.0,
    z_max=5.0,
    z_rate=1.0,
    car_length=5.0,
    car_width=2.0,
    x_min=jnp.array([-10.0]),
    x_max=jnp.array([10.0]),
    y_min=jnp.array([-10.0]),
    y_max=jnp.array([10.0]),
    vmax=5.0,
    amax=1.0,
    steer_max=0.25
):
    """Roll out a kinematic bicycle model for batched robots. 
    If robots get within the region defined by (x_min, x_max, y_min, y_max), their z position will go up to 5.0 at max.

    State: [x, y, z, theta, v]
    Control per step: [a, delta]

    Args:
        x0: array of shape (N, 5) containing initial states [x,y,z,theta,v]
        u: array of shape (N, T, 2) containing controls [a, delta] per timestep
        dt: timestep duration
        L: wheelbase length (meters)
        x_min, x_max, y_min, y_max: boundaries of the region where z position by 0.will increase until it hits z_max

    Returns:
        traj: array of shape (N, T, 4) containing states at each timestep

    Notes:
        - Integration uses simple forward Euler (sufficient for small dt).
        - The function is jitted and vectorized over the robot batch.
    """

    def rollout_single(x0_i, u_i):

        def body_in_region(x, y, theta, x_min_r, x_max_r, y_min_r, y_max_r):
            """Return True if the full vehicle footprint is inside one elevator zone."""
            half_w = 0.5 * car_width
            front = 0.2 * car_length  # Only the front 20% of the car needs to be in the elevator zone to count as "in the elevator"
            rear = -0.8 * car_length  # The rear can be a bit outside

            corners_local = jnp.array([
                [rear, -half_w],
                [rear,  half_w],
                [front, -half_w],
                [front,  half_w],
            ])

            c = jnp.cos(theta)
            s = jnp.sin(theta)
            rot = jnp.array([[c, -s], [s, c]])
            corners_world = corners_local @ rot.T + jnp.array([x, y])

            inside_x = (corners_world[:, 0] >= x_min_r) & (corners_world[:, 0] <= x_max_r)
            inside_y = (corners_world[:, 1] >= y_min_r) & (corners_world[:, 1] <= y_max_r)
            return jnp.all(inside_x & inside_y)

        def step_fn(x, ui):
            x_pos, y_pos, z, theta, v = x
            a, delta = ui

            # Clip controls
            a = jnp.clip(a, -amax, amax)
            delta = jnp.clip(delta, -steer_max, steer_max)
            v = jnp.clip(v, -vmax, vmax)

            # Bicycle dynamics
            dx = v * jnp.cos(theta)
            dy = v * jnp.sin(theta)
            dtheta = v / L * jnp.tan(delta)
            dv = a

            x_next = x_pos + dx * dt
            y_next = y_pos + dy * dt
            theta_next = theta + dtheta * dt
            v_next = v + dv * dt

            # --- Elevator logic ---
            in_any_region = jnp.any(
                jax.vmap(body_in_region, in_axes=(None, None, None, 0, 0, 0, 0))(
                    x_next,
                    y_next,
                    theta_next,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                )
            )
            # Increase z if inside region
            z_next = jnp.where(
                in_any_region,
                jnp.minimum(z + z_rate * dt, z_max),
                z  # no change outside
            )

            next_state = jnp.array([x_next, y_next, z_next, theta_next, v_next])
            return next_state, next_state

        _, traj = jax.lax.scan(step_fn, x0_i, u_i)
        return traj

    return jax.vmap(rollout_single)(x0, u)




# @partial(jax.jit, stati_argnums=(0, ))
def rollout_heterogenous_team(args, initial_states: jnp.ndarray, controls: jnp.ndarray, dt=0.25, vmax=5.0, amax=1.0, steer_max=0.25) -> jnp.ndarray:
    x_mins = jnp.array([args.x_min1, args.x_min2, args.x_min3, args.x_min4])
    x_maxs = jnp.array([args.x_max1, args.x_max2, args.x_max3, args.x_max4])
    y_mins = jnp.array([args.y_min1, args.y_min2, args.y_min3, args.y_min4])
    y_maxs = jnp.array([args.y_max1, args.y_max2, args.y_max3, args.y_max4])
    traj_elevator = elevator_kinematic_bicycle_rollout(
        initial_states[: args.N_elevator],
        controls[: args.N_elevator],
        dt=dt,
        x_min=x_mins,
        x_max=x_maxs,
        y_min=y_mins,
        y_max=y_maxs,
        car_length=args.car_length,
        car_width=args.car_width,
        vmax=vmax,
        amax=amax,
        steer_max=steer_max
    )
    traj_no_elevator = buffered_kinematic_bicycle_rollout(
        initial_states[args.N_elevator :],
        controls[args.N_elevator :],
        dt=dt,
        vmax=vmax,
        amax=amax,
        steer_max=steer_max
    )
    return jnp.concatenate([traj_elevator, traj_no_elevator], axis=0)