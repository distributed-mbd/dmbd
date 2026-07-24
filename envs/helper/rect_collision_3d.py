import jax.numpy as jnp
import jax

@jax.jit
def individual_obstacle_collision(sphere, robot_pos, car_w, car_l, car_h):
    """
    sphere: (4,) -> [cx, cy, cz, radius]
    robot_pos: (4,) -> [x, y, z, theta] 
    car_w : width of the car
    car_l : length of the car
    car_h : height of the car
    """
    cx, cy, cz, r = sphere
    ax, ay, az, theta = robot_pos

    dx = cx - ax
    dy = cy - ay
    dz = cz - az

    cos_a = jnp.cos(theta)
    sin_a = jnp.sin(theta)

    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a
    local_z = dz

    x_min, x_max = -0.8 * car_l, 0.2 * car_l
    y_min, y_max = -0.5 * car_w, 0.5 * car_w
    z_min, z_max = 0.0, car_h

    nearest_x = jnp.clip(local_x, x_min, x_max)
    nearest_y = jnp.clip(local_y, y_min, y_max)
    nearest_z = jnp.clip(local_z, z_min, z_max)

    dist_sq = (
        (local_x - nearest_x)**2 +
        (local_y - nearest_y)**2 +
        (local_z - nearest_z)**2
    )

    return jax.nn.relu(r**2 - dist_sq)


def obstacle_collision(pos_it, obs, car_width, car_length, car_height):
    """post_it: (4,) - robot k's position at time t
        obs: (Number_of_obstacles, 4) -  (cx, cy, cz, radius)    
        car_width: float - width of the car
        car_length: float - length of the car
        car_height: float - height of the car
    """
    penalties = jax.vmap(lambda obstacle: individual_obstacle_collision(
        obstacle, pos_it, car_width, car_length, car_height))(obs)
    return jnp.sum(penalties)


# ── Helpers for inter-robot collision ────────────────────────────────────────

@jax.jit
def _axle_to_center(ax, ay, cos_t, sin_t, car_l):
    """
    Shift front-axle position to geometric center of the rectangle.
    In local frame the box spans [-0.8l, +0.2l] so center is at -0.3l.
    """
    offset = -0.3 * car_l
    cx = ax + cos_t * offset
    cy = ay + sin_t * offset
    return cx, cy



@jax.jit
def _sat_separated(cx1, cy1, cos1, sin1, hl1, hw1,
                   cx2, cy2, cos2, sin2, hl2, hw2,
                   ax, ay):
    """True if the two rectangles are separated on axis (ax, ay)."""
    dist = jnp.abs(
        (cx1-cx2)*ax +
        (cy1-cy2)*ay
    )
    # Inline rect_support: half-width of projection of rectangle onto unit axis
    r1 = (jnp.abs(hl1 * (cos1 * ax + sin1 * ay)) +
          jnp.abs(hw1 * (-sin1 * ax + cos1 * ay)))
    r2 = (jnp.abs(hl2 * (cos2 * ax + sin2 * ay)) +
          jnp.abs(hw2 * (-sin2 * ax + cos2 * ay)))
    return dist > (r1 + r2)


@jax.jit
def _check_rect_collision_compact(cx1, cy1, cos1, sin1, hl1, hw1,
                                   cx2, cy2, cos2, sin2, hl2, hw2):
    """
    SAT collision check between two rotated rectangles.
    Works on geometric centers + symmetric half-extents.
    Returns True if collision.
    """
    # 4 separating axes: 2 per rectangle (parallel sides are redundant)
    axes_x = jnp.array([ cos1, -sin1,  cos2, -sin2])
    axes_y = jnp.array([ sin1,  cos1,  sin2,  cos2])

    def sep_on_axis(i):
        return _sat_separated(
            cx1, cy1, cos1, sin1, hl1, hw1,
            cx2, cy2, cos2, sin2, hl2, hw2,
            axes_x[i], axes_y[i]
        )

    separations = jax.vmap(sep_on_axis)(jnp.arange(4))
    return ~jnp.any(separations)


@jax.jit
def inter_robot_collision(pos_t, robot_k_pos_t, neighbors,
                                  car_width, car_length, robot_idx):
    """
    pos_t:          (N, 4)  — x, y, z, theta
    robot_k_pos_t:  (4,)
    neighbors:      (N,)
    car_width:      (N,)
    car_length:     (N,)
    robot_idx:      int
    """
    thetas  = pos_t[:, 3]
    cos_all = jnp.cos(thetas)         # (N,)
    sin_all = jnp.sin(thetas)         # (N,)
    half_ls = 0.5 * car_length        # (N,)
    half_ws = 0.5 * car_width         # (N,)

    # Shift all axle positions to box geometric centers
    cx_all, cy_all = jax.vmap(_axle_to_center)(
        pos_t[:, 0], pos_t[:, 1], cos_all, sin_all, car_length
    )  # (N,), (N,)

    # robot k scalars
    cos1 = jnp.cos(robot_k_pos_t[3])
    sin1 = jnp.sin(robot_k_pos_t[3])
    hl1  = 0.5 * car_length[robot_idx]
    hw1  = 0.5 * car_width[robot_idx]
    cx1, cy1 = _axle_to_center(
        robot_k_pos_t[0], robot_k_pos_t[1], cos1, sin1, car_length[robot_idx]
    )

    same_z = jnp.abs(pos_t[:, 2] - robot_k_pos_t[2]) < 1e-3  
    active = neighbors * same_z                                     

    def check_one(cx2, cy2, cos2, sin2, hl2, hw2):
        return _check_rect_collision_compact(
            cx1, cy1, cos1, sin1, hl1, hw1,
            cx2, cy2, cos2, sin2, hl2, hw2,
        )

    collisions = jax.vmap(check_one)(
        cx_all, cy_all, cos_all, sin_all, half_ls, half_ws
    )  

    return jnp.sum(collisions * active)



def goal_successes(traj: jnp.ndarray, goals: jnp.ndarray, goal_tolerance: float) -> bool:

    z_dist = jnp.abs(traj[:, -1, 2] - goals[:, 2])
    dist = jnp.linalg.norm(traj[:, -1, :2] - goals[:, :2], axis=-1)
    return jnp.all(dist <= goal_tolerance) & jnp.all(z_dist < 0.01) 


def count_inter_robot_collisions(traj: jnp.ndarray, lengths: jnp.ndarray, widths: float) -> float:
    """Count pairwise rectangle collisions over all timesteps."""
    n_agents = traj.shape[0]
    pos_time = traj[:, :, :4].transpose(1, 0, 2)  # (T, N, [x,y,z,theta])

    def per_time(pos_t):
        def per_agent(agent_idx):
            mask = jnp.arange(n_agents) != agent_idx
            return inter_robot_collision(
                pos_t,
                pos_t[agent_idx],
                mask,
                widths,
                lengths,
                agent_idx,
            )

        # each pair is counted twice, once from each agent perspective
        return jnp.sum(jax.vmap(per_agent)(jnp.arange(n_agents)))

    return float(jax.device_get(jnp.sum(jax.vmap(per_time)(pos_time))))


def count_obstacle_collisions(traj: jnp.ndarray, obs: jnp.ndarray, args, lengths, widths) -> float:
    """Count robot-obstacle collision events over all timesteps and agents."""
    n_agents = traj.shape[0]
    pos_time = traj[:, :, :4].transpose(1, 0, 2)  # (T, N, [x,y,z,theta])

    def per_time(pos_t):
        def per_agent(agent_idx):
            pos_i = pos_t[agent_idx]
            length_i = lengths[agent_idx]
            width_i = widths[agent_idx]
            penalties = jax.vmap(
                lambda obstacle: individual_obstacle_collision(
                    obstacle,
                    pos_i,
                    width_i,
                    length_i,
                    args.car_height,
                )
            )(obs)
            return jnp.sum(penalties > 0)

        return jnp.sum(jax.vmap(per_agent)(jnp.arange(n_agents)))

    return float(jax.device_get(jnp.sum(jax.vmap(per_time)(pos_time))))