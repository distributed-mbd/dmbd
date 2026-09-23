import jax
import jax.numpy as jnp

@jax.jit
def individual_obstacle_collision(circle, robot_pos, car_w, car_l, ratio =0.8):
    """
    circle: (3,) -> [cx, cy, radius]
    robot_pos: (3,) -> [x, y, theta]
    car_w : width of the car
    car_l : length of the car
    """
    cx, cy, r = circle
    ax, ay, theta = robot_pos

    # 1. Translate relative to front axle
    dx = cx - ax
    dy = cy - ay

    # 2. Rotate circle into car's local frame
    # (Using -theta to bring the world into alignment with the car)
    cos_a = jnp.cos(theta)
    sin_a = jnp.sin(theta)
    
    # Standard 2D rotation matrix: [cos -sin; sin cos] transposed for inverse
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a

    ratio2 = 1 - ratio
    # 3. Define local bounds relative to front axle (0,0)
    # Front is at 0.2L, Back at -0.8L
    x_min, x_max = -ratio * car_l, ratio2 * car_l
    y_min, y_max = -0.5 * car_w, 0.5 * car_w

    # 4. Clamp to the local rectangle
    nearest_x = jnp.clip(local_x, x_min, x_max)
    nearest_y = jnp.clip(local_y, y_min, y_max)

    # 5. Check distance
    dist_sq = (local_x - nearest_x)**2 + (local_y - nearest_y)**2
    
    # Return penalty (positive if colliding, 0 if not)
    return jax.nn.relu(r**2 - dist_sq)



def obstacle_collision(pos_it, obs, car_width, car_length, ratio=0.8):
    '''post_it: (3,) - robot k's position at time t
       obs: (Number_of_obstacles, 3) - M obstacles, each with (cx, cy, radius)
       car_width: float - width of the car
       car_length: float - length of the car
       ratio: float - ratio for defining the center of the rectangle (0.8 means 80% of the length is in front of the axle)'''
    
    # rec corners of robot i at time t
    penalties = jax.vmap(lambda obstacle: individual_obstacle_collision(obstacle, pos_it, car_width, car_length, ratio))(obs)

    return jnp.sum(penalties)

@jax.jit
def _axle_to_center(ax, ay, cos_t, sin_t, car_l):
    """Convert a front-axle pose to the rectangle's geometric center."""
    offset = -0.3 * car_l
    cx = ax + cos_t * offset
    cy = ay + sin_t * offset
    return cx, cy



@jax.jit
def _sat_separated(cx1, cy1, cos1, sin1, hl1, hw1,
                   cx2, cy2, cos2, sin2, hl2, hw2,
                   ax, ay):
    """Return True when two rectangles are separated on one axis."""
    dist = jnp.abs(
        (cx1-cx2)*ax +
        (cy1-cy2)*ay
    )
    # Projection radius of each rectangle on the test axis.
    r1 = (jnp.abs(hl1 * (cos1 * ax + sin1 * ay)) +
          jnp.abs(hw1 * (-sin1 * ax + cos1 * ay)))
    r2 = (jnp.abs(hl2 * (cos2 * ax + sin2 * ay)) +
          jnp.abs(hw2 * (-sin2 * ax + cos2 * ay)))
    return dist > (r1 + r2)


@jax.jit
def _check_rect_collision_compact(cx1, cy1, cos1, sin1, hl1, hw1,
                                   cx2, cy2, cos2, sin2, hl2, hw2):
    """Return True if two rotated rectangles overlap under SAT."""
    # Two unique axes from each rectangle are sufficient for SAT.
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
def inter_robot_collision(pos_t, robot_k_pos_t, mask,
                                  car_width, car_length, robot_idx):
    """pos_t: (N, 3) - all robots' positions at time t
       robot_k_pos_t: (3,) - robot k's position at time t
       mask: (N,) - boolean mask for which robots to consider for collision
       car_width: float - width of the car
       car_length: float - length of the car
       robot_idx: int - index of the robot k"""
    
    thetas  = pos_t[:, 2]
    cos_all = jnp.cos(thetas)
    sin_all = jnp.sin(thetas)
    half_ls = 0.5 * car_length
    half_ws = 0.5 * car_width

    # Convert each pose to a box center.
    cx_all, cy_all = jax.vmap(_axle_to_center)(
        pos_t[:, 0], pos_t[:, 1], cos_all, sin_all, car_length
    )  # (N,), (N,)

    # Active robot geometry.
    cos1 = jnp.cos(robot_k_pos_t[2])
    sin1 = jnp.sin(robot_k_pos_t[2])
    hl1  = 0.5 * car_length[robot_idx]
    hw1  = 0.5 * car_width[robot_idx]
    cx1, cy1 = _axle_to_center(
        robot_k_pos_t[0], robot_k_pos_t[1], cos1, sin1, car_length[robot_idx]
    )

    def check_one(cx2, cy2, cos2, sin2, hl2, hw2):
        return _check_rect_collision_compact(
            cx1, cy1, cos1, sin1, hl1, hw1,
            cx2, cy2, cos2, sin2, hl2, hw2,
        )

    collisions = jax.vmap(check_one)(cx_all, cy_all, cos_all, sin_all, half_ls, half_ws)

    return jnp.sum(collisions * mask)

# Check the resulting trajectory for collisions between agents.
def count_inter_robot_collisions(traj: jnp.ndarray, lengths: jnp.ndarray, widths: jnp.ndarray) -> float:
    """Count pairwise rectangle collisions over all timesteps."""
    n_agents = traj.shape[0]
    pos_time = traj[:, :, :3].transpose(1, 0, 2)  # (T, N, [x,y,theta])

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


## Faster version of inter_robot_collision that only considers rectangles with 0, 90, 180, or 270 degree rotations ##
def faster_inter_robot_collision(pos_t, robot_k_pos_t, mask, car_width, lengths, robot_idx, pentalty_weight):
    """pos_t: (N, 3) - all robots' positions at time t
       robot_k_pos_t: (3,) - robot k's position at time t
       mask: (N,) - boolean mask for which robots to consider for collision
       car_width: float - width of the car
       lengths: (N,) - lengths of all robots
       robot_idx: int - index of the robot k
       pentalty_weight: float - weight for the collision penalty"""

    pos_t = pos_t.at[robot_idx].set(robot_k_pos_t)
    pos = pos_t[:, :2] 
    thetas = pos_t[:, 2]

    # Rotation masks
    rot_90_mask = (thetas % (2*jnp.pi) == jnp.pi/2) | (thetas % (2*jnp.pi) == 3*jnp.pi/2)

    # Initialize bounds
    x_min = jnp.where(rot_90_mask, pos[:,0] - 0.5*car_width, pos[:,0] - 0.5*lengths)
    x_max = jnp.where(rot_90_mask, pos[:,0] + 0.5*car_width, pos[:,0] + 0.5*lengths)
    y_min = jnp.where(rot_90_mask, pos[:,1] - 0.5*lengths, pos[:,1] - 0.5*car_width)
    y_max = jnp.where(rot_90_mask, pos[:,1] + 0.5*lengths, pos[:,1] + 0.5*car_width)

    # robot k bounds
    xi_min, xi_max = x_min[robot_idx], x_max[robot_idx]
    yi_min, yi_max = y_min[robot_idx], y_max[robot_idx]

    # Overlaps
    overlap_x = jnp.maximum(0, jnp.minimum(xi_max, x_max) - jnp.maximum(xi_min, x_min))
    overlap_y = jnp.maximum(0, jnp.minimum(yi_max, y_max) - jnp.maximum(yi_min, y_min))
    collision = (overlap_x > 0) & (overlap_y > 0)

    # Ignore self and apply mask
    collision = collision * mask * pentalty_weight 
    return jnp.sum(collision)

## Faster version of count_inter_robot_collisions that only considers rectangles with 0, 90, 180, or 270 degree rotations ##
def faster_count_inter_robot_collisions(traj: jnp.ndarray, lengths: jnp.ndarray, car_width: float) -> float:
    """Count pairwise rectangle collisions over all timesteps."""
    n_robots = traj.shape[0]
    pos_time = traj[:, :, :3].transpose(1, 0, 2)  # (T, N, [x,y,z,theta])

    def per_time(pos_t):
        def per_robot(robot_idx):
            mask = jnp.arange(n_robots) != robot_idx
            return faster_inter_robot_collision(
                pos_t,
                pos_t[robot_idx],
                mask,
                car_width,
                lengths,
                robot_idx,
                jnp.array([1.0, 1.0, 1.0, 1.0])  # penalty weight for each robot
            )

        # each pair is counted twice, once from each robot perspective
        return jnp.sum(jax.vmap(per_robot)(jnp.arange(n_robots)))

    return float(jax.device_get(jnp.sum(jax.vmap(per_time)(pos_time))))




# Check the resulting trajectory for collisions between agents and obstacles.
def count_obstacle_collisions(traj: jnp.ndarray, obs: jnp.ndarray, lengths, widths, ratio=0.8) -> float:
    """Count robot-obstacle collision events over all timesteps and agents."""
    n_agents = traj.shape[0]
    pos_time = traj[:, :, :3].transpose(1, 0, 2)  # (T, N, [x,y,theta])

    def per_time(pos_t):
        def per_agent(agent_idx):
            pos_i = pos_t[agent_idx]
            length_i = lengths[agent_idx]
            penalties = jax.vmap(
                lambda obstacle: individual_obstacle_collision(
                    obstacle,
                    pos_i,
                    widths[agent_idx],
                    length_i,
                    ratio
                )
            )(obs)
            return jnp.sum(penalties > 0)

        return jnp.sum(jax.vmap(per_agent)(jnp.arange(n_agents)))

    return float(jax.device_get(jnp.sum(jax.vmap(per_time)(pos_time))))

