import jax
import jax.numpy as jnp

@jax.jit
def count_inter_robot_collisions(traj: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """Return the total amount by which agents overlap over time."""

    pos = traj[:, :, :2]  # (N, T, 2)
    diffs = pos[:, None, :, :] - pos[None, :, :, :]  # (N, N, T, 2)
    dist = jnp.linalg.norm(diffs, axis=-1)  # (N, N, T)
    radii_sum = radii[:, None] + radii[None, :]
    collision = dist < radii_sum[:, :, None]

    mask = 1.0 - jnp.eye(radii.shape[0])[:, :, None]

    return jnp.sum(collision * mask)

@jax.jit
def vector_inter_robot_collision(pos_t, radius):
    """faster but more memory intensive version of inter_robot_collision that only considers circles
    pos_t : (N, 2) - state of all robots
    radius : (N,) - radius of all robots
    """

    # Pairwise relative positions and distances.
    diff = pos_t[:, None, :] - pos_t[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)

    N = pos_t.shape[0]
    mask = 1.0 - jnp.eye(N)

    # Penalize overlap when the combined radius exceeds the distance.
    radius_sq = radius[:, None] + radius[None, :]
    radius_sq = radius_sq**2

    return jnp.sum(jax.nn.relu(radius_sq - dist_sq) * mask) 



@jax.jit
def inter_robot_collision(pos, robot_k_pos_t, radius, mask):
    """
    Vectorized collision penalty for circles (no Python loops).
    pos_t: (N, 2) - state of all robots
    robot_k_pos_t: (2,) - state of the active robot
    mask: (N,) binary mask for which other robots to include
    Returns: scalar penalty
    """
    # Compute distance to all other robots
    dist_sq = jnp.sum((pos - robot_k_pos_t)**2, axis=1)  # (N,)
    return jnp.sum(jax.nn.relu(radius**2 - dist_sq) * mask)