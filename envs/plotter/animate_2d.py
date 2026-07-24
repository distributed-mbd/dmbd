import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon as MplPolygon
import matplotlib.transforms as transforms
import jax
import jax.numpy as jnp

def get_color(N):
    if N <= 10:
        return plt.cm.tab10.colors
    else:
        return plt.cm.tab20.colors



def create_car_patches(x, y, theta, length, width, body_color, zorder=5, center_point=False, smaller_tires=False):
    """
    Create detailed car-shaped patches with body and tires.
    Reference point (x, y) is 20% from front, 80% from back.
    
    args:
        x, y: position (reference point at 0.8 * length from back)
        theta: orientation angle
        length: car body length
        width: car body width
        body_color: color tuple for the body
        zorder: z-order for rendering
        center_point: if True, the coordinate of the robot is at its center. If False, the coordinate is at the front axle

        
    Returns:
        list of matplotlib patches
    """
    # Car dimensions
    a = length * 0.37
    b = length * 0.18
    L = length
    W = width

    rear_overhang = (L - a - b) * 0.4
    front_overhang = (L - a - b) * 0.6

    # Reference point
    ref_offset = -b - rear_overhang + (0.5 if center_point else 0.8) * L

    # Small front chamfer amount
    chamfer = 0.08 * L

    x_rear = -b - rear_overhang - ref_offset
    x_front = a + front_overhang - ref_offset

    # ---------- RECTANGULAR BODY ----------
    body_vertices = np.array([
        [x_rear,             -W/2],
        [x_rear,              W/2],

        # keep full width over most of body
        [x_front - chamfer,   W/2],

        # slightly rounded front
        [x_front,             W*0.35],
        [x_front,            -W*0.35],

        [x_front - chamfer,  -W/2],
    ]).T
    
    # Rotation matrix
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    body_world = R @ body_vertices + np.array([[x], [y]])
    
    patches = [
        MplPolygon(
            body_world.T, closed=True,
            facecolor=body_color,
            edgecolor='black',
            linewidth=1.5, alpha=0.9, zorder=zorder
        )
    ]
    
    # Tire dimensions and positions
    tire_length = 0.5
    tire_width = 0.3

    if smaller_tires:
        tire_length = 0.375
        tire_width = 0.25

    tire_y_offset = W / 2 - tire_width / 2 - 0.1
    tire_positions = {
        'front_left': np.array([a, tire_y_offset]),
        'front_right': np.array([a, -tire_y_offset]),
        'rear_left': np.array([-b, tire_y_offset]),
        'rear_right': np.array([-b, -tire_y_offset])
    }
    
    tl = tire_length / 2
    tw = tire_width / 2
    tire_vertices = np.array([
        [-tl, -tw],
        [-tl, tw],
        [tl, tw],
        [tl, -tw]
    ]).T
    
    for pos in tire_positions.values():
        pos_local = pos - np.array([ref_offset, 0])
        pos_world = R @ pos_local.reshape(-1, 1) + np.array([[x], [y]])
        tire_world = R @ tire_vertices + pos_world
        patches.append(
            MplPolygon(
                tire_world.T, closed=True,
                facecolor=(0.3, 0.3, 0.3),
                edgecolor='black',
                linewidth=1, alpha=0.9, zorder=zorder + 1
            )
        )
    
    return patches


def animate_trajectories_rect(traj, args, lengths= None, obstacles=None, goal_positions=None,
                              save=False, interval=200, x_bounds =[-2, 18], y_bounds=[-3, 13.0],
                              center_point = False, smaller_tires = False):
    """
    Animate rectangular robots with orientation.

    args:
        traj: np.ndarray, shape (N_robots, T, state_dim) with state_dim >= 3 ([x, y, theta, ...])
        obstacles: np.ndarray of shape (num_obstacles, 3) -> [x, y, radius]
        goal_positions: np.ndarray of shape (N_robots, 2) -> [x, y]
        save: bool, whether to save to 'video.mp4'
        interval: int, ms between frames
        show_heading: bool, whether to show heading arrows (not used, kept for compatibility)
        center_point: if True, the coordinate of the robot is at its center. If False, the coordinate is at the front axle

    Requires in args:
        args.car_width   # robot width  (sideways)
    """

    N_robots, T, _ = traj.shape
    traj = np.array(jax.device_get(traj))
    positions = traj[:, :, :2]
    thetas = traj[:, :, 2]

    if goal_positions is None:
        goal_positions = np.zeros((N_robots, 2))

    # axis limits
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f9fa')

    # draw obstacles (circles or rectangles)
    if obstacles is not None:
        for obs in obstacles:
            if len(obs) == 3:
                # circle: [x, y, r]
                cx, cy, r = obs
                circ = plt.Circle((cx, cy), r, color='black', alpha=0.5, zorder=2)
                ax.add_patch(circ)

            elif len(obs) == 5:
                # rectangle: [x, y, theta, width, length]
                cx, cy, theta, w, l = obs

                rect = Rectangle(
                    (cx, cy),   
                    w,
                    l,
                    color='black',
                    alpha=0.5,
                    zorder=2
                )
                trans = transforms.Affine2D().rotate_around(cx, cy, theta) + ax.transData
                rect.set_transform(trans)
                ax.add_patch(rect)

    # robot parameters
    tab10 = plt.cm.tab10.colors
    colors = np.array([tab10[3]] + list(tab10[:3]) + list(tab10[4:]))  # index 3 is red
    colors = colors[np.arange(N_robots) % 10]
    width = float(args.car_width)

    # initialize robots - store patches for each robot
    car_patches_list = []

    # ---- Trail parameters ----
    trail_length = T
    trail_width = 16
    max_alpha = 0.25

    trail_segments = []

    for i in range(N_robots):
        x0, y0 = positions[i, 0]
        th = thetas[i, 0]

        patches = create_car_patches(
            x0, y0, th,
            lengths[i], width,
            colors[i],
            zorder=5,
            center_point=center_point,
            smaller_tires=smaller_tires
        )

        for patch in patches:
            ax.add_patch(patch)

        car_patches_list.append(patches)

        # Create fading trail segments
        segs = []
        for _ in range(trail_length):
            line, = ax.plot(
                [],
                [],
                color=colors[i],
                lw=trail_width,
                alpha=0,
                solid_capstyle="projecting",
                solid_joinstyle="miter",
                zorder=0,
            )
            segs.append(line)
        trail_segments.append(segs)

    # ---- Goals ----
    goal_markers = []
    for i in reversed(range(N_robots)):
        gx, gy = goal_positions[i]
        gm, = ax.plot([gx], [gy], marker='X', color=colors[i],
                      markersize=10, markeredgecolor='k', zorder=6)
        goal_markers.append(gm)

    artists = (
        [p for patches in car_patches_list for p in patches]
        + [seg for segs in trail_segments for seg in segs]
        + goal_markers
    )
    # ---- Init ----
    def init():
        for segs in trail_segments:
            for seg in segs:
                seg.set_data([], [])
                seg.set_alpha(0)

        return artists

    # ---- Update ----
    def update(frame):
        for i in range(N_robots):
            x, y = positions[i, frame]
            th = thetas[i, frame]

            # Remove old robot
            for patch in car_patches_list[i]:
                patch.remove()

            # Draw new robot
            patches = create_car_patches(
                x,
                y,
                th,
                lengths[i],
                width,
                colors[i],
                zorder=5,
                center_point=center_point,
                smaller_tires=smaller_tires

            )

            car_patches_list[i] = patches

            for patch in patches:
                ax.add_patch(patch)

            # -----------------------------
            # Thick fading trail
            # -----------------------------
            start = max(0, frame - trail_length)

            visible = frame - start

            for k, t in enumerate(range(start, frame)):
                trail_segments[i][k].set_data(
                    positions[i, t:t+2, 0],
                    positions[i, t:t+2, 1],
                )

                alpha = (k + 1) / trail_length
                trail_segments[i][k].set_alpha(max_alpha * alpha)

            # Hide unused segments
            for k in range(visible, trail_length):
                trail_segments[i][k].set_data([], [])
                trail_segments[i][k].set_alpha(0)

        ax.set_title(f"Time step {frame+1}/{T}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.tick_params(axis='both')

        return artists

    ani = FuncAnimation(
        fig, update, frames=T,
        init_func=init, interval=interval, blit=False
    )

    plt.tight_layout()

    if save:
        try:
            ani.save("sliding_traj.mp4", writer='ffmpeg', dpi=150)
        except Exception:
            print("Failed to save animation.")
            plt.show()
    else:
        plt.show()




def animate_trajectories(traj, args, obstacles=None, goal_positions=None, save=False, interval=200, show_heading=False, x_bounds=[-20.0, 20.0], y_bounds=[-20.0, 20.0]):
    """
    Animate circular robots with optional obstacles and goals.

    args:
        traj: np.ndarray, shape (N_robots, T, state_dim) with state_dim >= 3 ([x, y, theta, ...])
        obstacles: np.ndarray of shape (num_obstacles, 3) -> [x, y, radius]
        goal_positions: np.ndarray of shape (N_robots, 2) -> [x, y]
        save: bool, whether to save to 'video.mp4'
        interval: int, ms between frames
        show_heading: bool, whether to show heading arrows
    """
    N_robots, T, _ = traj.shape
    positions = np.array(jax.device_get(traj[:, :, :2]))
    thetas = np.array(jax.device_get(traj[:, :, 2]))

    if goal_positions is None:
        goal_positions = np.zeros((N_robots, 2))

    # ---------------- Figure ----------------
    min_x = float(np.min(positions[:, :, 0]))
    max_x = float(np.max(positions[:, :, 0]))
    min_y = float(np.min(positions[:, :, 1]))
    max_y = float(np.max(positions[:, :, 1]))

    pad = 2.0
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_facecolor("#f8f9fa")

    x_ticks = np.arange(x_bounds[0], x_bounds[1] + 1, 10)
    y_ticks = np.arange(y_bounds[0], y_bounds[1] + 1, 10)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # ---------------- Obstacles ----------------
    if obstacles is not None:
        for cx, cy, r in obstacles:
            circ = plt.Circle((cx, cy), r, color="black", alpha=0.5, zorder=2)
            ax.add_patch(circ)

    # ---------------- Robot parameters ----------------
    colors = get_color(N_robots)
    robot_radius = float(args.R_col)
    arrow_len = 1.2 * robot_radius

    # Trail parameters
    trail_length = T
    trail_width = 20
    max_alpha = 0.25

    circles = []
    heading_lines = []
    trail_segments = []

    # ---------------- Robots ----------------
    for i in range(N_robots):

        x0 = float(positions[i, 0, 0])
        y0 = float(positions[i, 0, 1])

        circle = plt.Circle(
            (x0, y0),
            robot_radius,
            facecolor=colors[i],
            alpha=0.9,
            edgecolor="black",
            zorder=5,
        )
        ax.add_patch(circle)
        circles.append(circle)

        # Thick fading trail
        segs = []
        for _ in range(trail_length):
            line, = ax.plot(
                [],
                [],
                color=colors[i],
                lw=trail_width,
                alpha=0,
                solid_capstyle="round",      # change to "projecting" if desired
                solid_joinstyle="round",
                zorder=0,
            )
            segs.append(line)

        trail_segments.append(segs)

        # Heading
        if show_heading:
            line, = ax.plot([], [], "-", color="k", linewidth=2)
            heading_lines.append(line)

    # ---------------- Goals ----------------
    goal_markers = []
    for i in range(N_robots):
        gx, gy = goal_positions[i, :2]
        gm, = ax.plot(
            [gx],
            [gy],
            marker="X",
            color=colors[i],
            markersize=10,
            markeredgecolor="k",
            zorder=6,
        )
        goal_markers.append(gm)

    artists = (
        circles
        + [seg for segs in trail_segments for seg in segs]
        + goal_markers
    )

    if show_heading:
        artists += heading_lines

    # ---------------- Init ----------------
    def init():

        for i in range(N_robots):

            circles[i].center = tuple(positions[i, 0])

            for seg in trail_segments[i]:
                seg.set_data([], [])
                seg.set_alpha(0)

            if show_heading:
                th = thetas[i, 0]
                dx = arrow_len * np.cos(th)
                dy = arrow_len * np.sin(th)

                x0, y0 = positions[i, 0]
                heading_lines[i].set_data(
                    [x0, x0 + dx],
                    [y0, y0 + dy],
                )

        return artists

    # ---------------- Update ----------------
    def update(frame):

        for i in range(N_robots):

            x, y = positions[i, frame]
            circles[i].center = (x, y)

            if show_heading:
                th = thetas[i, frame]
                dx = arrow_len * np.cos(th)
                dy = arrow_len * np.sin(th)

                heading_lines[i].set_data(
                    [x, x + dx],
                    [y, y + dy],
                )

            # Thick fading trail
            start = max(0, frame - trail_length)
            visible = frame - start

            for k, t in enumerate(range(start, frame)):

                trail_segments[i][k].set_data(
                    positions[i, t:t + 2, 0],
                    positions[i, t:t + 2, 1],
                )

                alpha = (k + 1) / trail_length
                trail_segments[i][k].set_alpha(max_alpha * alpha)

            # Hide unused trail segments
            for k in range(visible, trail_length):
                trail_segments[i][k].set_data([], [])
                trail_segments[i][k].set_alpha(0)

        ax.set_title(f"Time step {frame + 1}/{args.T}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.tick_params(axis="both")

        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=interval,
        blit=False,
    )

    if save:
        try:
            ani.save("circ_swap_traj.mp4", writer="ffmpeg", dpi=150)
        except Exception:
            print("Failed to save animation.")
            plt.show()
    else:
        plt.show()


def animate_denoising_trajectories_rect(control_history, initial_states, args,
                                        lengths=None, obstacles=None, goal_positions=None,
                                        rollout_fn=None,
                                        save=False, interval=200,
                                         x_bounds= [-17.5, 17.5], y_bounds=[-5.0, 35.0], center_point=False):
    """Animate how the predicted trajectories evolve across denoising steps.

    args:
        control_history: shape (Ndiffuse-1, N_agents, T, dim_u)
        initial_states: shape (N_agents, state_dim)
        args: configuration object with car dimensions
        lengths: optional per-agent vehicle lengths
        obstacles: optional obstacle list
        goal_positions: optional goal locations
        save: save to video.mp4 if True
        interval: milliseconds between frames
        show_heading: kept for API compatibility
        x_bounds, y_bounds: optional axis limits
    """
    traj_history = rollout_denoising_history(control_history, initial_states, rollout_fn=rollout_fn)
    n_steps, N_agents, T, _ = traj_history.shape

    initial_states = np.asarray(initial_states)
    initial_xy = np.asarray(initial_states[:, :2])
    positions = traj_history[:, :, :, :2]

    if lengths is None:
        lengths = np.array([getattr(args, "car_length", 5.0)] * N_agents)
    else:
        lengths = np.asarray(lengths)

    if goal_positions is None:
        goal_positions = np.zeros((N_agents, 2))

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f9fa')

    if obstacles is not None:
        for obs in obstacles:
            if len(obs) == 3:
                cx, cy, r = obs
                circ = plt.Circle((cx, cy), r, color='black', alpha=0.5, zorder=2)
                ax.add_patch(circ)
            elif len(obs) == 5:
                cx, cy, theta, w, l = obs
                rect = Rectangle((cx, cy), w, l, color='black', alpha=0.5, zorder=2)
                trans = transforms.Affine2D().rotate_around(cx, cy, theta) + ax.transData
                rect.set_transform(trans)
                ax.add_patch(rect)

    colors = get_color(N_agents)


    car_patches_list = []
    trails = []

    for i in range(N_agents):
        x0, y0, th0 = np.asarray(initial_states[i, :3])
        patches = create_car_patches(x0, y0, th0, lengths[i], float(args.car_width), colors[i], zorder=5, center_point=center_point)
        for patch in patches:
            ax.add_patch(patch)
        car_patches_list.append(patches)

        trail, = ax.plot([], [], '-', color=colors[i], alpha=0.6, linewidth=16)
        trails.append(trail)


    goal_markers = []
    for i in range(N_agents):
        gx, gy = goal_positions[i]
        gm, = ax.plot([gx], [gy], marker='X', color=colors[i], markersize=10,
                      markeredgecolor='k', zorder=6)
        goal_markers.append(gm)

    artists = [patch for patches in car_patches_list for patch in patches] + trails + goal_markers

    def init():
        for i in range(N_agents):
            trails[i].set_data([], [])
        return artists

    def update(frame):
        frame_traj = traj_history[frame]
        for i in range(N_agents):
            path = np.concatenate([initial_xy[i:i+1], frame_traj[i, :, :2]], axis=0)
            trails[i].set_data(path[:, 0], path[:, 1])

        ax.set_title(f"Denoising step {frame + 1}/{n_steps}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.tick_params(axis='both')
        return artists

    ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, interval=interval, blit=False)

    # plt.tight_layout()

    if save:
        try:
            ani.save("rect_denoising.mp4", writer='ffmpeg', dpi=150)
        except Exception:
            print("Failed to save animation.")
            plt.show()
    else:
        plt.show()





def animate_denoising_trajectories(control_history, initial_states, args,
                                        goal_positions=None,
                                        rollout_fn=None,
                                        save=False, interval=200,
                                        x_bounds=None, y_bounds=None):
    """Animate how circular-robot trajectories evolve across denoising steps.

    args:
        control_history: shape (Ndiffuse-1, N_agents, T, dim_u)
        initial_states: shape (N_agents, state_dim)
        args: configuration object with robot dimensions / radius
        obstacles: optional obstacle list
        goal_positions: optional goal locations
        rollout_fn: dynamics rollout to use for denoising history
        save: save to video.mp4 if True
        interval: milliseconds between frames
        show_heading: whether to draw heading arrows
        x_bounds, y_bounds: optional axis limits
    """
    traj_history = rollout_denoising_history(control_history, initial_states, rollout_fn=rollout_fn)
    n_steps, N_agents, T, _ = traj_history.shape

    initial_states = np.asarray(initial_states)
    initial_xy = np.asarray(initial_states[:, :2])
    positions = traj_history[:, :, :, :2]

    if goal_positions is None:
        goal_positions = np.zeros((N_agents, 2))

    if x_bounds is None:
        min_x = -17.5
        max_x = 17.5
        x_pad = max(2.0, 0.1 * (max_x - min_x + 1e-6))
        x_bounds = [min_x - x_pad, max_x + x_pad]

    if y_bounds is None:
        min_y = -17.5
        max_y = 17.5
        y_pad = max(2.0, 0.1 * (max_y - min_y + 1e-6))
        y_bounds = [min_y - y_pad, max_y + y_pad]

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    # ax.set_aspect('equal')
    ax.set_facecolor('#f8f9fa')
    x_ticks = np.arange(x_bounds[0], x_bounds[1]+1, 10)
    y_ticks = np.arange(y_bounds[0], y_bounds[1]+1, 10)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    colors = get_color(N_agents)
    agent_radius = float(getattr(args, 'R_col', 1.5))

    circles = []
    trails = []

    for i in range(N_agents):
        x0, y0 = initial_xy[i]
        c = plt.Circle((x0, y0), agent_radius, facecolor=colors[i], alpha=0.9, zorder=5, edgecolor="black")
        ax.add_patch(c)
        circles.append(c)

        trail, = ax.plot([], [], '-', color=colors[i], alpha=0.6, linewidth=20.0)
        trails.append(trail)

    goal_markers = []
    for i in range(N_agents):
        gx, gy = goal_positions[i]
        gm, = ax.plot([gx], [gy], marker='X', color=colors[i],
                      markersize=10, markeredgecolor='k', zorder=6)
        goal_markers.append(gm)

    artists = circles + trails + goal_markers

    def init():
        for i in range(N_agents):
            trails[i].set_data([], [])
            x0, y0 = initial_xy[i]
            circles[i].center = (x0, y0)
            
        return artists

    def update(frame):
        frame_traj = traj_history[frame]
        for i in range(N_agents):
            x, y = initial_xy[i]
            circles[i].center = (x, y)
            
            path = np.concatenate([initial_xy[i:i+1], frame_traj[i, :, :2]], axis=0)
            trails[i].set_data(path[:, 0], path[:, 1])

        ax.set_title(f"Denoising step {frame + 1}/{n_steps}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.tick_params(axis='both')
        return artists

    ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, interval=interval, blit=False)

    # plt.tight_layout()

    if save:
        try:
            ani.save("circ_swap_denoising.mp4", writer='ffmpeg', dpi=150)
        except Exception:
            print("Failed to save animation.")
            plt.show()
    else:
        plt.show()



def rollout_denoising_history(control_history, initial_states, rollout_fn):
    """Roll out every denoising step into state-space trajectories.

    args:
        control_history: array with shape (Ndiffuse-1, N_agents, T, dim_u)
        initial_states: array with shape (N_agents, state_dim)

    Returns:
        np.ndarray with shape (Ndiffuse-1, N_agents, T, state_dim)
    """
    control_history = jnp.asarray(control_history)
    traj_history = jax.vmap(
        lambda controls: rollout_fn(initial_states, controls)
    )(control_history)
    return np.asarray(jax.device_get(traj_history))