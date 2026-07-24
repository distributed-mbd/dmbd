import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import jax
from envs.plotter.animate_2d import rollout_denoising_history

def create_car_3d(cx, cy, cz, length, width, height, theta, center_point=False):
    """
    Create a detailed 3D car body with a chamfered front and raised cabin.

    args:
        cx, cy, cz: Reference point of the robot.
        length: Total vehicle length.
        width: Total vehicle width.
        height: Total vehicle height.
        theta: Yaw angle in radians.
        center_point: If True, (cx, cy, cz) is at the center of the car.
                      If False, (cx, cy, cz) is at the front axle.

    Returns:
        List of polygon faces compatible with Poly3DCollection.set_verts().
    """

    L = length
    W = width
    H = height

    # Body proportions
    a = L * 0.37
    b = L * 0.18

    rear_overhang = (L - a - b) * 0.4
    front_overhang = (L - a - b) * 0.6

    # Reference point offset
    ref_offset = -b - rear_overhang + (0.5 if center_point else 0.8) * L

    x_rear = -b - rear_overhang - ref_offset
    x_front = a + front_overhang - ref_offset

    # Front chamfer
    chamfer = 0.08 * L

    # Rotation matrix
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # ============================================================
    # LOWER BODY
    # ============================================================

    body_outline = np.array([
        [x_rear, -W / 2],
        [x_rear, W / 2],
        [x_front - chamfer, W / 2],
        [x_front, W * 0.35],
        [x_front, -W * 0.35],
        [x_front - chamfer, -W / 2],
    ])

    body_height = H * 0.55

    bottom_z = -H / 2
    top_z = bottom_z + body_height

    bottom = np.column_stack([body_outline, np.full(len(body_outline), bottom_z)])
    top = np.column_stack([body_outline, np.full(len(body_outline), top_z)])

    bottom_world = (R @ bottom.T).T + np.array([cx, cy, cz])
    top_world = (R @ top.T).T + np.array([cx, cy, cz])

    faces = []

    n = len(body_outline)

    # Bottom
    faces.append([bottom_world[i] for i in range(n)])

    # Top
    faces.append([top_world[i] for i in range(n)])

    # Side walls
    for i in range(n):
        j = (i + 1) % n
        faces.append([bottom_world[i], bottom_world[j], top_world[j], top_world[i]])

    # ============================================================
    # UPPER CABIN
    # ============================================================

    cabin_length = L * 0.75
    cabin_width = W * 0.82
    cabin_height = H * 0.45

    cabin_x_rear = -L * 0.35 - ref_offset
    cabin_x_front = cabin_x_rear + cabin_length

    cabin_chamfer = 0.08 * L

    cabin_outline = np.array([
        [cabin_x_rear, -cabin_width / 2],
        [cabin_x_rear, cabin_width / 2],
        [cabin_x_front - cabin_chamfer, cabin_width / 2],
        [cabin_x_front, cabin_width * 0.35],
        [cabin_x_front, -cabin_width * 0.35],
        [cabin_x_front - cabin_chamfer, -cabin_width / 2],
    ])

    cabin_bottom_z = top_z
    cabin_top_z = top_z + cabin_height

    cabin_bottom = np.column_stack([cabin_outline, np.full(len(cabin_outline), cabin_bottom_z)])
    cabin_top = np.column_stack([cabin_outline, np.full(len(cabin_outline), cabin_top_z)])

    cabin_bottom_world = (R @ cabin_bottom.T).T + np.array([cx, cy, cz])
    cabin_top_world = (R @ cabin_top.T).T + np.array([cx, cy, cz])

    n = len(cabin_outline)

    # Cabin top
    faces.append([cabin_top_world[i] for i in range(n)])

    # Cabin sides
    for i in range(n):
        j = (i + 1) % n
        faces.append([cabin_bottom_world[i], cabin_bottom_world[j], cabin_top_world[j], cabin_top_world[i]])

    return faces

def make_box_faces(cx, cy, cz, length, width, height, theta, center_point=False):
    """
    Return 6 faces (each a list of 4 vertices) for a box centred at
    (cx, cy, cz) with the given half-extents and yaw=theta.
    Front axle is at the centre; body extends [-0.8*l … +0.2*l] in local x.
    """
    lf  =  length * 0.2   # forward of axle
    lb  = -length * 0.8   # rear   of axle
    if center_point:
        lf = length * 0.5
        lb = -length * 0.5
    hw  =  width  * 0.5
    ht  =  height * 0.5   # half-height (robot body)

    # Local 8 corners [x, y, z]
    corners_local = np.array([
        [lb, -hw, -ht], [lf, -hw, -ht],
        [lf,  hw, -ht], [lb,  hw, -ht],
        [lb, -hw,  ht], [lf, -hw,  ht],
        [lf,  hw,  ht], [lb,  hw,  ht],
    ])

    # Rotate around z-axis by theta, then translate
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = (R @ corners_local.T).T + np.array([cx, cy, cz])

    # 6 faces (index pairs)
    idx_faces = [
        [0, 1, 2, 3],   # bottom
        [4, 5, 6, 7],   # top
        [0, 1, 5, 4],   # front-right
        [2, 3, 7, 6],   # back-left
        [0, 3, 7, 4],   # left
        [1, 2, 6, 5],   # right
    ]
    return [[corners[j] for j in face] for face in idx_faces]


def animate_trajectories_3d(
    traj,
    args,
    lengths=None,
    widths = None,
    obstacles=None,
    goal_positions=None,
    N_elevator=0,
    elevator_region=None,      # (x_min, x_max, y_min, y_max) footprint of elevator shaft
    save=False,
    interval=200,
    x_bounds=[-30.0, 30.0],
    y_bounds=[-30.0, 30.0],
    floor_z0=0.0,
    floor_z1=5.0,
    unassigned_goals=False,
    center_point=False
):
    """
    Animate rectangular robots in 3D with two floors and a moving elevator.
 
    args:
        traj            : np.ndarray (N_robots, T, state_dim), state_dim >= 3 [x, y, theta, ...]
        args            : namespace with args.car_width
        lengths         : list/array of robot lengths, one per robot
        obstacles       : array (N_obs, 3) [x, y, r] circles  OR  (N_obs, 5) [x, y, theta, w, l] rects
        goal_positions  : array (N_robots, 2)
        elevator_region : tuple (x_min, x_max, y_min, y_max) footprint of the elevator on the floor
        save            : save to 'video3d.mp4' if True
        interval        : ms between frames
        show_heading    : draw heading arrows
        x_bounds        : [xmin, xmax]
        y_bounds   Heterogeneous robots navigate between floors to cover designated areas while avoiding collisions. Only smaller robots (red) can take the elevators (shaded in yellow) to access the second floor, while the other, heavier robots (blue) must navigate around the elevators to reach their goal locations. The robots do not know the goal locations of each other     : [ymin, ymax]
        floor_z0        : z of the lower floor  (default 0.0)
        floor_z1        : z of the upper floor  (default 5.0)
        elevator_speed  : z-units moved per frame while lifting (default 1.0)
        center_point     : if True, the (x,y) point of the robot is at its center; if False, it's at the front axle (default False)
    """
 
    # ------------------------------------------------------------------ #
    #  Data prep
    # ------------------------------------------------------------------ #

    N_robots, T, _ = traj.shape
    traj = np.array(jax.device_get(traj))
    positions = traj[:, :, :3]          # (N, T, 3)
    thetas    = traj[:, :, 3]           # (N, T)
 
    if goal_positions is None:
        goal_positions = np.zeros((N_robots, 3))
 
    if lengths is None:
        lengths = [float(getattr(args, 'car_length', 1.0))] * N_robots
 
    if widths is None:
        widths = [float(getattr(args, 'car_width', 1.0))] * N_robots
 

    # ------------------------------------------------------------------ #
    #  Figure / axes
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
 
    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    ax.set_zlim(floor_z0 - 0.5, floor_z1 + 1.0)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
 
    # ------------------------------------------------------------------ #
    #  Draw static floors
    # ------------------------------------------------------------------ #
    def draw_floor(z, alpha=0.15, color='steelblue'):
        xs = [x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0]]
        ys = [y_bounds[0], y_bounds[0], y_bounds[1], y_bounds[1]]
        zs = [z, z, z, z]
        verts = [list(zip(xs, ys, zs))]
        poly  = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='gray', linewidth=0.5)
        ax.add_collection3d(poly)
 
    draw_floor(floor_z0, color='#a8d8ea', alpha=0.25)
    draw_floor(floor_z1, color='#a8d8ea', alpha=0.25)
 
    # ------------------------------------------------------------------ #
    #  Draw elevator shaft walls (static, semi-transparent)
    # ------------------------------------------------------------------ #
    for each_elevator_region in elevator_region:
        ex_min, ex_max, ey_min, ey_max = each_elevator_region
        shaft_color = '#f6d365'
        shaft_alpha = 0.18
 
        def shaft_face(xs, ys, zs):
            verts = [list(zip(xs, ys, zs))]
            poly  = Poly3DCollection(verts, alpha=shaft_alpha,
                                     facecolor=shaft_color, edgecolor='orange', linewidth=0.8)
            ax.add_collection3d(poly)
 
        # 4 vertical walls
        shaft_face([ex_min, ex_max, ex_max, ex_min],
                   [ey_min, ey_min, ey_min, ey_min],
                   [floor_z0, floor_z0, floor_z1, floor_z1])
        shaft_face([ex_min, ex_max, ex_max, ex_min],
                   [ey_max, ey_max, ey_max, ey_max],
                   [floor_z0, floor_z0, floor_z1, floor_z1])
        shaft_face([ex_min, ex_min, ex_min, ex_min],
                   [ey_min, ey_max, ey_max, ey_min],
                   [floor_z0, floor_z0, floor_z1, floor_z1])
        shaft_face([ex_max, ex_max, ex_max, ex_max],
                   [ey_min, ey_max, ey_max, ey_min],
                   [floor_z0, floor_z0, floor_z1, floor_z1])
 
    # ------------------------------------------------------------------ #
    #  Draw static obstacles on floor_z0
    # ------------------------------------------------------------------ #
    if obstacles is not None:
        for obs in obstacles:
            if len(obs) == 4:    # circle
                cx, cy, cz, r = obs       
                if cz != floor_z0:
                    continue  # Only draw circular obstacles on the lower floor
                theta_circ = np.linspace(0, 2 * np.pi, 32)
                xs = cx + r * np.cos(theta_circ)
                ys = cy + r * np.sin(theta_circ)
                z_bot = cz
                z_top = 6.0
 
                # Side wall — one quad strip per segment
                for j in range(len(theta_circ) - 1):
                    face = [
                        [xs[j],   ys[j],   z_bot],
                        [xs[j+1], ys[j+1], z_bot],
                        [xs[j+1], ys[j+1], z_top],
                        [xs[j],   ys[j],   z_top],
                    ]
                    poly = Poly3DCollection([face], alpha=0.5,
                                           facecolor='dimgray', edgecolor='none')
                    ax.add_collection3d(poly)
 
                # Bottom cap
                verts = [list(zip(xs, ys, np.full_like(xs, z_bot)))]
                ax.add_collection3d(Poly3DCollection(verts, alpha=0.65,
                                    facecolor='dimgray', edgecolor='black', linewidth=0.8))
 
                # Top cap
                verts = [list(zip(xs, ys, np.full_like(xs, z_top)))]
                ax.add_collection3d(Poly3DCollection(verts, alpha=0.65,
                                    facecolor='dimgray', edgecolor='black', linewidth=0.8))
 
            elif len(obs) == 5:         # rectangle obstacle
                ox, oy, oth, ow, ol = obs
                faces = make_box_faces(ox, oy, floor_z0, ol, ow, 0.3, oth, center_point=center_point)
                poly  = Poly3DCollection(faces, alpha=0.5, facecolor='dimgray', edgecolor='black')
                ax.add_collection3d(poly)
 
    # ------------------------------------------------------------------ #
    # Draw static floors
    # ------------------------------------------------------------------ #
    def draw_floor(z, alpha=0.18, color='gray'):
        xs = [x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0]]
        ys = [y_bounds[0], y_bounds[0], y_bounds[1], y_bounds[1]]
        zs = [z, z, z, z]
        verts = [list(zip(xs, ys, zs))]
        poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
        ax.add_collection3d(poly)

    draw_floor(floor_z0, color='lightgray', alpha=0.25)
    draw_floor(floor_z1, color='gray', alpha=0.18)

    tab10  = plt.cm.tab10.colors
    # Use function arg `N_elevator` if provided, otherwise fall back to `args.N_elevator`.
    if N_elevator == 0:
        N_elevator = int(getattr(args, 'N_elevator', 0))

    red = np.array([1.0, 0.0, 0.0])
    blue = np.array([0.0, 0.44705882, 0.74117647])
    robot_colors = np.array([red if i < N_elevator else blue for i in range(N_robots)])
    colors = robot_colors
    robot_height =  args.car_height if hasattr(args, 'car_height') else 0.2
 
    # ------------------------------------------------------------------ #
    #  Initialise per-robot mutable artists
    # ------------------------------------------------------------------ #
    robot_polys  = []   # Poly3DCollection per robot
    robot_polys = []
    trail_segments = []

    trail_length = T
    trail_width = 7.0
    max_alpha = 0.35     

    for i in range(N_robots):
        x0, y0, z0 = positions[i, 0]
        th0 = thetas[i, 0]

        faces = create_car_3d(x0, y0, z0 + robot_height / 2, lengths[i], widths[i], robot_height, th0, center_point=center_point)
        poly = Poly3DCollection(faces, alpha=0.9, facecolor=robot_colors[i], edgecolor='black', linewidth=0.8)
        ax.add_collection3d(poly)
        robot_polys.append(poly)

        segs = []
        for _ in range(trail_length):
            line, = ax.plot([], [], [], color=robot_colors[i], lw=trail_width, alpha=0, solid_capstyle='projecting')
            segs.append(line)

        trail_segments.append(segs)
 
    # Goal markers
    if unassigned_goals:
        # same color for all goals if unassigned
        colors = np.array([tab10[7]] * N_robots)
    for i in range(N_robots):
        gx, gy, gz = goal_positions[i,:3]
        ax.scatter([gx], [gy], [gz], marker='*',
                   color=colors[i], s=200, edgecolors='k', zorder=10, depthshade=False)
 
    def init():
        for segs in trail_segments:
            for seg in segs:
                seg.set_data_3d([], [], [])
                seg.set_alpha(0)

        return robot_polys + [seg for segs in trail_segments for seg in segs]
 
    # ------------------------------------------------------------------ #
    #  Update function
    # ------------------------------------------------------------------ #
    def update(frame):
        for i in range(N_robots):
            x, y, z = positions[i, frame]
            th = thetas[i, frame]

            # -----------------------------
            # Update robot
            # -----------------------------
            faces = create_car_3d(x, y, z + robot_height / 2, lengths[i], widths[i], robot_height, th, center_point=center_point)
            robot_polys[i].set_verts(faces)

            # -----------------------------
            # Update fading trail
            # -----------------------------
            start = max(0, frame - trail_length)
            visible = frame - start

            for k, t in enumerate(range(start, frame)):
                trail_segments[i][k].set_data_3d(
                    positions[i, t:t+2, 0],
                    positions[i, t:t+2, 1],
                    positions[i, t:t+2, 2],
                )

                alpha = (k + 1) / trail_length
                trail_segments[i][k].set_alpha(max_alpha * alpha)

            # Hide unused trail segments
            for k in range(visible, trail_length):
                trail_segments[i][k].set_data_3d([], [], [])
                trail_segments[i][k].set_alpha(0)

        ax.set_title(f"Time step {frame+1}/{T}", y = 0.98)

        return robot_polys + [seg for segs in trail_segments for seg in segs]
 
    ani = FuncAnimation(
        fig, update, frames=T,
        interval=interval, blit=False
    )
    plt.tight_layout()
 
    if save:
        try:
            ani.save("video3d.mp4", writer='ffmpeg', dpi=150)
            print("Saved video3d.mp4")
        except Exception as e:
            print(f"Failed to save: {e}")
            plt.show()
    else:
        plt.show()



def animate_denoising_trajectories_3d(
    control_history,
    initial_states,
    args,
    lengths=None,
    widths=None,
    obstacles=None,
    goal_positions=None,
    elevator_region=None,
    N_elevator=0,
    rollout_fn=None,
    save=False,
    interval=200,
    x_bounds=(-30.0, 30.0),
    y_bounds=(-30.0, 30.0),
    floor_z0=0.0,
    floor_z1=5.0,
    center_point=False,
):
    """
    Animate the evolution of predicted trajectories during denoising
    for a multi-floor 3D multi-robot scenario.

    The robots remain fixed at their initial states while their predicted
    trajectories evolve across denoising steps.

    Args:
        control_history:
            Control trajectories across denoising steps.
            Expected shape:
                (N_denoising_steps, N_agents, T, dim_u)

        initial_states:
            Initial robot states.
            Expected shape:
                (N_agents, state_dim)
            State format should contain at least [x, y, z, theta].

        args:
            Scenario configuration object.

        lengths:
            Vehicle lengths, one per robot.

        widths:
            Vehicle widths, one per robot.

        obstacles:
            Static obstacles.

        goal_positions:
            Goal positions, shape (N_agents, 3).

        elevator_region:
            List of elevator regions, each represented as:
            (x_min, x_max, y_min, y_max).

        N_elevator:
            Number of robots that can use the elevator.

        rollout_fn:
            Function used to roll out the control trajectories.

        save:
            If True, save the animation to '3d_denoising.mp4'.

        interval:
            Time between animation frames in milliseconds.

        x_bounds:
            X-axis limits.

        y_bounds:
            Y-axis limits.

        floor_z0:
            Height of the first floor.

        floor_z1:
            Height of the second floor.

        center_point:
            Whether the robot reference point is at its center.
            If False, the reference point is at the front axle.
    """

    # ================================================================
    # Data preparation
    # ================================================================

    traj_history = rollout_denoising_history(
        control_history,
        initial_states,
        rollout_fn=rollout_fn
    )

    traj_history = np.asarray(traj_history)
    initial_states = np.asarray(initial_states)

    n_steps, N_agents, T, _ = traj_history.shape

    # Trajectory positions
    # Expected format: x, y, z, theta, ...
    positions = traj_history[:, :, :, :3]

    # Initial robot positions
    initial_positions = initial_states[:, :3]

    # Initial robot orientations
    initial_thetas = initial_states[:, 3]

    # ================================================================
    # Robot dimensions
    # ================================================================

    if lengths is None:
        lengths = np.array(
            [getattr(args, "car_length", 5.0)] * N_agents
        )
    else:
        lengths = np.asarray(lengths)

    if widths is None:
        widths = np.array(
            [getattr(args, "car_width", 2.0)] * N_agents
        )
    else:
        widths = np.asarray(widths)

    robot_height = getattr(args, "car_height", 0.175)

    # ================================================================
    # Goal positions
    # ================================================================

    if goal_positions is None:
        goal_positions = np.zeros((N_agents, 3))
    else:
        goal_positions = np.asarray(goal_positions)

    # ================================================================
    # Figure and axes
    # ================================================================

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(x_bounds[0], x_bounds[1])
    ax.set_ylim(y_bounds[0], y_bounds[1])
    ax.set_zlim(floor_z0 - 0.5, floor_z1 + 1.0)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")


    # ================================================================
    # Colors
    # ================================================================

    if N_elevator == 0:
        N_elevator = int(getattr(args, "N_elevator", 0))

    red = np.array([1.0, 0.0, 0.0])
    blue = np.array([0.0, 0.44705882, 0.74117647])

    robot_colors = np.array([
        red if i < N_elevator else blue
        for i in range(N_agents)
    ])

    # ================================================================
    # Draw floors
    # ================================================================

    def draw_floor(z, alpha=0.18, color="gray"):
        xs = [
            x_bounds[0],
            x_bounds[1],
            x_bounds[1],
            x_bounds[0]
        ]

        ys = [
            y_bounds[0],
            y_bounds[0],
            y_bounds[1],
            y_bounds[1]
        ]

        zs = [z, z, z, z]

        verts = [list(zip(xs, ys, zs))]

        poly = Poly3DCollection(
            verts,
            alpha=alpha,
            facecolor=color,
            edgecolor="none"
        )

        ax.add_collection3d(poly)

    draw_floor(
        floor_z0,
        color="lightgray",
        alpha=0.25
    )

    draw_floor(
        floor_z1,
        color="gray",
        alpha=0.18
    )

    # ================================================================
    # Draw elevator shaft walls
    # ================================================================

    if elevator_region is not None:

        for each_elevator_region in elevator_region:

            ex_min, ex_max, ey_min, ey_max = each_elevator_region

            shaft_color = "#f6d365"
            shaft_alpha = 0.18

            def shaft_face(xs, ys, zs):

                verts = [
                    list(zip(xs, ys, zs))
                ]

                poly = Poly3DCollection(
                    verts,
                    alpha=shaft_alpha,
                    facecolor=shaft_color,
                    edgecolor="orange",
                    linewidth=0.8
                )

                ax.add_collection3d(poly)

            # Front wall
            shaft_face(
                [ex_min, ex_max, ex_max, ex_min],
                [ey_min, ey_min, ey_min, ey_min],
                [floor_z0, floor_z0, floor_z1, floor_z1]
            )

            # Back wall
            shaft_face(
                [ex_min, ex_max, ex_max, ex_min],
                [ey_max, ey_max, ey_max, ey_max],
                [floor_z0, floor_z0, floor_z1, floor_z1]
            )

            # Left wall
            shaft_face(
                [ex_min, ex_min, ex_min, ex_min],
                [ey_min, ey_max, ey_max, ey_min],
                [floor_z0, floor_z0, floor_z1, floor_z1]
            )

            # Right wall
            shaft_face(
                [ex_max, ex_max, ex_max, ex_max],
                [ey_min, ey_max, ey_max, ey_min],
                [floor_z0, floor_z0, floor_z1, floor_z1]
            )

    # ================================================================
    # Draw static obstacles
    # ================================================================

    if obstacles is not None:

        for obs in obstacles:

            # --------------------------------------------------------
            # Circular obstacle
            # --------------------------------------------------------

            if len(obs) == 4:
                cx, cy, cz, r = obs
                if cz != floor_z0:
                    continue  # Only draw circular obstacles on the lower floor
                theta_circ = np.linspace(
                    0,
                    2 * np.pi,
                    32
                )

                xs = cx + r * np.cos(theta_circ)
                ys = cy + r * np.sin(theta_circ)

                z_bot = cz
                z_top = cz + 6.0

                # Cylindrical side
                for j in range(len(theta_circ) - 1):

                    face = [
                        [xs[j], ys[j], z_bot],
                        [xs[j + 1], ys[j + 1], z_bot],
                        [xs[j + 1], ys[j + 1], z_top],
                        [xs[j], ys[j], z_top]
                    ]

                    poly = Poly3DCollection(
                        [face],
                        alpha=0.5,
                        facecolor="dimgray",
                        edgecolor="none"
                    )

                    ax.add_collection3d(poly)

                # Bottom cap
                verts = [
                    list(
                        zip(
                            xs,
                            ys,
                            np.full_like(xs, z_bot)
                        )
                    )
                ]

                ax.add_collection3d(
                    Poly3DCollection(
                        verts,
                        alpha=0.65,
                        facecolor="dimgray",
                        edgecolor="black"
                    )
                )

                # Top cap
                verts = [
                    list(
                        zip(
                            xs,
                            ys,
                            np.full_like(xs, z_top)
                        )
                    )
                ]

                ax.add_collection3d(
                    Poly3DCollection(
                        verts,
                        alpha=0.65,
                        facecolor="dimgray",
                        edgecolor="black"
                    )
                )

            # --------------------------------------------------------
            # Rectangular obstacle
            # --------------------------------------------------------

            elif len(obs) == 5:

                ox, oy, oth, ow, ol = obs

                faces = make_box_faces(
                    ox,
                    oy,
                    floor_z0,
                    ol,
                    ow,
                    0.3,
                    oth,
                    center_point=center_point
                )

                poly = Poly3DCollection(
                    faces,
                    alpha=0.5,
                    facecolor="dimgray",
                    edgecolor="black"
                )

                ax.add_collection3d(poly)

    # ================================================================
    # Initialize robots at initial states
    # ================================================================

    robot_polys = []

    for i in range(N_agents):

        x0, y0, z0 = initial_positions[i]
        th0 = initial_thetas[i]

        faces = create_car_3d(
            x0,
            y0,
            z0 + robot_height / 2,
            lengths[i],
            widths[i],
            robot_height,
            th0,
            center_point=center_point
        )

        poly = Poly3DCollection(
            faces,
            alpha=0.9,
            facecolor=robot_colors[i],
            edgecolor="black",
            linewidth=0.8
        )

        ax.add_collection3d(poly)

        robot_polys.append(poly)

    # ================================================================
    # Create trajectory lines
    # ================================================================

    traj_lines = []

    for i in range(N_agents):

        line, = ax.plot(
            [],
            [],
            [],
            "-",
            color=robot_colors[i],
            linewidth=7.0,
            alpha=0.8
        )

        traj_lines.append(line)

    # ================================================================
    # Goal markers
    # ================================================================

    goal_markers = []

    for i in range(N_agents):

        gx, gy, gz = goal_positions[i, :3]

        marker, = ax.plot(
            [gx],
            [gy],
            [gz],
            marker="*",
            color=robot_colors[i],
            markersize=12,
            markeredgecolor="black",
            linestyle="None"
        )

        goal_markers.append(marker)

    # ================================================================
    # Animation artists
    # ================================================================

    artists = (
        robot_polys
        + traj_lines
        + goal_markers
    )

    # ================================================================
    # Initialization
    # ================================================================

    def init():

        for line in traj_lines:

            line.set_data_3d(
                [],
                [],
                []
            )

        return artists

    # ================================================================
    # Update
    # ================================================================

    def update(frame):

        frame_traj = positions[frame]

        for i in range(N_agents):

            # --------------------------------------------------------
            # Current predicted trajectory
            # --------------------------------------------------------

            path = np.concatenate(
                [
                    initial_positions[i:i + 1],
                    frame_traj[i, :, :]
                ],
                axis=0
            )

            traj_lines[i].set_data_3d(
                path[:, 0],
                path[:, 1],
                path[:, 2]
            )

            # --------------------------------------------------------
            # Keep robot fixed at its initial state
            # --------------------------------------------------------

            x0, y0, z0 = initial_positions[i]
            th0 = initial_thetas[i]

            faces = create_car_3d(
                x0,
                y0,
                z0 + robot_height / 2,
                lengths[i],
                widths[i],
                robot_height,
                th0,
                center_point=center_point
            )

            robot_polys[i].set_verts(faces)

        ax.set_title(
            f"Denoising step {frame + 1}/{n_steps}", y = 0.98
        )

        return artists

    # ================================================================
    # Create animation
    # ================================================================

    ani = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        init_func=init,
        interval=interval,
        blit=False
    )
    plt.tight_layout()

    # ================================================================
    # Save or display
    # ================================================================

    if save:

        try:

            ani.save(
                "3d_denoising.mp4",
                writer="ffmpeg",
                dpi=150
            )

            print("Saved 3d_denoising.mp4")

        except Exception as e:

            print(f"Failed to save animation: {e}")

            plt.show()

    else:

        plt.show()

