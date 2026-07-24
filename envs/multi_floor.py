
from envs.helper.rect_collision_3d import goal_successes, inter_robot_collision, obstacle_collision,count_inter_robot_collisions, count_obstacle_collisions
from robots.kinematic_bicycle import buffered_kinematic_bicycle_rollout, elevator_kinematic_bicycle_rollout
from envs.plotter.animate_3d import animate_trajectories_3d
from envs.base_env import BaseEnv
import jax.numpy as jnp
import jax
import numpy as np

class EnvMultiFloor(BaseEnv):
    def __init__(self, args, rollout_fn=None):
        super().__init__()
        self.args = args
        self.rollout_fn = rollout_fn

        self.lengths = jnp.concatenate([
            jnp.full((args.N_elevator,), args.car_length),
            jnp.full((args.N_no_elevator,), args.car_length2),
        ])
        self.widths = jnp.concatenate([
            jnp.full((args.N_elevator,), args.car_width),
            jnp.full((args.N_no_elevator,), args.car_width2),
        ])

        self.car_height = args.car_height
        self.x_mins = jnp.array([args.x_min1, args.x_min2, args.x_min3, args.x_min4])
        self.x_maxs = jnp.array([args.x_max1, args.x_max2, args.x_max3, args.x_max4])
        self.y_mins = jnp.array([args.y_min1, args.y_min2, args.y_min3, args.y_min4])
        self.y_maxs = jnp.array([args.y_max1, args.y_max2, args.y_max3, args.y_max4])

        self.elevator_regions = [
            (self.args.x_min1, self.args.x_max1, self.args.y_min1, self.args.y_max1),
            (self.args.x_min2, self.args.x_max2, self.args.y_min2, self.args.y_max2),
            (self.args.x_min3, self.args.x_max3, self.args.y_min3, self.args.y_max3),
            (self.args.x_min4, self.args.x_max4, self.args.y_min4, self.args.y_max4)
        ]
        self.xrange = [-30.0, 30.0]
        self.yrange = [-30.0, 30.0]
        self.elevator_bounds = [self.x_mins, self.x_maxs, self.y_mins, self.y_maxs]
        self.S = jnp.array([args.S1, args.S2, args.S3])

        self.goal_tolerance = args.goal_tolerance
        
        self.is_3d = True
        self.obs = None
        self.states = None
        self.goals = None

    ### Distributed cost function for robot k ###
    def build_distributed_cost(self):
        """Build the distributed multi-floor objective for robot k."""
        @jax.jit
        def cost_fn(u_sampled, x0_k, team_trajs, robot_idx, goal_loc):
            """
            u_sampled: (T, dim_u) - robot k's control inputs
            x0_k: (state_dim,) - robot k's initial state        
            team_trajs: (N, T, state_dim) - trajectories of the whole team
            robot_idx: int - index of the robot k
            goal_loc: (state_dim,) - robot k's desired goal location
            """
            # Use conditional logic to select rollout type based on robot index. 
            # If the robot is an elevator robot (k < args.N_elevator), use the elevator rollout; otherwise, use the buffered rollout.
            my_traj = jax.lax.cond(
                robot_idx < self.args.N_elevator,
                lambda _: elevator_kinematic_bicycle_rollout(
                    x0_k[None, :],
                    u_sampled[None, :, :],
                    x_min=self.x_mins,
                    x_max=self.x_maxs,
                    y_min=self.y_mins,
                    y_max=self.y_maxs,
                    dt = self.args.dt
                ),
                lambda _: buffered_kinematic_bicycle_rollout(
                    x0_k[None, :],
                    u_sampled[None, :, :],
                    dt=self.args.dt
                ),
                operand=None
            )[0]
            
            # Penalize velocity
            vel_penal = jnp.sum(jnp.square(my_traj[:, 4]))


            # Penalize collisions against the rest of the team by vmaping collision check over time T
            mask = jnp.arange(team_trajs.shape[0]) != robot_idx
            inter_col_penal = jnp.sum(jax.vmap(inter_robot_collision, in_axes=(0, 0, None, None, None, None))(
                team_trajs[:, :, :4].transpose(1, 0, 2), 
                my_traj[:,:4],
                mask, 
                self.widths,
                self.lengths,
                robot_idx
            ))

            # Penalize collisions between robot k and obstacles
            obs_col_penal = jnp.sum(jax.vmap(obstacle_collision, in_axes=(0, None, None, None, None))(
                my_traj[:,:4],    
                self.obs, 
                self.widths[robot_idx], 
                self.lengths[robot_idx],
                self.car_height
            ))

            # Compute distance to goal and terminal cost
            dist_goal = self.S*(my_traj[:,:3] - goal_loc)
            terminal_cost = jnp.sum(jnp.linalg.norm(dist_goal[-25:], axis=1))

            return -(self.args.Q*vel_penal + terminal_cost + self.args.R*inter_col_penal + self.args.P*jnp.linalg.norm(u_sampled) + self.args.O*obs_col_penal)
        return cost_fn


    ### Centralized cost function for all robots ###
    def build_centralized_cost(self):
        """Build the centralized multi-floor objective for all robots."""
        @jax.jit
        def cost_fn(u_sampled, x0, goal_loc):
            """
            u_sampled: (N, T, dim_u) - Control inputs for all robots
            x0: (N, state_dim) initial state of all robots
            goal_loc: (N, state_dim) desired goal locations for all robots
            """
            
            # Rollout the team with the given control inputs.
            trajs = self.rollout_fn(x0, u_sampled)  
            
            # Penalize velocity
            vel_penal = jnp.sum(trajs[:, :, 4] ** 2)

            trajs_t = jnp.swapaxes(trajs[:, :, :4], 0, 1)  # (T, N, 4)

            def inter_step(carry, pos_t):
                def single_robot_penalty(robot_idx):
                    mask = jnp.arange(self.args.N) != robot_idx
                    return inter_robot_collision(
                        pos_t,
                        pos_t[robot_idx],
                        mask,
                        self.widths,
                        self.lengths,
                        robot_idx
                    )
                step_penal = jax.vmap(single_robot_penalty)(jnp.arange(self.args.N))
                return carry + jnp.sum(step_penal), None

            # Compute inter-robot collision penalty over the whole rollout
            inter_col_penal, _ = jax.lax.scan(inter_step, 0.0, trajs_t)
            inter_col_penal /= self.args.N

            def obs_col_penal_single(robot_idx):
                def per_timestep(pos_t):
                    return obstacle_collision(
                        pos_t[robot_idx],
                        self.obs,
                        self.widths[robot_idx],
                        self.lengths[robot_idx],
                        self.args.car_height,
                    )
                penalties = jax.vmap(per_timestep)(trajs_t[:, :, :4])
                return jnp.sum(penalties)

            # Compute obstacle collision penalty over the whole rollout
            obs_col_penal = jnp.sum(jax.vmap(obs_col_penal_single)(jnp.arange(trajs.shape[0])))

            # Compute terminal cost over the last 25 timesteps
            dist_goal = self.S*(trajs[:, :, :3] - goal_loc[:, None, :])
            terminal_cost = jnp.sum(
                        jnp.linalg.norm(dist_goal[:, -25:, :], axis=-1)
                    )

            # Compute the norm of the control inputs for regularization
            control_norm = jnp.sum(jnp.linalg.norm(u_sampled, axis=(1, 2)))

            return -(self.args.Q*vel_penal + terminal_cost + self.args.R*inter_col_penal + self.args.P*control_norm + self.args.O*obs_col_penal)
        return cost_fn
    
    def get_obs(self, obstacle_radius=1.0):
        obstacle_locations = [
                            (-20,0,0,obstacle_radius), (-20,0,5,obstacle_radius),
                            (20,0,0,obstacle_radius), (20,0,5,obstacle_radius),
                            (0,-20,0,obstacle_radius), (0,-20,5,obstacle_radius),
                            (0,20,0,obstacle_radius), (0,20,5,obstacle_radius)]

        self.obs = jnp.array(obstacle_locations)
        return    self.obs 


    def generate_goals(self, z_top=5, z_bottom=0, x_range=(-20, 20), y_range=(-20, 20),
                            min_dist_between=7.0, min_dist_obstacle=4.0, min_dist_elevator=3.0, seed=42):
        """
        Randomly generate 6 goals, 2-5 on top (z=5), rest on bottom (z=0),
        no overlap with each other, obstacles, or elevator.
        obstacles: jnp.array of shape (num_obs, 3) with [x, y, radius]
        elevator_bounds: [x_mins, x_maxs, y_mins, y_maxs] (each shape (2,))
        Returns: (6, 3) array of [x, y, z] goals
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Randomly choose how many on top


        # the first three can be on the top, the fourth is on bottom, the fifth and sixth are on top.
        max_top_N = self.args.N//2
        zs = [z_top for _ in range(max_top_N)] + [z_bottom for _ in range(self.args.N - max_top_N)]
        goals = []
        attempts = 0
        max_attempts = 100000000
        # Convert obstacles to numpy for easier indexing
        obstacles_np = np.array(self.obs)

        # Convert elevator bounds to numpy
        if self.elevator_bounds is not None:
            x_mins, x_maxs, y_mins, y_maxs = [np.array(b) for b in self.elevator_bounds]
        else:
            x_mins = x_maxs = y_mins = y_maxs = None

        def min_dist_to_elevator_rects(xy):
            if x_mins is None:
                return np.inf
            min_dist = np.inf
            for i in range(len(x_mins)):
                # Rectangle bounds
                x_min, x_max = x_mins[i], x_maxs[i]
                y_min, y_max = y_mins[i], y_maxs[i]
                # Clamp point to rectangle
                x_clamp = np.clip(xy[0], x_min, x_max)
                y_clamp = np.clip(xy[1], y_min, y_max)
                dist = np.linalg.norm(xy - np.array([x_clamp, y_clamp]))
                min_dist = min(min_dist, dist)
            return min_dist

        while len(goals) < self.args.N and attempts < max_attempts:
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
            z = zs[len(goals)]
            candidate = np.array([x, y, z])
            # Check distance to elevator rectangles
            if self.elevator_bounds is not None and min_dist_to_elevator_rects(candidate[:2]) < min_dist_elevator:
                attempts += 1
                continue
            # Check distance to other goals
            if any(np.linalg.norm(candidate[:2] - g[:2]) < min_dist_between for g in goals):
                attempts += 1
                continue
            # Check distance to obstacles
            if obstacles_np is not None:
                for obs in obstacles_np:
                    obs_xy = obs[:2]
                    obs_r = obs[2]
                    if np.linalg.norm(candidate[:2] - obs_xy) < (obs_r + min_dist_obstacle):
                        break
                else:
                    goals.append(candidate)
                    continue
                attempts += 1
                continue
            # If no obstacles, just add
            goals.append(candidate)
        self.goals = np.stack(goals)
        return self.goals

    def generate_states(self,
        seed =42,
        z=0.0):
        """Sample initial rob        xrange,
        yrange,
        lengths,
        N,ot states in safe free space on the bottom floor.

        The returned states are [x, y, z, theta, v] with z fixed to the zeroth floor.
        Candidates are rejected if they:
        - lie too close to any elevator footprint
        - lie too close to any circular obstacle
        - violate pairwise spacing with already accepted robots
        """
        rng = np.random.default_rng(seed)
        initial_states = []
        obstacles = np.asarray(self.obs)

        def candidate_is_safe(x, y, i):
            half_len = 0.5 * float(self.lengths[i])

            # Keep the robot inside the world bounds with a margin.
            if not (
                self.xrange[0] + half_len < x < self.xrange[1] - half_len
                and self.yrange[0] + half_len < y < self.yrange[1] - half_len
            ):
                return False

            # Avoid elevator footprints with a conservative margin.
            for region in self.elevator_regions:
                ex_min, ex_max, ey_min, ey_max = region
                if (
                    ex_min - half_len <= x <= ex_max + half_len
                    and ey_min - half_len <= y <= ey_max + half_len
                ):
                    return False

            # Avoid circular obstacles.
            if obstacles is not None:
                for obs in obstacles:
                    if len(obs) < 4:
                        continue
                    ox, oy, _, r = obs
                    if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < r + half_len:
                        return False

            # Keep away from previously sampled robots on the same floor.
            for j, prev_state in enumerate(initial_states):
                min_sep = float(self.lengths[i] + self.lengths[j])
                if np.linalg.norm(np.array([x, y]) - prev_state[:2]) < min_sep:
                    return False

            return True

        for i in range(self.args.N):
            for attempts in range(10000):
                x = rng.uniform(*self.xrange)
                y = rng.uniform(*self.yrange)
                theta = rng.uniform(0, 2 * np.pi)
                v = 0.0

                if not candidate_is_safe(x, y, i):
                    continue

                initial_states.append(np.array([x, y, z, theta, v]))
                break
            else:
                raise RuntimeError(f"Could not sample a valid initial state for robot {i}.")
        self.states = jnp.array(initial_states)
        return self.states


    ### Visualization and result checking functions ####
    def visualize(self, traj, save=False):
        animate_trajectories_3d(traj, self.args, lengths = self.lengths, widths=self.widths, obstacles=self.obs, goal_positions=self.goals, N_elevator = self.args.N_elevator, elevator_region= self.elevator_regions, save=save, interval=30, unassigned_goals = False, center_point=self.center)

    def check_results(self, traj):
        '''Check if all robots reached their goals without collisions. 
        Returns a dictionary with success status and number of collisions.
        success = 1 if all robots reached their goals (within tolerance in the same floor) without collisions, 0 otherwise.'''
        col_nums = count_inter_robot_collisions(traj, self.lengths, self.widths)
        obs_col_nums = count_obstacle_collisions(traj, self.obs, self.args, self.lengths, self.widths)
        reached = goal_successes(traj, self.goals, self.goal_tolerance)
        return {
                "success": int(reached and jnp.sum(col_nums) == 0 and jnp.sum(obs_col_nums) == 0),
                "goal_reached": reached,
                "num_collisions": int(jnp.sum(col_nums)),
                }
