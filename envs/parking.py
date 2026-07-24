
from envs.helper.rect_collision import inter_robot_collision, count_inter_robot_collisions, obstacle_collision, count_obstacle_collisions
from envs.base_env import BaseEnv
import jax.numpy as jnp
import jax

class EnvParking(BaseEnv):
    def __init__(self, args, rollout_fn=None):
        super().__init__()
        self.args = args
        self.rollout_fn = rollout_fn
        self.lengths = jnp.full((args.N,), args.car_length)
        self.widths = jnp.full((args.N,), args.car_width)
        self.S = jnp.array([args.S1, args.S2])
        self.goal_tolerance = args.goal_tolerance
        self.xbounds = [-17.5, 17.5]
        self.ybounds = [-5.0, 30.0]
        self.random_x_min = -15.0
        self.random_x_max = 15.0
        self.random_y_min = 17.5
        self.random_y_max = 25.0

        self.is_circular = False
        self.center = False
        self.obs = None
        self.states = None
        self.goals = None

    
    ### Distributed cost function for robot k ###
    def build_distributed_cost(self):
        """Build the distributed parking objective for robot k."""
        @jax.jit
        def cost_fn(u_sampled, x0_k, team_trajs, robot_idx, goal_loc):
            """
            u_sampled: (T, dim_u) - robot k's control inputs
            x0_k: (state_dim,) - robot k's initial state        
            team_trajs: (N, T, state_dim) - trajectories of the whole team
            robot_idx: int - index of the robot k
            goal_loc: (state_dim,) - robot k's desired goal location
            """
            # Rollout the trajectory for robot k
            my_traj = self.rollout_fn(x0_k[None, :], u_sampled[None, :, :])[0]
            
            # Penalize velocity
            vel_penal = jnp.sum(jnp.square(my_traj[:, 3]))
            mask = jnp.arange(team_trajs.shape[0]) != robot_idx

            # Penalize collisions against the rest of the team by vmaping collision check over time T
            inter_col_penal = jnp.sum(jax.vmap(inter_robot_collision, in_axes=(0, 0, None, None, None, None))(
                team_trajs[:, :, :3].transpose(1, 0, 2),
                my_traj[:,:3],
                mask,  # (N, 1)
                self.widths,
                self.lengths,
                robot_idx,
            ))
            # Penalize obstacle collisions
            obs_col_penal = jnp.sum(jax.vmap(obstacle_collision, in_axes=(0, None, None, None))(
                my_traj[:,:3],  
                self.obs, 
                self.widths[robot_idx], 
                self.lengths[robot_idx]
            ))
            # Penalize terminal distance to goal
            dist_goal = my_traj[-1, :2] - goal_loc
            terminal_cost = jnp.sum(self.S*jnp.square(dist_goal))

            # Penalize boundary violations (so that robots stay within the parking lot)
            boundary_penal_y = jnp.sum(jnp.where(my_traj[:, 1] >= 25.0, 10.0, 0.0))
            boundary_penal_x1 = jnp.sum(jnp.where(my_traj[:, 0] <= -15.0, 10.0, 0.0))
            boundary_penal_x2 = jnp.sum(jnp.where(my_traj[:, 0] >= 15.0, 10.0, 0.0))
            boundary_penal = boundary_penal_y + boundary_penal_x1 + boundary_penal_x2

            return -(self.args.Q*vel_penal + terminal_cost + self.args.R*inter_col_penal +  self.args.P*jnp.linalg.norm(u_sampled) + self.args.O*obs_col_penal + boundary_penal)
        return cost_fn

    def generate_states(self, seed = 42):
        """Generate a valid initial state of robot 2, ensuring no collisions with obstacles or other robots."""
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)

        robot_2_state =  [
            jax.random.uniform(k1, (), minval=self.random_x_min, maxval=self.random_x_max),
            jax.random.uniform(k2, (), minval=self.random_y_min, maxval=self.random_y_max),
            jax.random.uniform(k3, (), minval=0.0, maxval=2*jnp.pi),
            0.0
        ]
        self.states = jnp.array([ [-5.0, 10.0, 3*jnp.pi/2, 0.0], robot_2_state ])
        return self.states

    def generate_goals(self, seed = 42):
        self.goals =  jnp.array([[-5.0, 14.0],  [ -5.0, 5.0] ])
        return self.goals


    def get_obs(self, parking_cols=8, space_w=3.5, space_l=7.0,
                            occupied_spaces=None, parking_start_y=0.0, obstacle_radius=1.5):
        """
        Compute circular obstacle locations for a parking scenario.
        """
        if occupied_spaces is None:
            occupied_spaces = [1,2,4,5,6,7,8,9,10,12,13,14,15, 16]

        parking_start_x = -(parking_cols * space_w) / 2
        obstacle_locations = []

        for space_num in occupied_spaces:
            idx = space_num - 1
            row = idx // parking_cols
            col = idx % parking_cols

            cx = parking_start_x + (col + 0.5) * space_w
            cy = parking_start_y + (row + 0.5) * space_l

            # three circles per parking slot
            for dy in [1.75, -1.75]:
                obstacle_locations.append([cx, cy + dy, obstacle_radius])

        # Add bottom row of circles
        bottom_y = parking_start_y -0.5* space_l+ 1.75
        for col in range(parking_cols):
            cx = parking_start_x + (col + 0.5) * space_w
            obstacle_locations.append([cx, bottom_y, obstacle_radius])
        self.obs = jnp.array(obstacle_locations)
        return  self.obs

    def check_results(self, traj):
        '''Check if all robots reached their goals without collisions. 
        Returns a dictionary with success status and number of collisions.
        success = 1 if all robots reached their goals without collisions, 0 otherwise.'''
        col_nums = count_inter_robot_collisions(traj, self.lengths, self.widths)
        obs_col_nums = count_obstacle_collisions(traj, self.obs, self.lengths, self.widths)
        reached = jnp.all(jnp.linalg.norm(traj[:, -1, :2] - self.goals, axis=-1) <= self.goal_tolerance)
        return {
                "success": int(reached and jnp.sum(col_nums) == 0 and jnp.sum(obs_col_nums) == 0),
                "goal_reached": reached,
                "num_collisions": int(jnp.sum(col_nums)),
                }
