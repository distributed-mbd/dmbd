
from envs.helper.rect_collision import faster_inter_robot_collision, faster_count_inter_robot_collisions, obstacle_collision, count_obstacle_collisions
from envs.base_env import BaseEnv
import jax.numpy as jnp
import jax

class EnvRushHour(BaseEnv):
    def __init__(self, args, rollout_fn=None):
        super().__init__()
        self.args = args
        self.rollout_fn = rollout_fn
        self.lengths = jnp.array([args.car_length,args.car_length2, args.car_length3, args.car_length4]) + args.buffer_length
        self.widths = jnp.full((args.N,), args.car_width)
        self.goal_tolerance = args.goal_tolerance

        self.xbounds = [-1.0, 18.0]
        self.ybounds = [-2.0, 12.0]
        self.ratio = 0.5  # ratio for defining the center of the rectangle

        self.center = True
        self.obs = None
        self.states = None
        self.goals = None


        self.random_x_min=2.5
        self.random_x_max = 10.5
        self.random_y_min = 3.0
        self.random_y_max = 6.5

    
    ### Distributed cost function for robot k ###
    def build_distributed_cost(self):
        """Build the distributed rush hour objective for robot k."""
        @jax.jit
        def cost_fn(u_sampled, x0_k, team_trajs, robot_idx, goal_loc):
            """
            u_sampled: (T, dim_u) - robot k's control inputs
            x0_k: (state_dim,) - robot k's initial state        
            team_trajs: (N, T, state_dim) - trajectories of the whole team
            robot_idx: int - index of the robot k
            goal_loc: (state_dim,) - robot k's desired goal location
            """
            
            # Rollout robot k's trajectory
            my_traj = self.rollout_fn(x0_k[None, :], u_sampled[None, :, :])[0]

            # Penalize velocity
            vel_penal = jnp.sum(jnp.square(my_traj[:, 3]))
            S = jnp.where(robot_idx ==0, self.args.S, 0.0)
            # Penalize collisions against the rest of the team by vmaping collision check over time T
            mask = jnp.arange(team_trajs.shape[0]) != robot_idx
            inter_col_penal = jnp.sum(jax.vmap(faster_inter_robot_collision, in_axes=(0, 0, None, None, None, None, None))(
                team_trajs[:, :, :3].transpose(1, 0, 2),
                my_traj[:,:3],
                mask, 
                self.widths,
                self.lengths,
                robot_idx,
                jnp.array([self.args.R1, self.args.R2, self.args.R3, self.args.R4])
            ))

            # Penalize collisions between robot k and obstacles
            obs_col_penal = jnp.sum(jax.vmap(obstacle_collision, in_axes=(0, None, None, None, None))(
                my_traj[:,:3],   
                self.obs, 
                self.widths[robot_idx], 
                self.lengths[robot_idx],
                0.5                # ratio for defining the center of the rectangle
            ))
            # Compute distance to goal and terminal cost
            dist_goal = my_traj[:, :2] - goal_loc
            final_pos_error = jnp.sum(jnp.linalg.norm(dist_goal[-15:], axis=1))
            terminal_cost = jnp.sum(S*jnp.square(dist_goal)) + S*10*final_pos_error 
            return -(self.args.Q*vel_penal + terminal_cost + inter_col_penal + self.args.P*jnp.linalg.norm(u_sampled) + self.args.O*obs_col_penal)
        return cost_fn


    def generate_states(self, seed = 42, initial_buffer=1.0):

        """Generate a valid initial state for all robots, ensuring no collisions with obstacles or other robots."""
        rng = jax.random.PRNGKey(seed)
        while True:
            rng, k1, k2, k3  = jax.random.split(rng, 4)
            state = jnp.array([[
                [3.0, 8, 0.0, 0.0],  
                [6.5, jax.random.uniform(k1, minval=self.random_y_min, maxval=self.random_y_max), jnp.pi/2, 0.0],   
                [jax.random.uniform(k2, minval=self.random_x_min, maxval=self.random_x_max), 2.5, 0.0, 0.0],      
                [12.75, jax.random.uniform(k3, minval=self.random_y_min, maxval=self.random_y_max), jnp.pi/2, 0.0]  
            ]])
            # Add exactly 1.0 buffer to the lengths of the cars for the initial states
            new_lengths = self.lengths  - self.args.buffer_length + initial_buffer

            transformed_state = state.transpose(1, 0, 2) 
            inter_robot_collision = faster_count_inter_robot_collisions(transformed_state, new_lengths , self.widths)
            obstacle_collision = count_obstacle_collisions(transformed_state, self.obs, new_lengths , self.widths, ratio =0.5)
            if inter_robot_collision + obstacle_collision == 0:
                self.states = state[0]
                return  self.states 


    def generate_goals(self, seed = 42):
        self.goals=  jnp.array([[16, 8], [16, 8], [16, 8], [16, 8]])
        return self.goals


    def get_obs(self, x0=0.0, y0=0.0, space_w=15, space_l=10.0, radius=1.0):
        """
        Returns a "box" of circles forming a square boundary.
        The entrance is created by omitting the top-right-most circle.
        """
        obstacle_locations = []
        step = 2 * radius  # Diameter: circles will touch edge-to-edge

        # 1. Bottom Wall (y = y0)
        # Covers from x0 to x0 + space_w
        for x in jnp.arange(x0, x0 + space_w + step, step):
            obstacle_locations.append([x, y0, radius])

        # 2. Left Wall (x = x0)
        # Covers from y0 + step up to y0 + space_l
        for y in jnp.arange(y0 + step, y0 + space_l + step, step):
            obstacle_locations.append([x0, y,radius])
            
        # 3. Right Wall (x = x0 + space_w)
        # We stop one 'step' short of the top corner to leave the entrance
        for y in jnp.arange(y0 + step, y0 + space_l - step, step):
            obstacle_locations.append([x0 + space_w + radius, y, radius])

        # 4. Top Wall (y = y0 + space_l)
        # We stop one 'step' short of the right corner to leave the entrance
        for x in jnp.arange(x0 + step, x0 + space_w + step, step):
            obstacle_locations.append([x, y0 + space_l,radius])
        self.obs = jnp.array(obstacle_locations)
        return self.obs


    def check_results(self, traj):
        '''Check if all robots reached their goals without collisions. 
        Returns a dictionary with success status and number of collisions.
        success = 1 if the first robot reached its goal without any collisions for all robots, 0 otherwise.'''
        col_nums = faster_count_inter_robot_collisions(traj, self.lengths, self.widths)
        obs_col_nums = count_obstacle_collisions(traj, self.obs, self.lengths, self.widths, ratio =self.ratio)
        reached = jnp.all(jnp.linalg.norm(traj[0, -1, :2] - self.goals[0]) <= self.goal_tolerance)
        return {
                "success": int(reached and jnp.sum(col_nums) == 0 and jnp.sum(obs_col_nums) == 0),
                "goal_reached": reached,
                "num_collisions": int(jnp.sum(col_nums)),
                }
