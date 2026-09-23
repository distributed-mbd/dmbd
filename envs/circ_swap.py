from envs.helper.circ_collision import inter_robot_collision, count_inter_robot_collisions,  vector_inter_robot_collision
from envs.base_env import BaseEnv
import jax.numpy as jnp
import jax

class EnvCircSwap(BaseEnv):
    def __init__(self, args, rollout_fn=None):
        super().__init__()
        self.args = args
        if getattr(args, "R_col", None) is not None:
            self.all_radius = jnp.full((args.N,), args.R_col)
        else:
            self.all_radius = jnp.full((args.N,), 1.5)

        self.rollout_fn = rollout_fn
        self.circle_radius = args.circle_radius
        self.goal_tolerance = args.goal_tolerance

        self.is_circular = True
        self.states = None
        self.goals = None
        self.obs = None

        self.xbounds = [-25.0, 25.0]
        self.ybounds = [-25.0, 25.0]


    ### Distributed cost function for all robots ###
    def build_distributed_cost(self):
        @jax.jit
        def cost_fn(u_sampled, x0_i, team_trajs, robot_idx, goal_loc):
            """
            u_sampled: (T, dim_u) - robot k's control inputs
            x0_k: (state_dim,) - robot k's initial state        
            team_trajs: (N, T, state_dim) - trajectories of the whole team
            robot_idx: int - index of the robot k
            goal_loc: (state_dim,) - robot k's desired goal location
            """
            # Rollout the robot k's trajectory.
            my_traj = self.rollout_fn(x0_i[None, :], u_sampled[None, :, :])[0]
            # Penalize velocity.
            vel_penal = jnp.sum(jnp.square(my_traj[:, 2:4]))
            vel_penal += 100*jnp.sum(jnp.square(my_traj[-50:, 2:4]))

            # Penalize collisions against the rest of the team by vmaping collision check over time T
            robot_radius = self.all_radius[robot_idx]
            mask = jnp.arange(team_trajs.shape[0]) != robot_idx
            radius = self.all_radius + robot_radius
            inter_col_penal = jnp.sum(jax.vmap(inter_robot_collision, in_axes=(0, 0, None, None))(
                team_trajs[:, :, :2].transpose(1, 0, 2), # (T, N, 2)
                my_traj[:,:2],
                radius + self.args.radius_buffer, # (N,1),
                mask
            ))

            # Penalize the final goal mismatch.
            dist_goal = my_traj[:, :2] - goal_loc
            final_pos_error = jnp.sum(
                jnp.linalg.norm(dist_goal[-25:, :], axis=-1)
            )
            terminal_cost = jnp.sum(self.args.S1*jnp.square(dist_goal)) + 100*self.args.S1*final_pos_error
            
            return -(self.args.Q*vel_penal + terminal_cost + self.args.R*inter_col_penal +  self.args.P*jnp.linalg.norm(u_sampled))
        return cost_fn

    ### Centralized cost function for all robots ###
    def build_centralized_cost(self):
        """Build the centralized circular-swap objective for goal-swap scenario"""

        @jax.jit
        def cost_fn(u_sampled, x0, goal_loc):
            """u_sampled: (N, T, dim_u) - Control inputs for all robots
            x0: (N, state_dim) initial state of all robots
            goal_loc: (N, state_dim) desired goal locations for all robots"""

            # Rollout the team with the given control inputs.
            trajs = self.rollout_fn(x0, u_sampled)

            # Penalize velocity
            vel_penal = jnp.sum(jnp.square(trajs[:, :, 2:4]))
            vel_penal += 100*jnp.sum(jnp.square(trajs[:, -50:, 2:4]))

            # Penalize pairwise overlap between robots over the whole rollout.
            # inter_col_penal = count_inter_robot_collisions(trajs[:, :, :2], self.all_radius +  self.args.radius_buffer/2)/ self.args.N
            
            inter_col_penal = jnp.sum(jax.vmap(vector_inter_robot_collision, in_axes=(0, None))(
                trajs[:, :, :2].transpose(1, 0, 2),        # (T, N, 2)
                self.all_radius + self.args.radius_buffer            # (N,1)
            ))

            obs_col_penal = 0.0

            # traj = shape (N, T, state_dim), goal_loc = (N, state_dim)
            # Penalize distance to goal for each robot over the whole rollout.
            dist_goal = trajs[:, :, :2] - goal_loc[:, None, :2]
            final_pos_error = jnp.sum(
                jnp.linalg.norm(dist_goal[:, -25:, :], axis=-1)
            )
            terminal_cost = jnp.sum(self.args.S1*jnp.square(dist_goal)) + 100*self.args.S1*final_pos_error

            # Compute the norm of the control inputs for regularization
            control_norm = jnp.sum(jnp.linalg.norm(u_sampled, axis=(1, 2)))

            return -(self.args.Q*vel_penal + terminal_cost + self.args.R*inter_col_penal + self.args.P*control_norm + self.args.O*obs_col_penal)
        return cost_fn


    def setup_goal_swap(self):
        """
        Generate initial states and goal locations for the circular swap scenario.
        """

        angles = jnp.linspace(0, 2*jnp.pi, self.args.N, endpoint=False)

        # positions on circle
        x = self.circle_radius * jnp.cos(angles)
        y = self.circle_radius * jnp.sin(angles)
        x_goal = jnp.stack([-x, -y], axis=1)
        
        # face toward origin (inside circle)
        v0 = jnp.zeros(self.args.N)

        if self.is_circular:
            v1 = jnp.zeros(self.args.N)
            state = jnp.stack([x, y, v0, v1], axis=1)
            return state, x_goal
        else:
            theta = jnp.arctan2(-y, -x)
            state = jnp.stack([x, y, theta, v0], axis=1)
            return state, x_goal

    def generate_states(self, seed = 42):
        self.states = self.setup_goal_swap()[0]
        return self.states

    def generate_goals(self, seed = 42):
        self.goals = self.setup_goal_swap()[1]
        return self.goals

    def check_results(self, traj):
        '''Check if all robots reached their goals without collisions. 
        Returns a dictionary with success status and number of collisions.
        success = 1 if all robots reached their goals without collisions, 0 otherwise.'''

        col_nums = count_inter_robot_collisions(traj[:, :, :2], self.all_radius)
        reached = jnp.all(jnp.linalg.norm(traj[:, -1, :2] - self.goals, axis=1) <= self.goal_tolerance)
        return {
                "success": int(reached and jnp.sum(col_nums) == 0),
                "goal_reached": reached,
                "num_collisions": int(jnp.sum(col_nums)),
                }