
from envs.helper.rect_collision import inter_robot_collision, count_inter_robot_collisions
from envs.circ_swap import EnvCircSwap
import jax.numpy as jnp
import jax

class EnvRectSwap (EnvCircSwap):
    def __init__(self, args, rollout_fn=None):
        super().__init__(args)
        self.args = args
        self.rollout_fn = rollout_fn
        self.lengths = jnp.full((args.N,), args.car_length)
        self.widths = jnp.full((args.N,), args.car_width)
        self.S = args.S
        self.goal_tolerance = args.goal_tolerance
        self.circle_radius = args.circle_radius
    
        self.is_circular = False
        self.states = None
        self.goals = None
        self.obs = None

        self.xbounds = [-35.0, 35.0]
        self.ybounds = [-35.0, 35.0]

    ### Distributed cost function for robot k ###
    def build_distributed_cost(self):
        """Build the distributed rectangular-swap objective for robot k."""
        @jax.jit
        def cost_fn(u_sampled, x0_k, team_trajs, robot_idx, goal_loc):
            """
            u_sampled: (T, dim_u) - robot k's control inputs
            x0_k: (state_dim,) - robot k's initial state        
            team_trajs: (N, T, state_dim) - trajectories of the whole team
            robot_idx: int - index of the robot k
            goal_loc: (state_dim,) - robot k's desired goal location
            """
            # Rollout the robot k's trajectory.
            my_traj = self.rollout_fn(x0_k[None, :], u_sampled[None, :, :])[0]

            # Penalize velocity.
            vel_penal = jnp.sum(jnp.square(my_traj[:, 3]))


            # Penalize collisions against the rest of the team by vmaping collision check over time T
            mask = jnp.arange(team_trajs.shape[0]) != robot_idx
            inter_col_penal = jnp.sum(jax.vmap(inter_robot_collision, in_axes=(0, 0, None, None, None, None))(
                team_trajs[:, :, :3].transpose(1, 0, 2),
                my_traj[:,:3],
                mask,
                self.widths,
                self.lengths,
                robot_idx
            ))

            # Penalize the final goal mismatch.
            dist_goal = my_traj[:, :2] - goal_loc
            terminal_cost = jnp.sum(self.S*jnp.linalg.norm(dist_goal[-25:], axis=1))

            return -(self.args.Q*vel_penal + terminal_cost + self.args.R*inter_col_penal +  self.args.P*jnp.linalg.norm(u_sampled))
        return cost_fn


    ### Centralized cost function for all robots ###
    def build_centralized_cost(self):
        """Build the centralized rectangular-swap objective for all robots."""
        @jax.jit
        def cost_fn(u_sampled, x0, goal_loc):
            """
            u_sampled: (N, T, dim_u) - control inputs for all robots
            x0: (N, state_dim) initial state of all robots
            goal_loc: (N, 2) goal locations for all robots
            """
            
            # Rollout trajectories for all robots, shape (N, T, state_dim)
            trajs = self.rollout_fn(x0, u_sampled)  

            # Penalize velocity
            vel_penal = jnp.sum(jnp.square(trajs[:, :, 3]))

            
            def per_timestep(pos_t):
                def robot_i(robot_idx):
                    return inter_robot_collision(
                        pos_t,          # (N, 3)
                        pos_t[robot_idx],
                        jnp.arange(pos_t.shape[0]) != robot_idx,
                        self.widths,
                        self.lengths,
                        robot_idx
                    )
                return jnp.sum(jax.vmap(robot_i)(jnp.arange(trajs.shape[0])))

            # Penalize inter-robot collisions
            inter_col_penal = jnp.sum(jax.vmap(per_timestep)(trajs[:, :, :3].transpose(1, 0, 2))) / self.args.N


            # Penalize terminal distance to goal
            dist_goal = trajs[:, -1, :2] - goal_loc # shape (N, 2)
            terminal_cost = jnp.sum(self.args.S*jnp.linalg.norm(dist_goal[:,-25:], axis=1))

            # Compute the norm of the control inputs for regularization
            control_norm = jnp.sum(jnp.linalg.norm(u_sampled, axis=(1, 2)))

            return -(self.args.Q*vel_penal + terminal_cost + self.args.R*inter_col_penal + self.args.P*control_norm)
        return cost_fn


    def check_results(self, traj):
        '''Check if all robots reached their goals without collisions. 
        Returns a dictionary with success status and number of collisions.
        success = 1 if all robots reached their goals without collisions, 0 otherwise.'''

        col_nums = count_inter_robot_collisions(traj, self.lengths, self.widths)
        reached = jnp.all(jnp.linalg.norm(traj[:, -1, :2] - self.goals, axis=-1) <= self.goal_tolerance)
        return {
                "success": int(reached and jnp.sum(col_nums) == 0),
                "goal_reached": reached,
                "num_collisions": int(jnp.sum(col_nums)),
                }
