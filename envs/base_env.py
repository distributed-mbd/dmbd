from envs.plotter.animate_2d import animate_trajectories, animate_trajectories_rect, animate_denoising_trajectories, animate_denoising_trajectories_rect
from envs.plotter.animate_3d import animate_trajectories_3d, animate_denoising_trajectories_3d

class BaseEnv:
    def __init__(self):
        
        self.lengths = None
        self.widths = None

        self.center = False
        self.is_circular = False
        self.is_3d = False
        self.states = None
        self.goals = None
        self.obs = None
        self.rollout_fn = None
        
        self.xbounds = [-25.0, 25.0]
        self.ybounds = [-25.0, 25.0]


    ### Distributed cost function for all robots ###
    def build_distributed_cost(self):
        def cost_fn(u_sampled, x0_i, team_trajs, robot_idx, goal_loc):
            """
            u_sampled: (T, dim_u) - robot k's control inputs
            x0_k: (state_dim,) - robot k's initial state        
            team_trajs: (N, T, state_dim) - trajectories of the whole team
            robot_idx: int - index of the robot k
            goal_loc: (state_dim,) - robot k's desired goal location
            """
            raise NotImplementedError("This method should be implemented in subclasses.")
        return cost_fn

    ### Centralized cost function for all robots ###
    def build_centralized_cost(self):
        """Build the centralized circular-swap objective for goal-swap scenario"""
        def cost_fn(u_sampled, x0, goal_loc):
            """
            u_sampled: (N, T, dim_u) - control inputs for all robots
            x0: (N, state_dim) initial state of all robots
            goal_loc: (N, 2) goal locations for all robots
            """
            raise NotImplementedError("This method should be implemented in subclasses.")
        return cost_fn

    def get_obs(self):
        """Get the current observation of the environment."""
        pass

    def generate_states(self, seed = 42):
        """Generate initial states for all robots."""
        pass

    def generate_goals(self, seed = 42):
        """Generate goal locations for all robots."""
        pass

    def visualize(self, traj, save=False):
        """Visualize the trajectories of all robots."""
        if self.is_circular:
            animate_trajectories(traj, self.args, obstacles=self.obs, goal_positions=self.goals, save=save, interval=30, x_bounds=self.xbounds, y_bounds=self.ybounds)
        elif self.is_3d:
            animate_trajectories_3d(traj, self.args, lengths = self.lengths, widths=self.widths, obstacles=self.obs, goal_positions=self.goals, N_elevator = self.args.N_elevator, elevator_region= self.elevator_regions, save=save, interval=30, unassigned_goals = False, center_point=self.center)
        else:
            animate_trajectories_rect(traj, self.args, lengths=self.lengths, obstacles=self.obs, goal_positions=self.goals, save=save, interval=30, x_bounds=self.xbounds, y_bounds=self.ybounds, center_point=self.center)
    
    def visualize_denoising(self, u, save = False):
        """Visualize the denoising process of the trajectories."""
        if self.is_circular:
            animate_denoising_trajectories(u, self.states, args = self.args, goal_positions=self.goals, rollout_fn=self.rollout_fn, save=save, interval=30, x_bounds=self.xbounds, y_bounds=self.ybounds)
        elif self.is_3d:
            animate_denoising_trajectories_3d(u, self.states, self.args, lengths = self.lengths, widths=self.widths, obstacles=self.obs, goal_positions=self.goals, save=save, interval=30, rollout_fn = self.rollout_fn, N_elevator=self.args.N_elevator, elevator_region= self.elevator_regions)
        else:
            animate_denoising_trajectories_rect(u, self.states ,args = self.args, lengths=self.lengths, rollout_fn=self.rollout_fn, obstacles=self.obs, goal_positions=self.goals, save=save, interval=30, x_bounds=self.xbounds, y_bounds=self.ybounds, center_point=self.center)

    def check_results(self, traj):
        '''Check if all robots reached their goals without collisions. '''
        pass