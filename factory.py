# ============================================================
# Environment imports
# ============================================================

from envs.circ_swap import EnvCircSwap
from envs.rect_swap import EnvRectSwap
from envs.multi_floor import EnvMultiFloor
from envs.parking import EnvParking
from envs.rush_hour import EnvRushHour


# ============================================================
# Robot dynamics / rollout imports
# ============================================================

from robots.kinematic_bicycle import one_D_kinematic_bicycle_rollout, kinematic_bicycle_rollout, rollout_heterogenous_team
from robots.double_integrator import double_integrator_rollout


# ============================================================
# Planner imports
# ============================================================

from planner.mbd import central_mbd
from planner.dmbd import distributed_mbd
from planner.d4orm import d4orm
from planner.cem import cem
from planner.mppi import mppi


import time
import yaml


# ============================================================
# Configuration utilities
# ============================================================

class Args:
    """Simple configuration container that exposes YAML entries as attributes."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_args_from_yaml(scenario: str, N=10):
    """Load scenario-specific configuration parameters from a YAML file."""

    config_file = f"configs/{scenario}{N}.yaml"

    # Parking and rush-hour use a single configuration
    # independent of the number of robots.
    if scenario in ["parking", "rush_hour"]:
        config_file = f"configs/{scenario}.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return Args(**config)


# ============================================================
# Robot dynamics / rollout selection
# ============================================================

def load_rollout_fn(scenario: str, args: Args):
    """Return the appropriate robot dynamics rollout function for a scenario."""

    if scenario == "circ_swap":
        return lambda state, u: double_integrator_rollout(state, u, vmax=args.vmax, amax=args.umax, dt=args.dt)

    elif scenario == "rect_swap":
        return lambda state, u: kinematic_bicycle_rollout(state, u, vmax=args.vmax, amax=args.umax, steer_max=args.steer_max, dt=args.dt)

    elif scenario == "multi_floor":
        return lambda state, u: rollout_heterogenous_team(args, state, u, vmax=args.vmax, amax=args.umax, steer_max=args.steer_max, dt=args.dt)

    elif scenario == "parking":
        return lambda state, u: kinematic_bicycle_rollout(state, u, vmax=args.vmax, amax=args.umax, steer_max=args.steer_max, dt=args.dt)

    elif scenario == "rush_hour":
        return lambda state, u: one_D_kinematic_bicycle_rollout(state, u, vmax=args.vmax, amax=args.umax, dt=args.dt)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ============================================================
# Planner selection
# ============================================================

def load_planner(planner_name: str):
    """Return the planner implementation corresponding to the planner name."""

    if planner_name == "dmbd":
        return distributed_mbd

    elif planner_name == "mbd":
        return central_mbd

    elif planner_name == "d4orm":
        return d4orm

    elif planner_name == "cem":
        return cem

    elif planner_name == "mppi":
        return mppi

    else:
        raise ValueError(f"Unknown planner: {planner_name}")


# ============================================================
# Environment selection
# ============================================================

def load_env(scenario: str, args: Args, rollout_fn):
    """Instantiate the environment corresponding to the selected scenario."""

    if scenario == "circ_swap":
        return EnvCircSwap(args, rollout_fn=rollout_fn)

    elif scenario == "rect_swap":
        return EnvRectSwap(args, rollout_fn=rollout_fn)

    elif scenario == "multi_floor":
        return EnvMultiFloor(args, rollout_fn=rollout_fn)

    elif scenario == "parking":
        return EnvParking(args, rollout_fn=rollout_fn)

    elif scenario == "rush_hour":
        return EnvRushHour(args, rollout_fn=rollout_fn)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ============================================================
# Scenario / planner compatibility validation
# ============================================================

def validate_scenario_planner_combination(scenario: str, planner_name: str):
    """Validate that the selected planner is supported by the scenario."""

    if scenario == "circ_swap" and planner_name not in ["dmbd", "mbd", "d4orm", "cem", "mppi"]:
        raise ValueError(f"Invalid planner {planner_name} for scenario {scenario}")

    elif scenario == "rect_swap" and planner_name not in ["dmbd", "mbd", "d4orm", "cem", "mppi"]:
        raise ValueError(f"Invalid planner {planner_name} for scenario {scenario}")

    elif scenario == "multi_floor" and planner_name not in ["dmbd", "mbd", "d4orm", "cem", "mppi"]:
        raise ValueError(f"Invalid planner {planner_name} for scenario {scenario}")

    elif scenario == "parking" and planner_name not in ["dmbd"]:
        raise ValueError(f"Invalid planner {planner_name} for scenario {scenario}")

    elif scenario == "rush_hour" and planner_name not in ["dmbd"]:
        raise ValueError(f"Invalid planner {planner_name} for scenario {scenario}")


# ============================================================
# Simulation factory
# ============================================================

class Simulation:
    """Factory class that initializes and runs a scenario-planner combination."""

    def __init__(self, scenario: str, planner: str, N: int):

        # Validate the requested scenario and planner combination.
        validate_scenario_planner_combination(scenario, planner)

        self.scenario = scenario
        self.planner_name = planner

        # Load scenario-specific configuration.
        self.args = load_args_from_yaml(scenario, N)

        # Select robot dynamics and instantiate the environment.
        self.rollout_fn = load_rollout_fn(scenario, self.args)
        self.env = load_env(scenario, self.args, self.rollout_fn)

        # Select the requested trajectory optimization planner.
        self.planner = load_planner(planner)

        # Build the appropriate cost function.
        # DMBD uses a distributed cost; all other planners use
        # the centralized cost formulation.
        self.cost_fn = self.env.build_distributed_cost() if planner == "dmbd" else self.env.build_centralized_cost()


    def run_simulation(self, state_goal_seed: int = 42, planner_seed: int = 42):
        """
        Run a single simulation trial.

        Args:
            state_goal_seed: Random seed used to generate initial states and goals.
            planner_seed: Random seed used by the trajectory optimization planner.

        Returns:
            success: Whether the resulting trajectory satisfies the scenario requirements.
            time_elapsed: Planner execution time in seconds.
            traj: Planned trajectory generated from the optimized controls.
        """

        # Initialize environment observations.
        self.env.get_obs()

        # Generate initial robot states and goal states.
        state = self.env.generate_states(seed=state_goal_seed)
        x_goal = self.env.generate_goals(seed=state_goal_seed)

        # --------------------------------------------------------
        # Run trajectory optimization
        # --------------------------------------------------------

        time_init = time.time()
        u = None

        # DMBD requires the additional rollout function argument.
        if self.planner_name == "dmbd":
            u = self.planner(self.args, self.cost_fn, self.rollout_fn, state, x_goal, planner_seed)

        # All other planners use the centralized cost function.
        else:
            u = self.planner(self.args, self.cost_fn, state, x_goal, planner_seed)

        # Measure planner execution time.
        time_elapsed = time.time() - time_init

        print(
            f"Scenario: {self.scenario}, "
            f"Number of Robots: {self.args.N}, "
            f"Planner: {self.planner_name}, "
            f"State/Goal Seed: {state_goal_seed}, "
            f"Planner Seed: {planner_seed}, "
            f"Time Elapsed: {time_elapsed:.2f} seconds"
        )

        # Roll out the final optimized control sequence.
        traj = self.rollout_fn(state, u[-1])

        # Evaluate whether the resulting trajectory is successful.
        success = self.env.check_results(traj)["success"]

        return success, time_elapsed, traj, u, state, x_goal