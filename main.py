'''Main script to run the simulation with specified scenario and planner.'''

''' Scenario Options 
circ_swap, rect_swap, multi_floor, parking, rush_hour

planner Options
dmbd, mbd, d4orm, cem, mppi'''

from factory import Simulation

scenario = "rect_swap"  # Change this to the desired scenario
planner = "dmbd"  # Change this to the desired planner

N = 10  # Change this to the desired number of robots (only applicable for circ_swap, rect_swap, multi_floor)
seed = 10


sim = Simulation(scenario, planner, N)

# Warm-up / JIT compilation
print("Warming up JIT...")
sim.run_simulation(state_goal_seed=0)

print("Running simulations...")
success, time_elapsed, traj, u, state, x_goal = sim.run_simulation(planner_seed=seed)
print(f"Success: {bool(success)}")
sim.env.visualize_denoising(u, save = False)
sim.env.visualize(traj, save = False)
