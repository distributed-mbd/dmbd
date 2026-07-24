# Distributed Model-Based Diffusion (DMBD)

## Overview

**Distributed Model-Based Diffusion (DMBD)** is a distributed robot-server framework for scalable multi-robot trajectory optimization based on [**Model-Based Diffusion (MBD)**](https://arxiv.org/abs/2407.01573).

Trajectory optimization for multi-robot systems is challenging due to the high-dimensional, non-convex, and often non-differentiable nature of multi-robot coordination problems. Existing sampling-based optimization methods typically operate over the joint trajectory space of all robots, causing the optimization dimension to grow with team size.

DMBD addresses this challenge by distributing the reverse diffusion process across individual robots. Instead of performing denoising over the full joint trajectory space, each robot independently performs **local conditional denoising** in its own control space while conditioning on the current trajectory estimates of other robots at each denoising step.

A central server aggregates and broadcasts the robots' current state trajectories at each denoising iteration.

For more information, please check our [paper](https://arxiv.org/pdf/2607.20992).

### Key Idea

For a team of  robots, centralized MBD operates in the joint trajectory space, whereas DMBD decomposes the inference problem into local trajectory spaces.

Thus, each robot performs trajectory optimization using its own:

- Dynamics
- Objective function
- Local constraints
- Control space

while conditioning on the current trajectory estimates of the other robots.

This decomposition improves scalability and sample efficiency while requiring only trajectory information to be exchanged between robots and the server.

---

### Why DMBD?

Compared with centralized sampling-based trajectory optimization, DMBD:

- Decomposes high-dimensional inference into per-robot inference problems.
- Improves scalability as the number of robots increases.
- Supports heterogeneous robots with different dynamics and objectives.
- Requires only trajectory information to be communicated.
- Is zeroth-order and does not require differentiable objectives or constraints.
- Retains the multi-modal trajectory optimization capabilities of MBD.
- Does not require offline training or learned score networks.

---

## Simulations

We evaluate DMBD on a diverse set of multi-robot trajectory optimization tasks:

- **Circular Goal Swap** (N=2-20)
- **Rectangular Goal Swap** (N=2-20)
- **Multi-Floor Coverage** (N=2-10)
- **Parking** (N=2)
- **Rush Hour** (N=4)

Experiments use:

- **Double-integrator dynamics**
- **Kinematic bicycle dynamics**
and support both circular and rectangular robot geometries.

### Representative Results
<table>
  <tr>
    <th></th>
    <th>Swap with 20 Circular Robots</th>
    <th>Swap with 20 Rectangular Robots</th>
    <th>Multi-Floor Coverage with 10 Rectangular Robots</th>
    <th>Narrow Parking</th>
    <th>Rush Hour</th>
  </tr>
  
  <tr>
    <th>Denoising</th>
    <td>
      <img
        width="220"
        alt="Scenario 1 Denoising"
        src="https://github.com/user-attachments/assets/4846d10a-e454-481c-b8aa-6447711c7aab"
      />
    </td>
    <td>
      <img
        width="220"
        alt="Scenario 2 Denoising"
        src="https://github.com/user-attachments/assets/983a06be-5452-448f-a1ed-6b02aefc9e7c"
      />
    </td>
    <td>
      <img
        width="220"
        alt="Scenario 3 Denoising"
        src="https://github.com/user-attachments/assets/9d744836-da32-405f-8dc2-f00f9359cedb" 
    </td>
    <td>
      <img
        width="220"
        alt="Scenario 4 Denoising"
        src="https://github.com/user-attachments/assets/f2de4bf6-6e92-4db4-9f61-b0d4aa771d5b" 
      />
    </td>
    <td>
      <img
        width="220"
        alt="Scenario 5 Denoising"
        src="https://github.com/user-attachments/assets/cad30e2f-8664-4ecc-8cdb-738cf653e363" 
      />
    </td>
  </tr>

  <tr>
    <th>Trajectory</th>
    <td>
      <img
        width="220"
        alt="Scenario 1 Trajectory"
        src="https://github.com/user-attachments/assets/adf138d1-239c-4ee9-811a-c7ee456b8aea" 
      />
    </td>
    <td>
      <img
        width="220"
        alt="Scenario 2 Trajectory"
        src="https://github.com/user-attachments/assets/19c676bd-8a84-4ed7-85c5-3fd373697358" 
      />
    </td>
    <td>
      <img
        width="220"
        alt="Scenario 3 Trajectory"
        src="https://github.com/user-attachments/assets/113ea36d-0be3-4bbf-a1cc-ec71a77a1d2f" 
      />
    </td>
    <td>
      <img
        width="220"
        alt="Scenario 3 Trajectory"
        src="https://github.com/user-attachments/assets/dfa7f8b7-140b-4b92-acea-d66ac0bb4a55" 
      />
    </td>
    <td>
      <img
        width="220"
        alt="Scenario 4 Trajectory"
        src="https://github.com/user-attachments/assets/b8a525b1-c4b0-481f-8b7b-ef812dfdd1ee" 
      />
    </td>
  </tr>
</table>


<h3>Additional Scenarios</h3>

<table>
  <tr>
    <th></th>
    <th>Narrow-Space Parking (N=6)</th>
    <th>Rush Hour (N=5)</th>
  </tr>

  <tr>
    <th>Denoising</th>
    <td>
      <img
        width="350"
        alt="Narrow-Space Parking Denoising"
        src="https://github.com/user-attachments/assets/fcbd52f4-2cc2-4b8b-8cd9-6c3c8b2c2213" 
      />
    </td>
    <td>
      <img
        width="350"
        alt="Rush Hour Denoising"
        src="https://github.com/user-attachments/assets/876471b8-32e5-4ed7-bef7-4f8444764931" 
      />
    </td>
  </tr>

  <tr>
    <th>Trajectory</th>
    <td>
      <img
        width="350"
        alt="Narrow-Space Parking Trajectory"
        src="https://github.com/user-attachments/assets/9c52c6ac-1bc7-4f31-b866-50606f8286c2" 
      />
    </td>
    <td>
      <img
        width="350"
        alt="Rush Hour Trajectory"
        src="https://github.com/user-attachments/assets/886525a6-7223-448f-b36f-524d55fc89c9" 
      />
    </td>
  </tr>
</table>


## Installation

The code is implemented in Python and uses **JAX** v0.9 for accelerated numerical computation.

Install the requirements
```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python main.py
