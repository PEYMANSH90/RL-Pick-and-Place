# RL Pick-and-Place

**Reinforcement Learning-Based Minimum-Torque Control for Robotic Pick-and-Place Operations**

This repository implements the methodology from the paper:

> *Intelligent Control of Robots with Minimal Power Consumption in Pick-and-Place Operations*

The robot platform is an **ABB IRB1600-6/1.2** (6-axis manipulator). The control objective is to find the joint configuration that minimises total static torque, then train a **TD3** agent — implemented entirely from scratch in PyTorch — to track variable external loads at that joint using minimal motor torque.

---

## Repository Structure

```
RL-Pick-and-Place/
├── pick_place/                  # Main Python package
│   ├── kinematics/
│   │   ├── forward_kinematics.py   # DH transform chain, FK function, robot constants
│   │   ├── inverse_kinematics.py   # Multi-start L-BFGS-B IK solver (J2/J3/J5)
│   │   └── workspace.py            # Workspace sampling and 3-D visualisation
│   ├── dynamics/
│   │   └── inverse_statics.py      # RNE-based joint torque computation (gravity + payload)
│   ├── environments/
│   │   └── joint_control_env.py    # Custom Gymnasium environment for torque tracking
│   ├── agents/
│   │   ├── networks.py             # Actor and Critic MLPs (PyTorch, from scratch)
│   │   ├── replay_buffer.py        # Fixed-capacity circular replay buffer
│   │   └── td3.py                  # Full TD3 algorithm + TD3Config dataclass
│   ├── training/
│   │   ├── trainer.py              # Training loop with curriculum progression
│   │   └── curriculum.py           # CurriculumScheduler and level definitions
│   └── utils/
│       ├── logging.py              # get_logger() — stdout handler setup
│       ├── plotting.py             # Training-curve and load-tracking plots
│       └── random.py               # set_seed() — Python / NumPy / PyTorch
├── conf/
│   └── default.py                  # Typed dataclass config (EnvConfig, TD3Config, …)
├── scripts/
│   ├── train.py                    # CLI entry point for training
│   └── evaluate.py                 # CLI entry point for evaluation
├── tests/                          # Unit tests
├── ros2_ws/                        # ROS 2 workspace (Gazebo + MoveIt integration)
├── requirements.txt
└── setup.py
```

---

## Method Overview

### 1. Forward Kinematics
The IRB1600 is modelled with 8 DH frames (6 active joints + 2 fixed wrist frames).
The `forward_kinematics()` function accepts 6 joint angles and returns the 4×4
end-effector transformation matrix.

### 2. Inverse Kinematics
A multi-start L-BFGS-B solver searches over joints J2, J3, J5 (J1=J4=J6=0)
to minimise Cartesian XZ position error to a target pose, subject to a
quadratic joint-limit penalty.  Five valid configurations (JK1–JK5) are
identified matching Table 3 of the paper.

### 3. Inverse Statics (Torque Selection)
For each valid IK configuration, joint torques are computed via the
Recursive Newton-Euler (RNE) algorithm:

```
τ_total = τ_gravity + J(q)ᵀ · F_payload
```

The configuration with minimum total active torque |τ_J2 + τ_J3 + τ_J5|
is selected for RL training.

### 4. Reinforcement Learning — TD3
The selected joint is controlled by a TD3 agent.  The agent outputs a motor
torque command T* which drives the joint through a gear ratio N = 120:

```
T_joint = N · T*
```

**Environment**
| Field | Value |
|---|---|
| Observation | 6-D: [error, prev_error, integral, deriv, T*, load] — normalised |
| Action | T* ∈ [−2, +2] N·m (motor torque) |
| Reward | −(normalised_error² + 0.01 · T*²) |
| Timestep | 160 ms |
| Episode length | 100 steps |
| Load profile | 12-step variable profile from the paper |

**TD3 Hyperparameters** (paper values)
| Parameter | Value |
|---|---|
| Network | MLP [256, 256], ReLU |
| Learning rate | 3 × 10⁻⁴ |
| Replay buffer | 500 000 transitions |
| Batch size | 256 |
| Discount γ | 0.99 |
| Soft-update τ | 0.005 |
| Policy delay | 2 |
| Target noise σ | 0.2 (clipped ±0.5) |

**Curriculum learning**: 3 stages — 10% → 50% → 100% of the paper load
profile, 30 episodes each.  The agent carries its weights and replay buffer
across stages (transfer learning).

---

## Installation

```bash
git clone https://github.com/PEYMANSH90/RL-Pick-and-Place.git
cd RL-Pick-and-Place

# Install in editable mode (registers pick-place-train / pick-place-evaluate CLIs)
pip install -e .
```

Python ≥ 3.10 required.

---

## Usage

### Training

```bash
# Default config — paper hyperparameters, 200-episode budget
python scripts/train.py

# CLI overrides
python scripts/train.py --max-episodes 100 --device cuda --seed 42

# Quick smoke-test (10 episodes)
python scripts/train.py --max-episodes 10 --log-interval 1
```

Checkpoints are saved to `checkpoints/level_0/`, `checkpoints/level_1/`,
`checkpoints/level_2/`, and `checkpoints/final/`.
A training-curve plot is written to `results/`.

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/final
python scripts/evaluate.py --checkpoint checkpoints/final --episodes 5
```

Prints MSE, MAE, max tracking error and total reward per episode.
Saves a two-panel load-tracking plot to `results/`.

### Library API

```python
from pick_place.kinematics.forward_kinematics import forward_kinematics
from pick_place.dynamics.inverse_statics import InverseStatics, PAPER_CONFIGURATIONS
from pick_place.agents.td3 import TD3Agent, TD3Config
from pick_place.environments.joint_control_env import JointControlEnv

# Compute joint torques for all paper configurations
solver = InverseStatics(payload_kg=52.0)
results = solver.evaluate_configurations(PAPER_CONFIGURATIONS)
best_label, best = solver.minimum_torque_config(PAPER_CONFIGURATIONS)

# Train an agent
env   = JointControlEnv(load_scale=1.0)
agent = TD3Agent(obs_dim=6, action_dim=1, action_max=2.0)
```

---

## Key Design Principles

- **TD3 from scratch** — no stable-baselines3 or other RL library.
  `networks.py`, `replay_buffer.py`, and `td3.py` contain the complete
  implementation so every component is inspectable and modifiable.

- **Single responsibility** — the environment knows nothing about training;
  the agent knows nothing about curriculum; the trainer owns the loop.

- **Config-driven** — all hyperparameters live in `conf/default.py` as
  typed Python dataclasses.  CLI flags override individual fields without
  touching the config file.

- **No side effects on import** — every module is a library; no code runs
  at import time.

---

## Robot Specifications

| Property | Value |
|---|---|
| Model | ABB IRB1600-6/1.2 |
| Degrees of freedom | 6 |
| DH frames | 8 (6 active + 2 fixed wrist) |
| Link masses | [70, 60, 50, 30, 20, 10] kg |
| Gear ratios | [120, 120, 120, 100, 100, 100] |
| Max reach | 1.45 m |

---

## License

MIT
