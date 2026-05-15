# RL Pick-and-Place

**Reinforcement Learning-Based Minimum-Torque Control for Robotic Pick-and-Place Operations**

This repository implements the methodology from the paper:

> *Intelligent Control of Robots with Minimal Power Consumption in Pick-and-Place Operations*

The robot platform is an **ABB IRB1600-6/1.2** (6-axis manipulator). The control objective is to find the joint configuration that minimises total static torque, then train a **TD3** agent — implemented entirely from scratch in PyTorch — to track variable external loads at that joint using minimal motor torque.

---

## Repository Structure

```
RL-Pick-and-Place/
├── pick_place/
│   ├── kinematics/
│   │   ├── forward_kinematics.py
│   │   ├── inverse_kinematics.py
│   │   └── workspace.py
│   ├── dynamics/
│   │   └── inverse_statics.py
│   ├── environments/
│   │   └── joint_control_env.py
│   ├── agents/
│   │   ├── networks.py
│   │   ├── replay_buffer.py
│   │   └── td3.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── curriculum.py
│   └── utils/
│       ├── logging.py
│       ├── plotting.py
│       └── random.py
├── conf/
│   └── default.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── tests/
├── ros2_ws/
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
pip install -e .
```

Python ≥ 3.10 required.

---

## Usage

### Training

```bash
python scripts/train.py
python scripts/train.py --max-episodes 100 --device cuda --seed 42
python scripts/train.py --max-episodes 10 --log-interval 1
```

Checkpoints are saved under `checkpoints/`; a training-curve plot is written to `results/`.

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/final
python scripts/evaluate.py --checkpoint checkpoints/final --episodes 5
```

### Library API

```python
from pick_place.kinematics.forward_kinematics import forward_kinematics
from pick_place.dynamics.inverse_statics import InverseStatics, PAPER_CONFIGURATIONS
from pick_place.agents.td3 import TD3Agent, TD3Config
from pick_place.environments.joint_control_env import JointControlEnv

solver = InverseStatics(payload_kg=52.0)
best_label, best = solver.minimum_torque_config(PAPER_CONFIGURATIONS)

env   = JointControlEnv(load_scale=1.0)
agent = TD3Agent(obs_dim=6, action_dim=1, action_max=2.0)
```

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
