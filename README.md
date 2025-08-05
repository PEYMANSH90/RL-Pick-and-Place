# RL-Pick_and_Place 🤖

**Intelligent Control of Robots with Minimal Power Consumption in Pick-and-Place Operations**

This repository implements reinforcement learning (RL) based control strategies for robotic pick-and-place operations, focusing on minimizing power consumption while maintaining high performance.

## 📁 Repository Structure

```
RL-Pick-and-Place/
├── section2.1 DH convention/     # DH Convention Implementation
├── section2.2 IK Solver/         # IK Solver
├── section2.3 IS solver/         # IS Solver  
├── section2.4 RL picking/        # RL-based Torque Control
├── results/                      # Training results and plots
└── README.md                     # This file
```

## 🎯 Project Overview

This project implements the research paper: **"Intelligent Control of Robots with Minimal Power Consumption in Pick-and-Place Operations"**

### Key Features:
- **Section 2.1**: DH (Denavit-Hartenberg) Convention implementation
- **Section 2.2**: Inverse Kinematics (IK) Solver
- **Section 2.3**: Intelligent Solver (IS) implementation
- **Section 2.4**: RL-based torque control using TD3 agent

## 🚀 Quick Start

### Prerequisites
```bash
# Python environment with required packages
pip install numpy matplotlib stable-baselines3 gymnasium
```

### Running the RL Control (Section 2.4)
```bash
cd "section2.4 RL picking"/
python "in picking.py"
```

## 📊 Results

The RL training automatically saves:
- **Trained models**: `TD3_model_*.pkl`
- **Performance plots**: `RL_control_plot.png`, `training_progress.png`
- **Metrics**: `performance_data.json`, `training_summary.json`

## 🔧 Technical Details

### RL Agent Configuration
- **Algorithm**: TD3 (Twin Delayed Deep Deterministic Policy Gradient)
- **Network**: [256, 256] MLP
- **Training**: 100,000 timesteps (~20 minutes on CPU)
- **Features**: Action smoothing, enhanced reward function, curriculum learning

### Environment Features
- **Observation Space**: 6D [error, integral_error, derivative_error, current_torque, load_torque, prev_T_star]
- **Action Space**: Continuous torque control (±50 range)
- **Reward Function**: Multi-objective (tracking error, action penalty, smoothness)

## 📈 Performance Metrics

The system tracks:
- **RMSE**: Root Mean Square Error
- **Max Error**: Maximum absolute error
- **Mean Error**: Average absolute error
- **Saturation**: Action saturation percentage
- **Power Consumption**: Motor power analysis

## 🎓 Research Context

This implementation follows the methodology described in the research paper, providing:
- Realistic motor dynamics simulation
- Power consumption optimization
- Smooth control policies
- Comprehensive performance analysis

## 🤝 Contributing

Feel free to contribute by:
- Improving the RL algorithms
- Adding new control strategies
- Enhancing the visualization
- Optimizing performance

## 📄 License

This project is for research and educational purposes.

## 📞 Contact

For questions or collaboration, please open an issue or contact the maintainer.

---

**Note**: This repository contains experimental implementations of RL-based robotic control. Results may vary depending on hardware and training conditions. 