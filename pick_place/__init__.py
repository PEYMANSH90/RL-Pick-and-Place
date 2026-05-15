"""
pick_place — RL-based minimum-torque pick-and-place for the ABB IRB1600.

Modules
-------
kinematics   : forward kinematics, inverse kinematics, workspace analysis
dynamics     : inverse statics, joint-torque computation
environments : custom Gymnasium environments for torque-control tasks
agents       : TD3 agent (networks, replay buffer, algorithm) written from scratch
training     : training loop, curriculum learning
utils        : logging, plotting, random-seed helpers
"""

__version__ = "0.1.0"
