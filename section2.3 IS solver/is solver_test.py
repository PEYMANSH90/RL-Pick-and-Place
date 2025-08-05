import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from scipy.optimize import minimize

# Constants
g = 9.81  # gravity (m/s^2)

# 5 valid configurations (for joints J2, J3, J5)
valid_configurations = [
    [-1.0996, 0.6283, 1.2601],   # JK1
    [-0.8482, 1.5080, -1.5045],  # JK2
    [-0.7226, 0.3770, 1.6371],   # JK3
    [-0.7226, 1.5080, -1.6301],  # JK4
    [-0.3456, 1.3823, -2.0071]   # JK5
]

# Target torques from paper Table 4 (J2, J3, J5 for each JK)
target_torques = np.array([
    [122,  -592, -49],   # JK1
    [73,   -528, -46],   # JK2
    [-25,  -584, -23],   # JK3
    [64,   -495, -43],   # JK4
    [-78,  -349, -39]    # JK5
])

# Masses and link lengths (in meters)
masses = [70, 60, 50, 30, 20, 10]  # kg
link_lengths = [0.15, 0.475, 0.5, 0, 0.165, 0]  # for reference

# Define robot using 8-link structure based on paper's Table 2
links = [
    RevoluteDH(a=0.15,  alpha=-np.pi/2, d=0.286, offset=0,         m=masses[0], r=[0.15/2, 0, 0]),        # Joint 1
    RevoluteDH(a=0.475, alpha=0,        d=0,     offset=-np.pi/2,  m=masses[1], r=[0.475/2, 0, 0]),       # Joint 2 (θ2 - π/2)
    RevoluteDH(a=0.5,   alpha=0,        d=0,     offset=np.pi/2,   m=masses[2], r=[0.5/2, 0, 0]),         # Joint 3 (θ3 + π/2)
    RevoluteDH(a=0,     alpha=np.pi/2,  d=0,     offset=np.pi/2,   m=masses[3], r=[0, 0, 0]),             # B4a (fixed π/2)
    RevoluteDH(a=0,     alpha=-np.pi/2, d=0.1,   offset=0,         m=masses[3], r=[0, 0, 0.1/2]),         # Joint 4
    RevoluteDH(a=0.165, alpha=0,        d=0,     offset=-np.pi/2,  m=masses[4], r=[0.165/2, 0, 0]),       # Joint 5 (θ5 - π/2)
    RevoluteDH(a=0,     alpha=np.pi/2,  d=0,     offset=np.pi/2,   m=masses[5], r=[0, 0, 0]),             # B6a (fixed π/2)
    RevoluteDH(a=0,     alpha=0,        d=0.2,   offset=0,         m=masses[5], r=[0, 0, 0.2/2])          # Joint 6
]

# Create robot model
robot = DHRobot(links, name='IRB1600')
robot.gravity = [0, 0, -g]

def calc_torques(force):
    """Calculate torques for all configurations given an end-effector force."""
    results = []
    for config in valid_configurations:
        full_config = [0, config[0], config[1], np.pi/2, 0, config[2], np.pi/2, 0]
        tau_gravity = robot.rne(q=full_config, qd=[0]*8, qdd=[0]*8)
        J = robot.jacob0(full_config)[0:3, :]
        tau_ee = J.T @ force
        tau_total = tau_gravity + tau_ee
        results.append([tau_total[1], tau_total[2], tau_total[5]])
    return np.array(results)

def error_fz(fz):
    fz = fz[0]  # minimize passes an array
    force = np.array([0, 0, fz])
    torques = calc_torques(force)
    return np.sum((torques - target_torques)**2)

def error_fxyz(force):
    torques = calc_torques(force)
    return np.sum((torques - target_torques)**2)

# --- Optimize Fz only (vertical mass) ---
res_fz = minimize(error_fz, -1000, method='Powell')
best_fz = res_fz.x[0]
best_force_fz = np.array([0, 0, best_fz])
print("\nBest vertical load (Fz only): Fz = {:.2f} N (mass = {:.2f} kg)".format(best_fz, -best_fz/g))
torques_fz = calc_torques(best_force_fz)
print(f"{'Config':<7} {'J2 (Nm)':>10} {'J3 (Nm)':>10} {'J5 (Nm)':>10} {'Sum (Nm)':>12}")
for i, (j2, j3, j5) in enumerate(torques_fz, 1):
    total = j2 + j3 + j5
    print(f"{i:<7} {j2:>10.2f} {j3:>10.2f} {j5:>10.2f} {total:>12.2f}")

# --- Optimize Fx, Fy, Fz (full 3D force) ---
res_fxyz = minimize(error_fxyz, [0, 0, -1000], method='Powell')
best_force_fxyz = res_fxyz.x
print("\nBest 3D load: Fx = {:.2f} N, Fy = {:.2f} N, Fz = {:.2f} N (mass = {:.2f} kg)".format(
    best_force_fxyz[0], best_force_fxyz[1], best_force_fxyz[2], -best_force_fxyz[2]/g))
torques_fxyz = calc_torques(best_force_fxyz)
print(f"{'Config':<7} {'J2 (Nm)':>10} {'J3 (Nm)':>10} {'J5 (Nm)':>10} {'Sum (Nm)':>12}")
for i, (j2, j3, j5) in enumerate(torques_fxyz, 1):
    total = j2 + j3 + j5
    print(f"{i:<7} {j2:>10.2f} {j3:>10.2f} {j5:>10.2f} {total:>12.2f}")
