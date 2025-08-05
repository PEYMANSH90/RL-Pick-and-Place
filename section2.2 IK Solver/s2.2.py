import numpy as np
import sympy as sp
from scipy.optimize import minimize

# Symbolic variables (only J2, J3, J5 used)
theta_syms = sp.symbols('theta2 theta3 theta5')
pi = sp.pi

def dh_matrix(a, alpha, d, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Forward Kinematics (with J1, J4, J6 fixed = 0)
T1 = dh_matrix(0.15, -pi/2, 0.286, 0)
T2 = dh_matrix(0.475, 0, 0, theta_syms[0] - pi/2)
T3 = dh_matrix(0.5, 0, 0, theta_syms[1] + pi/2)
T4a = dh_matrix(0, pi/2, 0, pi/2)
T4 = dh_matrix(0, -pi/2, 0.1, 0)
T5 = dh_matrix(0.165, 0, 0, theta_syms[2] - pi/2)
T6a = dh_matrix(0, pi/2, 0, pi/2)
T6 = dh_matrix(0, 0, 0.2, 0)

T_final = sp.simplify(T1 * T2 * T3 * T4a * T4 * T5 * T6a * T6)
fk_func = sp.lambdify((theta_syms[0], theta_syms[1], theta_syms[2]), T_final[:3, 3], 'numpy')

# Target pose and joint limits
target_pose = np.array([0.5, 0.0, 0.5])
joint_limits = [(-1.1, 1.92), (-2.62, 2.62), (-2.0, 2.0)]
tolerance = 0.01  # Max allowable error in XZ

# Cost function with joint limit penalty
def cost(j):
    try:
        pos = np.array(fk_func(*j)).astype(np.float64).flatten()
        position_error = np.linalg.norm(pos[[0, 2]] - target_pose[[0, 2]])
        
        # Add penalty for joint limit violations
        limit_penalty = 0
        for i, (angle, (low, high)) in enumerate(zip(j, joint_limits)):
            if angle < low:
                limit_penalty += 1000 * (low - angle)**2  # Quadratic penalty
            elif angle > high:
                limit_penalty += 1000 * (angle - high)**2  # Quadratic penalty
        
        return position_error + limit_penalty
    except:
        return 1e6

# Optimization loop with bounded optimization
num_guesses = 500
max_steps = 20
accepted_configs = []

for i in range(num_guesses):
    # Generate random initial guess within joint limits
    guess = np.array([np.random.uniform(low, high) for (low, high) in joint_limits])
    print(f" IK Trial #{i+1} | Initial guess (J2, J3, J5): {np.round(guess, 3)}")

    def step_callback(xk):
        pos = np.array(fk_func(*xk)).flatten()
        err = np.linalg.norm(pos[[0, 2]] - target_pose[[0, 2]])
        print(f"   ➤ Step | EE XZ: {np.round(pos[[0,2]], 3)} | Error: {err:.6f}")
        print(f"   ➤ Joints: {np.round(xk, 4)}")


    # Use L-BFGS-B method for bounded optimization
    result = minimize(cost, guess, method='L-BFGS-B', 
                      bounds=joint_limits,  # Enforce joint limits
                      callback=step_callback,
                      options={'disp': False, 'maxiter': max_steps})

    final_pos = np.array(fk_func(*result.x)).flatten()
    final_err = np.linalg.norm(final_pos[[0, 2]] - target_pose[[0, 2]])
    print(f"Final EE XZ: {np.round(final_pos[[0,2]], 3)} | Final error: {final_err:.6f}")

    # Check if solution is within joint limits
    within_limits = True
    for j, (angle, (low, high)) in enumerate(zip(result.x, joint_limits)):
        if angle < low or angle > high:
            within_limits = False
            print(f"    Joint {j+2} ({angle:.4f}) outside limits [{low}, {high}]")
            break
    
    if final_err < tolerance and within_limits:
        accepted_configs.append(result.x)
        print(f"   Configuration accepted (within limits)")
    else:
        print(f"   Configuration rejected")

# Summary of accepted configurations
print(" Valid Joint Configurations (within tolerance AND joint limits")
if accepted_configs:
    for idx, config in enumerate(accepted_configs):
        print(f"   Config {idx+1}: J2={config[0]:.4f}, J3={config[1]:.4f}, J5={config[2]:.4f}")
    # Print as Python list for easy copy-paste
    print("Python list for is solver")
    print("[")
    for config in accepted_configs:
        print(f"    [{config[0]:.4f}, {config[1]:.4f}, {config[2]:.4f}],")
    print("]")
else:
    print("  No configurations met both error tolerance and joint limits.")
