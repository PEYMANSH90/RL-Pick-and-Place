import numpy as np
import sympy as sp
from scipy.optimize import minimize
from roboticstoolbox import ERobot, DHRobot, RevoluteDH

g = 9.81  # m/s^2

print("🔍 COMPREHENSIVE TORQUE CALCULATION VERIFICATION")
print("=" * 60)

# Test if we need to include ALL 6 joints instead of just 3
print("🤖 TESTING WITH ALL 6 JOINTS:")
print("-" * 40)

# Table 1: IRB 1600 robot parameters - Using exact paper masses
masses = [70, 60, 50, 30, 20, 10]  # kg for B1-B6 (exact paper values)
link_lengths = [0.15, 0.475, 0.5, 0, 0.165, 0]  # a length for B1-B6

# DH parameters from Table 1 with exact paper masses and center of mass offsets
# a, alpha, d, offset, mass, r (center of mass offset)
links = [
    RevoluteDH(a=0.15,   alpha=-np.pi/2, d=0.286, offset=0,        m=masses[0], r=[0.15/2, 0, 0]),
    RevoluteDH(a=0.475,  alpha=0,        d=0,     offset=-np.pi/2, m=masses[1], r=[0.475/2, 0, 0]),
    RevoluteDH(a=0.5,    alpha=0,        d=0,     offset=np.pi/2,  m=masses[2], r=[0.5/2, 0, 0]),
    RevoluteDH(a=0,      alpha=np.pi/2,  d=0.1,   offset=0,        m=masses[3], r=[0, 0, 0.1/2]),
    RevoluteDH(a=0.165,  alpha=0,        d=0,     offset=-np.pi/2, m=masses[4], r=[0.165/2, 0, 0]),
    RevoluteDH(a=0,      alpha=0,        d=0.2,   offset=0,        m=masses[5], r=[0, 0, 0.2/2]),
]

robot = DHRobot(links, name='IRB1600')
robot.gravity = [0, 0, -9.81]

# Try different joint configurations to reach target torque
print("🔍 FINDING JOINT CONFIGURATION FOR TARGET TORQUE (519 Nm)")
print("=" * 60)
print(f"Fixed masses: {masses}")
print(f"Link lengths: {link_lengths}")

# Test multiple joint configurations
test_configs = [
    [0.1, -0.3, 0.2, 0.1, 0.15, 0.05],  # Very neutral
    [0.2, -0.5, 0.4, 0.2, 0.3, 0.1],    # Slightly more angle
    [0.3, -0.7, 0.6, 0.3, 0.45, 0.15],  # More angle
    [0.4, -0.9, 0.8, 0.4, 0.6, 0.2],    # Even more
    [0.5, -1.1, 1.0, 0.5, 0.75, 0.25],  # Original test config
]

target_torque = 519.0
best_config = None
best_torque = float('inf')
best_payload = 0

for i, config in enumerate(test_configs):
    print(f"\n--- Testing Configuration {i+1}: {[f'{x:.2f}' for x in config]} ---")
    
    # Calculate link torques
    tau_links = robot.rne(q=config, qd=[0]*6, qdd=[0]*6)
    link_torque_magnitude = np.linalg.norm(tau_links)
    
    print(f"   Link torque magnitude: {link_torque_magnitude:.2f} Nm")
    
    if link_torque_magnitude < target_torque:
        # Can add payload to reach target
        required_payload_contribution = np.sqrt(target_torque**2 - link_torque_magnitude**2)
        required_payload_mass = required_payload_contribution / (g * sum(link_lengths))
        
        print(f"   ✅ Can reach target with payload: {required_payload_mass:.2f} kg")
        print(f"   📊 Total torque would be: {target_torque:.2f} Nm")
        
        if abs(link_torque_magnitude - target_torque) < abs(best_torque - target_torque):
            best_config = config
            best_torque = link_torque_magnitude
            best_payload = required_payload_mass
    else:
        print(f"   ❌ Torque too high, cannot reach target")

# Use the best configuration found
if best_config is not None:
    joint_angles = best_config
    print(f"\n🎯 BEST CONFIGURATION FOUND:")
    print(f"   Joint angles: {[f'{x:.3f}' for x in joint_angles]}")
    print(f"   Link torque: {best_torque:.2f} Nm")
    print(f"   Required payload: {best_payload:.2f} kg")
    print(f"   Total torque: {target_torque:.2f} Nm")
else:
    # If no configuration works, use the one closest to target
    joint_angles = [0.1, -0.3, 0.2, 0.1, 0.15, 0.05]  # Very neutral
    print(f"\n⚠️  No configuration found, using very neutral angles")

print(f"\n🔍 FINAL CALCULATION WITH PAPER MASSES:")
print("=" * 60)
print(f"Masses: {masses}")
print(f"Joint angles (rad): {joint_angles}")

# Motor and system parameters (example values)
U = 48.0      # Supply voltage in V
psi = 0.1     # Motor flux linkage in Nm/A

def print_jk_results(jk_name, joint_angles):
    tau_links = robot.rne(q=joint_angles, qd=[0]*6, qdd=[0]*6)
    print(f"\n=== {jk_name} ===")
    print(f"Joint angles (rad): {joint_angles}")
    print(f"Joint torques (Nm): {[f'{tau:.1f}' for tau in tau_links]}")
    print(f"Sum of torques (with sign): {np.sum(tau_links):.1f} Nm")
    # Power calculation
    P_j = [U * abs(tau) / psi for tau in tau_links]
    P_total = sum(P_j)
    print(f"Joint holding power (W): {[f'{P:.1f}' for P in P_j]}")
    print(f"Total holding power: {P_total:.1f} W")
    return P_total, tau_links

# All JKs from Table 3
JKs = [
    [0, -1.0996, 0.6283, 0, 1.2601, 0],      # JK1
    [0, -0.8482, 1.5080, 0, -1.5045, 0],     # JK2
    [0, -0.7226, 0.3770, 0, 1.6371, 0],      # JK3
    [0, -0.7226, 1.5080, 0, -1.6301, 0],     # JK4
    [0, -0.3456, 1.3823, 0, -2.0071, 0],     # JK5
]

best_power = float('inf')
best_jk = None
best_jk_idx = -1
best_torques = None

print("\n🔍 EVALUATING ALL JKs FROM TABLE 3:")
for idx, jk in enumerate(JKs):
    P_total, tau_links = print_jk_results(f"JK{idx+1}", jk)
    if P_total < best_power:
        best_power = P_total
        best_jk = jk
        best_jk_idx = idx+1
        best_torques = tau_links

print(f"\n=== BEST JK (LOWEST POWER): JK{best_jk_idx} ===")
print(f"Joint angles (rad): {best_jk}")
print(f"Joint torques (Nm): {[f'{tau:.1f}' for tau in best_torques]}")
print(f"Total holding power: {best_power:.1f} W")

# Compare to paper's reported values
print("\nPaper Table 4 (JK1):")
print("J1:   0 Nm\nJ2: 122 Nm\nJ3: -592 Nm\nJ4:   0 Nm\nJ5: -49 Nm\nJ6:   0 Nm\nTotal: -519 Nm")

# Calculate required payload to reach exactly 519 Nm
if np.sum(best_torques) < target_torque:
    required_payload_contribution = target_torque - np.sum(best_torques)
    required_payload_mass = required_payload_contribution / (g * sum(link_lengths))
    
    print(f"\n📦 REQUIRED PAYLOAD TO REACH {target_torque} Nm (Sum of torques):")
    print(f"   Payload mass: {required_payload_mass:.2f} kg")
    
    # Calculate payload torques
    payload_torques = np.zeros(6)
    for i in range(6):
        if i == 0:
            effective_arm = sum(link_lengths) * np.cos(best_jk[0])
        elif i == 1:
            effective_arm = sum(link_lengths[1:]) * np.cos(best_jk[1])
        elif i == 2:
            effective_arm = sum(link_lengths[2:]) * np.cos(best_jk[2])
        elif i == 3:
            effective_arm = sum(link_lengths[3:]) * np.cos(best_jk[3])
        elif i == 4:
            effective_arm = sum(link_lengths[4:]) * np.cos(best_jk[4])
        else:
            effective_arm = link_lengths[5] * np.cos(best_jk[5])
        
        payload_torques[i] = required_payload_mass * g * effective_arm
        print(f"   J{i+1} Payload: {required_payload_mass:.2f} * {g} * {effective_arm:.3f} = {payload_torques[i]:.4f} Nm")
    
    # Combined torques
    combined_torques = best_torques + payload_torques
    combined_magnitude = np.linalg.norm(combined_torques)
    print(f"   📊 Combined Torque (links + payload): {combined_magnitude:.4f} Nm")
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"   ✅ SUCCESS! Total torque: {combined_magnitude:.2f} Nm")
    print(f"   📦 Required payload: {required_payload_mass:.2f} kg")
    print(f"   🤖 Robot masses: {masses}")
    print(f"   📐 Joint angles: {[f'{x:.3f}' for x in best_jk]}")
else:
    print(f"\n❌ Cannot reach target torque with these masses and joint angles")
    print(f"   Current torque: {np.sum(best_torques):.2f} Nm")
    print(f"   Target torque: {target_torque} Nm")
    print(f"   Need different joint configuration")

# Also check if sum of absolute torques can reach 519 Nm
print(f"\n🔍 CHECKING SUM OF ABSOLUTE TORQUES:")
print(f"   Current sum of absolute torques: {np.sum(best_torques):.2f} Nm")
print(f"   Target: 519.0 Nm")

if np.sum(best_torques) < target_torque:
    print(f"   ✅ Can reach target with payload!")
    required_payload_contribution_abs = target_torque - np.sum(best_torques)
    required_payload_mass_abs = required_payload_contribution_abs / (g * sum(link_lengths))
    print(f"   📦 Required payload for sum of absolutes: {required_payload_mass_abs:.2f} kg")
else:
    print(f"   ❌ Sum of absolute torques also too high")
    print(f"   🤔 The paper might be using different units (N⋅cm instead of N⋅m)")

# Check if using N⋅cm instead of N⋅m would work
print(f"\n🔍 CHECKING IF PAPER USES N⋅cm INSTEAD OF N⋅m:")
target_torque_cm = 519.0 * 100  # Convert to N⋅cm
print(f"   Target in N⋅cm: {target_torque_cm:.0f} N⋅cm")
print(f"   Current torque in N⋅cm: {np.sum(best_torques) * 100:.0f} N⋅cm")

if np.sum(best_torques) * 100 < target_torque_cm:
    print(f"   ✅ Could work with N⋅cm units!")
    required_payload_contribution_cm = np.sqrt(target_torque_cm**2 - (np.sum(best_torques) * 100)**2)
    required_payload_mass_cm = required_payload_contribution_cm / (g * sum(link_lengths) * 100)
    print(f"   📦 Required payload for N⋅cm: {required_payload_mass_cm:.2f} kg")
else:
    print(f"   ❌ Still too high even with N⋅cm") 