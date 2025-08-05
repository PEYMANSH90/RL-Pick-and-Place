import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. DH Parameters from the paper (Table 2)
dh_params = np.array([
    [0.15,     -np.pi/2, 0.286, 0],
    [0.475,     0,       0,     -np.pi/2],
    [0.5,       0,       0,      np.pi/2],
    [0,         np.pi/2, 0,      np.pi/2],
    [0,        -np.pi/2, 0.1,    0],
    [0.165,     0,       0,     -np.pi/2],
    [0,         np.pi/2, 0,      np.pi/2],
    [0,         0,       0.2,    0]
])

# 2. Joint limits (from Table 1)
joint_limits = [
    (-2.62, 2.62),
    (-1.1, 1.92),
    (-2.62, 2.62),
    (-3.14, 3.14),
    (-2, 2),
    (-3.14, 3.14),
    (0, 0),
    (0, 0)
]

# 3. Discretize joint angles
N = 50  # number of steps per joint
joint_grids = [np.linspace(lim[0], lim[1], N) for lim in joint_limits]

# 4. FK function
def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forward_kinematics(joint_angles):
    T = np.eye(4)
    for i in range(len(dh_params)):
        a, alpha, d, theta_offset = dh_params[i]
        theta = joint_angles[i] + theta_offset
        T = T @ dh_transform(a, alpha, d, theta)
    return T

# 5. Evaluate FK over simplified subspace (J1, J2, J3 only), filter out singularities
tolerance = 1e-3
valid_poses = []

for t1 in joint_grids[0]:
    for t2 in joint_grids[1]:
        for t3 in joint_grids[2]:
            joint_angles = [t1, t2, t3, 0, 0, 0, 0, 0]  # Lock joints J4–J8
            T = forward_kinematics(joint_angles)
            position = T[:3, 3]
            R = T[:3, :3]

            # 6. Filter: Check singularity via determinant of rotation matrix
            if abs(np.linalg.det(R) - 1) < tolerance:  # valid orientation
                valid_poses.append(position)

# Convert to DataFrame
df_filtered_workspace = pd.DataFrame(valid_poses, columns=["X (m)", "Y (m)", "Z (m)"])

# Export to CSV
df_filtered_workspace.to_csv("/mnt/data/filtered_ee_poses.csv", index=False)

# Plotting the filtered workspace
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_filtered_workspace["X (m)"], df_filtered_workspace["Y (m)"], df_filtered_workspace["Z (m)"], s=1, c='green')
ax.set_title('Filtered EE Poses (Non-Singular)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.tight_layout()
plt.show()
