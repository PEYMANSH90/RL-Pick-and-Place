"""Forward kinematics for the ABB IRB1600-6/1.2 using the DH convention.

DH parameters taken from Table 2 of the paper:
  "Intelligent Control of Robots with Minimal Power Consumption
   in Pick-and-Place Operations"

Frame convention: modified DH (Craig).  8 frames total (6 joints +
two fixed wrist frames T4a, T6a that account for the wrist geometry).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Robot constants (DH table, joint limits) — single source of truth
# ---------------------------------------------------------------------------

# Columns: a (m), alpha (rad), d (m), theta_offset (rad)
DH_PARAMS: np.ndarray = np.array([
    [0.150,  -np.pi / 2, 0.286,  0.0],          # frame 1
    [0.475,   0.0,       0.0,   -np.pi / 2],     # frame 2
    [0.500,   0.0,       0.0,    np.pi / 2],     # frame 3
    [0.0,     np.pi / 2, 0.0,    np.pi / 2],     # frame 4a (fixed wrist)
    [0.0,    -np.pi / 2, 0.1,    0.0],           # frame 4
    [0.165,   0.0,       0.0,   -np.pi / 2],     # frame 5
    [0.0,     np.pi / 2, 0.0,    np.pi / 2],     # frame 6a (fixed wrist)
    [0.0,     0.0,       0.2,    0.0],           # frame 6 (tool)
], dtype=float)

# Joint limits in radians (Table 1).  Indices match the 6 active joints.
JOINT_LIMITS: list[tuple[float, float]] = [
    (-2.618, 2.618),   # J1  ±150°
    (-1.100, 1.920),   # J2  -63° / +110°
    (-2.618, 2.618),   # J3  ±150°
    (-3.142, 3.142),   # J4  ±180°
    (-2.007, 2.007),   # J5  ±115°
    (-3.142, 3.142),   # J6  ±180°
]

# Gear ratios (motor → joint) used by the RL torque controller
GEAR_RATIOS: list[float] = [120.0, 120.0, 120.0, 100.0, 100.0, 100.0]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Return the 4×4 homogeneous DH transformation matrix for one frame.

    Parameters
    ----------
    a, alpha, d, theta:
        Standard DH parameters for a single link frame.

    Returns
    -------
    np.ndarray, shape (4, 4)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,  -st * ca,  st * sa,  a * ct],
        [st,   ct * ca, -ct * sa,  a * st],
        [0.0,  sa,       ca,       d],
        [0.0,  0.0,      0.0,      1.0],
    ])


def forward_kinematics(
    joint_angles: np.ndarray,
    dh_params: np.ndarray = DH_PARAMS,
) -> np.ndarray:
    """Compute the end-effector pose via the full DH kinematic chain.

    The function accepts either 6 active joint values (preferred) or 8
    values when the two fixed wrist offsets are included explicitly.

    Parameters
    ----------
    joint_angles:
        Array of 6 active joint angles [J1 … J6] in radians.
    dh_params:
        DH table with shape (N_frames, 4) — defaults to the IRB1600 table.

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        Homogeneous transformation from the base frame to the tool frame.
    """
    angles = np.asarray(joint_angles, dtype=float)

    # Map 6 active joints → 8 frames (frames 4 and 7 are fixed wrist frames)
    if angles.size == 6:
        full_angles = np.zeros(8)
        full_angles[0] = angles[0]   # J1 → frame 1
        full_angles[1] = angles[1]   # J2 → frame 2
        full_angles[2] = angles[2]   # J3 → frame 3
        # frame 4  (index 3) is fixed — stays 0
        full_angles[4] = angles[3]   # J4 → frame 4 (after fixed wrist)
        full_angles[5] = angles[4]   # J5 → frame 5
        # frame 6a (index 6) is fixed — stays 0
        full_angles[7] = angles[5]   # J6 → frame 6 (tool)
    else:
        full_angles = angles

    T = np.eye(4)
    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = full_angles[i] + theta_offset
        T = T @ dh_transform(a, alpha, d, theta)
    return T
