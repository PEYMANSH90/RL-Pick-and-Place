"""Workspace analysis — samples the reachable end-effector positions of the IRB1600."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pick_place.kinematics.forward_kinematics import (
    DH_PARAMS,
    JOINT_LIMITS,
    forward_kinematics,
)


class WorkspaceAnalyzer:
    """Samples the robot's reachable workspace by discretising joint space.

    Parameters
    ----------
    n_samples:
        Number of evenly-spaced samples per active joint.  Total FK
        evaluations = n_samples ** n_active_joints — keep this small (≤ 20)
        when exploring all 6 joints simultaneously.
    active_joints:
        Indices (0-based) of the joints to sweep.  The remaining joints
        are locked at 0.  Defaults to [0, 1, 2] (first three joints),
        which is sufficient to map the translational workspace.
    singularity_tol:
        Maximum allowed deviation of det(R) from 1.0.  Poses with a
        more singular rotation matrix are discarded.
    """

    def __init__(
        self,
        n_samples: int = 30,
        active_joints: list[int] | None = None,
        singularity_tol: float = 1e-3,
    ) -> None:
        self.n_samples = n_samples
        self.active_joints = active_joints if active_joints is not None else [0, 1, 2]
        self.singularity_tol = singularity_tol

    # ------------------------------------------------------------------

    def compute(self) -> np.ndarray:
        """Return valid end-effector positions as an (N, 3) array [x, y, z]."""
        grids = [
            np.linspace(JOINT_LIMITS[j][0], JOINT_LIMITS[j][1], self.n_samples)
            for j in self.active_joints
        ]

        valid_positions: list[np.ndarray] = []
        for combo in np.ndindex(*[self.n_samples] * len(self.active_joints)):
            joint_angles = np.zeros(6)
            for k, joint_idx in enumerate(self.active_joints):
                joint_angles[joint_idx] = grids[k][combo[k]]

            T = forward_kinematics(joint_angles)
            R = T[:3, :3]

            if abs(np.linalg.det(R) - 1.0) < self.singularity_tol:
                valid_positions.append(T[:3, 3])

        return np.array(valid_positions)

    # ------------------------------------------------------------------

    def plot(self, positions: np.ndarray, save_path: Path | None = None) -> None:
        """Render a 3-D scatter plot of the sampled workspace.

        Parameters
        ----------
        positions:
            Output of :meth:`compute` — shape (N, 3).
        save_path:
            Optional path to save the figure (e.g. ``Path("workspace.png")``).
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=1,
            c="steelblue",
            alpha=0.5,
        )
        ax.set_title("ABB IRB1600 — Reachable Workspace (non-singular poses)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=150)
        plt.show()

    # ------------------------------------------------------------------

    def save(self, positions: np.ndarray, csv_path: Path) -> None:
        """Save workspace positions to a CSV file.

        Parameters
        ----------
        positions:
            Output of :meth:`compute` — shape (N, 3).
        csv_path:
            Destination file, e.g. ``Path("results/workspace.csv")``.
        """
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(positions, columns=["x_m", "y_m", "z_m"])
        df.to_csv(csv_path, index=False)
