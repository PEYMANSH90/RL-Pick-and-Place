"""Inverse statics — joint-torque computation via Recursive Newton-Euler (RNE).

Gravity torques are computed using the roboticstoolbox DHRobot RNE solver.
End-effector payload torques are added via the manipulator Jacobian:

    tau_total = tau_gravity + J(q)^T * F_payload

Reference: Section 2.3 of the project paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH

from pick_place.kinematics.forward_kinematics import JOINT_LIMITS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Robot physical constants (paper values)
# ---------------------------------------------------------------------------

# Link masses [kg] for frames 1–6 (Table 2 of paper)
LINK_MASSES: list[float] = [70.0, 60.0, 50.0, 30.0, 20.0, 10.0]

# Default end-effector payload [kg] used in the paper's case study
DEFAULT_PAYLOAD_KG: float = 52.0

# IK solutions from Table 3 of the paper: [J2, J3, J5] in radians
PAPER_CONFIGURATIONS: dict[str, list[float]] = {
    "JK1": [-1.0996,  0.6283,  1.2601],
    "JK2": [-0.8482,  1.5080, -1.5045],
    "JK3": [-0.7226,  0.3770,  1.6371],
    "JK4": [-0.7226,  1.5080, -1.6301],
    "JK5": [-0.3456,  1.3823, -2.0071],
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TorqueResult:
    """Per-joint torques for one robot configuration.

    Attributes
    ----------
    label:
        Human-readable name for this configuration (e.g. "JK1").
    joint_angles_6dof:
        Full 6-DOF joint vector [J1 … J6] in radians.
    tau_gravity:
        Gravity torques from RNE, shape (8,) matching the 8-frame model.
    tau_ee:
        Torques due to the end-effector payload, shape (8,).
    tau_total:
        Sum tau_gravity + tau_ee, shape (8,).
    """

    label: str
    joint_angles_6dof: np.ndarray
    tau_gravity: np.ndarray
    tau_ee: np.ndarray
    tau_total: np.ndarray

    @property
    def active_torques(self) -> dict[str, float]:
        """Return torques for the three active joints J2, J3, J5 (N·m)."""
        return {
            "J2": float(self.tau_total[1]),
            "J3": float(self.tau_total[2]),
            "J5": float(self.tau_total[5]),
        }

    @property
    def total_active(self) -> float:
        """Sum of J2 + J3 + J5 torques (N·m)."""
        t = self.active_torques
        return t["J2"] + t["J3"] + t["J5"]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class InverseStatics:
    """Computes static joint torques for the ABB IRB1600 using RNE.

    The robot is modelled with 8 DH frames (6 active joints + 2 fixed
    wrist frames T4a, T6a) to match the paper's Table 2 exactly.

    Parameters
    ----------
    link_masses:
        List of 6 link masses [kg] for frames 1–6.
        Defaults to the paper values: [70, 60, 50, 30, 20, 10] kg.
    payload_kg:
        Mass of the end-effector payload [kg].  Applied as a vertical
        downward force at the tool frame.
    """

    GRAVITY: np.ndarray = np.array([0.0, 0.0, -9.81])

    def __init__(
        self,
        link_masses: list[float] | None = None,
        payload_kg: float = DEFAULT_PAYLOAD_KG,
    ) -> None:
        self.link_masses = link_masses if link_masses is not None else LINK_MASSES
        self.payload_kg = payload_kg
        self._robot: DHRobot = self._build_robot()
        self._ee_force: np.ndarray = np.array([0.0, 0.0, -9.81 * payload_kg])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        joint_angles_6dof: np.ndarray,
        label: str = "",
    ) -> TorqueResult:
        """Compute static joint torques for a single 6-DOF configuration.

        Parameters
        ----------
        joint_angles_6dof:
            Array of 6 joint angles [J1 … J6] in radians.
        label:
            Optional name tag for the result (e.g. "JK1").

        Returns
        -------
        TorqueResult
        """
        q6 = np.asarray(joint_angles_6dof, dtype=float)
        q8 = self._to_8dof(q6)

        tau_gravity = self._robot.rne(
            q=q8, qd=np.zeros(8), qdd=np.zeros(8)
        )

        J = self._robot.jacob0(q8)[:3, :]   # position-part of Jacobian
        tau_ee = J.T @ self._ee_force

        tau_total = tau_gravity + tau_ee

        logger.debug(
            "%s | J2=%.2f  J3=%.2f  J5=%.2f  sum=%.2f N·m",
            label or "config",
            tau_total[1], tau_total[2], tau_total[5],
            tau_total[1] + tau_total[2] + tau_total[5],
        )

        return TorqueResult(
            label=label,
            joint_angles_6dof=q6,
            tau_gravity=tau_gravity,
            tau_ee=tau_ee,
            tau_total=tau_total,
        )

    def evaluate_configurations(
        self,
        configurations: dict[str, list[float]],
    ) -> dict[str, TorqueResult]:
        """Evaluate a set of named [J2, J3, J5] configurations.

        The three active values are placed into the correct positions of
        a full 6-DOF array (J1=J4=J6=0).

        Parameters
        ----------
        configurations:
            Mapping of label → [J2, J3, J5] in radians.
            Matches the format of :data:`PAPER_CONFIGURATIONS`.

        Returns
        -------
        dict mapping label → TorqueResult
        """
        results: dict[str, TorqueResult] = {}
        for label, angles_3dof in configurations.items():
            q6 = np.zeros(6)
            q6[1], q6[2], q6[4] = angles_3dof   # J2, J3, J5
            results[label] = self.compute(q6, label=label)
        return results

    def minimum_torque_config(
        self,
        configurations: dict[str, list[float]],
    ) -> tuple[str, TorqueResult]:
        """Return the label and result with the smallest |sum of active torques|.

        Parameters
        ----------
        configurations:
            Same format as :meth:`evaluate_configurations`.

        Returns
        -------
        (label, TorqueResult) for the minimum-torque configuration.
        """
        results = self.evaluate_configurations(configurations)
        best_label = min(results, key=lambda k: abs(results[k].total_active))
        return best_label, results[best_label]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_robot(self) -> DHRobot:
        """Construct the 8-frame DHRobot matching the paper's DH table."""
        m = self.link_masses
        links = [
            # Frame 1
            RevoluteDH(a=0.150, alpha=-np.pi/2, d=0.286, offset=0.0,
                       m=m[0], r=[0.075, 0.0, 0.0]),
            # Frame 2 (θ2 − π/2)
            RevoluteDH(a=0.475, alpha=0.0,       d=0.0,   offset=-np.pi/2,
                       m=m[1], r=[0.2375, 0.0, 0.0]),
            # Frame 3 (θ3 + π/2)
            RevoluteDH(a=0.500, alpha=0.0,       d=0.0,   offset=np.pi/2,
                       m=m[2], r=[0.25, 0.0, 0.0]),
            # Frame 4a — fixed wrist (no active joint)
            RevoluteDH(a=0.0,   alpha=np.pi/2,   d=0.0,   offset=np.pi/2,
                       m=m[3], r=[0.0, 0.0, 0.0]),
            # Frame 4
            RevoluteDH(a=0.0,   alpha=-np.pi/2,  d=0.1,   offset=0.0,
                       m=m[3], r=[0.0, 0.0, 0.05]),
            # Frame 5 (θ5 − π/2)
            RevoluteDH(a=0.165, alpha=0.0,       d=0.0,   offset=-np.pi/2,
                       m=m[4], r=[0.0825, 0.0, 0.0]),
            # Frame 6a — fixed wrist (no active joint)
            RevoluteDH(a=0.0,   alpha=np.pi/2,   d=0.0,   offset=np.pi/2,
                       m=m[5], r=[0.0, 0.0, 0.0]),
            # Frame 6 / tool
            RevoluteDH(a=0.0,   alpha=0.0,       d=0.2,   offset=0.0,
                       m=m[5], r=[0.0, 0.0, 0.1]),
        ]
        robot = DHRobot(links, name="IRB1600")
        robot.gravity = self.GRAVITY.tolist()
        return robot

    @staticmethod
    def _to_8dof(q6: np.ndarray) -> np.ndarray:
        """Map 6 active joint angles → 8-frame vector (fixed frames stay 0)."""
        q8 = np.zeros(8)
        q8[0] = q6[0]   # J1
        q8[1] = q6[1]   # J2
        q8[2] = q6[2]   # J3
        # q8[3] = 0      # frame 4a — fixed
        q8[4] = q6[3]   # J4
        q8[5] = q6[4]   # J5
        # q8[6] = 0      # frame 6a — fixed
        q8[7] = q6[5]   # J6
        return q8
