"""Numerical inverse kinematics for the ABB IRB1600 using L-BFGS-B optimisation.

The solver treats IK as an unconstrained optimisation problem: minimise the
Cartesian position error subject to soft joint-limit penalties.  Only the
XZ-plane position is used as the target (the paper fixes J1=J4=J6=0, reducing
the task to a planar problem).

Reference: Section 2.2 of the project paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import minimize, OptimizeResult

from pick_place.kinematics.forward_kinematics import JOINT_LIMITS, forward_kinematics

logger = logging.getLogger(__name__)


@dataclass
class IKSolution:
    """A single accepted inverse-kinematics solution."""

    joint_angles: np.ndarray   # shape (3,): J2, J3, J5 in radians
    position_error: float      # Euclidean XZ error at convergence (m)
    success: bool


@dataclass
class IKResult:
    """Collection of solutions returned by :class:`InverseKinematicsSolver`."""

    solutions: list[IKSolution] = field(default_factory=list)

    @property
    def n_solutions(self) -> int:
        return len(self.solutions)

    def best(self) -> IKSolution | None:
        """Return the solution with the smallest position error."""
        if not self.solutions:
            return None
        return min(self.solutions, key=lambda s: s.position_error)


class InverseKinematicsSolver:
    """Numerical IK solver using multi-start L-BFGS-B optimisation.

    The solver optimises over three joints (J2, J3, J5) while fixing
    J1 = J4 = J6 = 0, reducing the problem to a planar XZ task.

    Parameters
    ----------
    target_position:
        Desired end-effector position [x, y, z] in metres.
    tolerance:
        Maximum acceptable XZ position error (m).  Default 0.01 m.
    num_guesses:
        Number of random initial guesses for the multi-start search.
    max_iter:
        Maximum L-BFGS-B iterations per trial.
    limit_penalty:
        Weight of the quadratic joint-limit penalty term.
    """

    # Indices of the three joints being optimised, within the 6-DOF array
    _ACTIVE = [1, 2, 4]   # J2, J3, J5 (0-based)
    _ACTIVE_LIMITS = [JOINT_LIMITS[1], JOINT_LIMITS[2], JOINT_LIMITS[4]]

    def __init__(
        self,
        target_position: np.ndarray,
        tolerance: float = 0.01,
        num_guesses: int = 500,
        max_iter: int = 20,
        limit_penalty: float = 1_000.0,
    ) -> None:
        self.target = np.asarray(target_position, dtype=float)
        self.tolerance = tolerance
        self.num_guesses = num_guesses
        self.max_iter = max_iter
        self.limit_penalty = limit_penalty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> IKResult:
        """Run the multi-start optimisation and return all accepted solutions.

        Returns
        -------
        IKResult
            Container with all solutions that satisfy the tolerance and
            joint-limit constraints.
        """
        result = IKResult()
        rng = np.random.default_rng()

        for trial in range(self.num_guesses):
            guess = np.array([
                rng.uniform(lo, hi) for lo, hi in self._ACTIVE_LIMITS
            ])
            opt: OptimizeResult = minimize(
                self._cost,
                guess,
                method="L-BFGS-B",
                bounds=self._ACTIVE_LIMITS,
                options={"disp": False, "maxiter": self.max_iter},
            )

            error = self._position_error(opt.x)
            within_limits = self._check_limits(opt.x)

            if error < self.tolerance and within_limits:
                result.solutions.append(
                    IKSolution(
                        joint_angles=opt.x.copy(),
                        position_error=error,
                        success=True,
                    )
                )
                logger.debug(
                    "Trial %d accepted: J2=%.4f J3=%.4f J5=%.4f  err=%.6f m",
                    trial + 1, *opt.x, error,
                )

        logger.info(
            "IK search complete: %d / %d trials accepted.",
            result.n_solutions, self.num_guesses,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _full_joint_angles(self, active: np.ndarray) -> np.ndarray:
        """Expand 3 active joints → full 6-DOF array (others fixed at 0)."""
        q = np.zeros(6)
        for i, idx in enumerate(self._ACTIVE):
            q[idx] = active[i]
        return q

    def _position_error(self, active: np.ndarray) -> float:
        """XZ Euclidean distance between the FK position and the target."""
        q = self._full_joint_angles(active)
        T = forward_kinematics(q)
        pos = T[:3, 3]
        return float(np.linalg.norm(pos[[0, 2]] - self.target[[0, 2]]))

    def _cost(self, active: np.ndarray) -> float:
        """Objective: position error + quadratic joint-limit penalty."""
        error = self._position_error(active)
        penalty = 0.0
        for angle, (lo, hi) in zip(active, self._ACTIVE_LIMITS):
            if angle < lo:
                penalty += self.limit_penalty * (lo - angle) ** 2
            elif angle > hi:
                penalty += self.limit_penalty * (angle - hi) ** 2
        return error + penalty

    def _check_limits(self, active: np.ndarray) -> bool:
        return all(
            lo <= angle <= hi
            for angle, (lo, hi) in zip(active, self._ACTIVE_LIMITS)
        )
