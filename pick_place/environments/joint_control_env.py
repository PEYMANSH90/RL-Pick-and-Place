"""Gymnasium environment for RL-based joint torque control.

The task: an agent commands a motor torque T* at each timestep.  The motor
drives a gearbox (ratio N) that produces a joint torque T_joint = N · T*.
The joint must compensate a time-varying external load drawn from the paper's
12-step load profile.  The agent observes a PID-like error state and is
rewarded for minimising tracking error while keeping actions small.

Reference: Section 2.4 of the project paper.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load profile from the paper (Section 2.4, variable external loads in N·m)
# ---------------------------------------------------------------------------
PAPER_LOADS: list[float] = [10.0, 75.0, -25.0, 40.0, 10.0, -10.0,
                              -30.0, 5.0, 18.0, 10.0, 18.0, 17.0]


class JointControlEnv(gym.Env):
    """Single-joint torque-tracking environment for the ABB IRB1600.

    Observation space (6-D, all normalised to roughly [-1, 1])
    ----------------------------------------------------------
    0  error          : (load - T_joint) / LOAD_SCALE
    1  prev_error     : previous step's error / LOAD_SCALE
    2  integral_error : accumulated error / LOAD_SCALE
    3  deriv_error    : rate of error change / LOAD_SCALE
    4  motor_torque   : current T* / ACTION_SCALE
    5  load_torque    : current external load / LOAD_SCALE

    Action space (1-D continuous)
    -----------------------------
    T* in [−ACTION_MAX, +ACTION_MAX] N·m  (motor torque command)

    Reward
    ------
    r_t = −( (error / LOAD_SCALE)² + λ · (T*)² )
    where λ = 0.01 weights the action-magnitude penalty.

    Parameters
    ----------
    load_scale:
        Fraction of the paper load profile to apply.  1.0 = full loads,
        0.5 = half loads, 0.1 = 10% loads (for curriculum learning).
    gear_ratio:
        Motor-to-joint gear ratio N.  Default 120 (IRB1600 paper value).
    steps_per_episode:
        Horizon length.  Default 100 steps = 16 s at dt=0.16 s.
    dt:
        Timestep duration in seconds.  Default 0.16 s (160 ms).
    action_max:
        Upper bound on |T*|.  Default 2.0 N·m.
    action_penalty:
        Weight λ on the action-magnitude term in the reward.
    """

    metadata = {"render_modes": []}

    # Normalisation scale (matches max |load| in the paper profile × safety margin)
    LOAD_SCALE: float = 100.0

    def __init__(
        self,
        load_scale: float = 1.0,
        gear_ratio: float = 120.0,
        steps_per_episode: int = 100,
        dt: float = 0.16,
        action_max: float = 2.0,
        action_penalty: float = 0.01,
    ) -> None:
        super().__init__()

        self.load_scale = load_scale
        self.gear_ratio = gear_ratio
        self.steps_per_episode = steps_per_episode
        self.dt = dt
        self.action_max = action_max
        self.action_penalty = action_penalty

        # Pre-compute the full load profile for one episode
        self._load_profile: np.ndarray = self._build_load_profile()

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=-action_max,
            high=action_max,
            shape=(1,),
            dtype=np.float32,
        )
        obs_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_max, high=obs_max, dtype=np.float32
        )

        # State — initialised in reset()
        self._step: int = 0
        self._motor_torque: float = 0.0
        self._prev_error: float = 0.0
        self._integral_error: float = 0.0
        self._deriv_error: float = 0.0
        self._current_load: float = 0.0

        logger.info(
            "JointControlEnv: load_scale=%.1f  gear_ratio=%.0f  "
            "steps=%d  dt=%.3f s",
            load_scale, gear_ratio, steps_per_episode, dt,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step = 0
        self._motor_torque = 0.0
        self._prev_error = 0.0
        self._integral_error = 0.0
        self._deriv_error = 0.0
        self._current_load = float(self._load_profile[0])

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        t_star = float(np.clip(action[0], -self.action_max, self.action_max))
        t_joint = t_star * self.gear_ratio

        self._motor_torque = t_star
        self._step += 1

        # Advance load
        load_idx = min(self._step, len(self._load_profile) - 1)
        self._current_load = float(self._load_profile[load_idx])

        # PID-style error signals
        error = self._current_load - t_joint
        self._integral_error += error * self.dt
        self._deriv_error = (error - self._prev_error) / self.dt
        self._prev_error = error

        obs = self._get_obs()
        reward = self._compute_reward(error, t_star)
        terminated = self._step >= self.steps_per_episode
        return obs, reward, terminated, False, {}

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Properties exposed for logging / evaluation
    # ------------------------------------------------------------------

    @property
    def current_load(self) -> float:
        return self._current_load

    @property
    def current_motor_torque(self) -> float:
        return self._motor_torque

    @property
    def load_profile(self) -> np.ndarray:
        return self._load_profile.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_load_profile(self) -> np.ndarray:
        """Tile the paper load profile to fill exactly steps_per_episode steps."""
        steps_per_second = round(1.0 / self.dt)   # ≈ 6 at dt=0.16
        scaled = [v * self.load_scale for v in PAPER_LOADS]

        # Each load value lasts 1 second → repeat by steps_per_second
        base = np.repeat(scaled, steps_per_second)

        # Tile until we have enough steps, then trim
        reps = int(np.ceil(self.steps_per_episode / len(base)))
        profile = np.tile(base, reps)[: self.steps_per_episode]
        return profile.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        s = self.LOAD_SCALE
        a = self.action_max
        return np.array([
            self._prev_error       / s,
            self._prev_error       / s,   # prev_error (updated after error assigned)
            self._integral_error   / s,
            self._deriv_error      / s,
            self._motor_torque     / a,
            self._current_load     / s,
        ], dtype=np.float32).clip(-1.0, 1.0)

    def _compute_reward(self, error: float, t_star: float) -> float:
        """r = −( normalised_error² + λ · action² )"""
        norm_error = error / self.LOAD_SCALE
        return -(norm_error ** 2 + self.action_penalty * t_star ** 2)
