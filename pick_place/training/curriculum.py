"""Curriculum learning scheduler for progressive load difficulty.

The agent starts on a scaled-down version of the load profile and advances
to the next level once it has trained for a minimum number of episodes or
reached a reward threshold.  This mirrors the three-stage curriculum in
the project paper (10 % → 50 % → 100 % of paper loads).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CurriculumLevel:
    """One stage of the curriculum.

    Parameters
    ----------
    load_scale:
        Fraction of the paper load profile applied to the environment.
    episodes:
        Maximum number of episodes to train at this level before
        advancing, regardless of reward.
    reward_threshold:
        If the rolling mean reward (over ``window`` episodes) exceeds
        this value, advance immediately.
    label:
        Human-readable name shown in logs.
    """
    load_scale:        float
    episodes:          int
    reward_threshold:  float
    label:             str = ""


# Default three-level curriculum from the paper
DEFAULT_CURRICULUM: list[CurriculumLevel] = [
    CurriculumLevel(load_scale=0.10, episodes=30,
                    reward_threshold=-600.0, label="Level-0 (10 %)"),
    CurriculumLevel(load_scale=0.50, episodes=30,
                    reward_threshold=-550.0, label="Level-1 (50 %)"),
    CurriculumLevel(load_scale=1.00, episodes=30,
                    reward_threshold=-493.0, label="Level-2 (100 %)"),
]


class CurriculumScheduler:
    """Tracks training progress and decides when to advance curriculum levels.

    Parameters
    ----------
    levels:
        Ordered list of :class:`CurriculumLevel` definitions.
    reward_window:
        Number of recent episodes used to compute the rolling mean reward
        for the advancement check.
    """

    def __init__(
        self,
        levels: list[CurriculumLevel] | None = None,
        reward_window: int = 5,
    ) -> None:
        self.levels        = levels or DEFAULT_CURRICULUM
        self.reward_window = reward_window
        self._level_idx    = 0
        self._episode_count = 0         # episodes at the current level
        self._reward_history: list[float] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_level(self) -> CurriculumLevel:
        return self.levels[self._level_idx]

    @property
    def load_scale(self) -> float:
        return self.current_level.load_scale

    @property
    def level_index(self) -> int:
        return self._level_idx

    @property
    def is_complete(self) -> bool:
        """True once the final level has been exhausted."""
        return self._level_idx >= len(self.levels)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def record_episode(self, episode_reward: float) -> bool:
        """Register the result of one completed episode.

        Parameters
        ----------
        episode_reward:
            Total undiscounted reward for the finished episode.

        Returns
        -------
        bool
            True if the curriculum just advanced to the next level (or
            finished), so the caller knows to rebuild the environment.
        """
        if self.is_complete:
            return False

        self._episode_count   += 1
        self._reward_history.append(episode_reward)

        level = self.current_level
        rolling_mean = self._rolling_mean()

        advanced = False
        if self._episode_count >= level.episodes:
            logger.info(
                "%s: reached episode limit (%d).  Rolling mean reward = %.2f.",
                level.label, level.episodes, rolling_mean,
            )
            advanced = self._advance()

        elif rolling_mean > level.reward_threshold:
            logger.info(
                "%s: reward threshold met (%.2f > %.2f) after %d episodes.",
                level.label, rolling_mean,
                level.reward_threshold, self._episode_count,
            )
            advanced = self._advance()

        return advanced

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _advance(self) -> bool:
        self._level_idx    += 1
        self._episode_count = 0
        self._reward_history.clear()
        if self.is_complete:
            logger.info("Curriculum complete — all levels finished.")
        else:
            logger.info("Advancing to %s.", self.current_level.label)
        return True

    def _rolling_mean(self) -> float:
        recent = self._reward_history[-self.reward_window:]
        return sum(recent) / len(recent) if recent else float("-inf")
