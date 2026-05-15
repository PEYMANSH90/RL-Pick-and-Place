"""Training loop for the TD3 agent on the JointControlEnv.

The Trainer owns the agent and drives the environment interaction loop.
It delegates curriculum advancement to a CurriculumScheduler, logs metrics
via the standard logging module, and saves a checkpoint at the end of every
curriculum level and at training completion.

Separation of concerns
-----------------------
- Environment construction  → Trainer._make_env()
- Curriculum decisions      → CurriculumScheduler
- Network updates           → TD3Agent.train_step()
- Metric logging            → Python logging + TrainingResult
- Plotting / evaluation     → scripts/train.py  (not here)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pick_place.agents.td3 import TD3Agent, TD3Config
from pick_place.environments.joint_control_env import JointControlEnv
from pick_place.training.curriculum import CurriculumScheduler, DEFAULT_CURRICULUM, CurriculumLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Aggregated statistics produced by one complete training run."""

    episode_rewards:    list[float] = field(default_factory=list)
    episode_lengths:    list[int]   = field(default_factory=list)
    critic1_losses:     list[float] = field(default_factory=list)
    critic2_losses:     list[float] = field(default_factory=list)
    actor_losses:       list[float] = field(default_factory=list)
    level_boundaries:   list[int]   = field(default_factory=list)  # episode indices where level changed

    @property
    def best_reward(self) -> float:
        return max(self.episode_rewards) if self.episode_rewards else float("-inf")

    @property
    def n_episodes(self) -> int:
        return len(self.episode_rewards)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Runs the full TD3 + curriculum training loop.

    Parameters
    ----------
    td3_cfg:
        TD3 hyperparameter config.  Defaults to paper values.
    curriculum_levels:
        List of curriculum stages.  Defaults to three-stage paper schedule.
    checkpoint_dir:
        Directory for saving model checkpoints.
    max_episodes:
        Hard cap on total episodes across all curriculum levels.
        Default 200 (paper specification).
    log_interval:
        Print a log line every N episodes.
    device:
        Torch device string, ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        td3_cfg:            TD3Config | None             = None,
        curriculum_levels:  list[CurriculumLevel] | None = None,
        checkpoint_dir:     Path | str                   = Path("checkpoints"),
        max_episodes:       int                          = 200,
        log_interval:       int                          = 5,
        device:             str                          = "cpu",
    ) -> None:
        self.td3_cfg           = td3_cfg or TD3Config()
        self.curriculum_levels = curriculum_levels or DEFAULT_CURRICULUM
        self.checkpoint_dir    = Path(checkpoint_dir)
        self.max_episodes      = max_episodes
        self.log_interval      = log_interval
        self.device            = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """Run training over all curriculum levels and return aggregated stats.

        Returns
        -------
        TrainingResult
            Per-episode rewards, losses, level boundaries, and best reward.
        """
        scheduler = CurriculumScheduler(levels=self.curriculum_levels)
        result    = TrainingResult()

        # Build initial environment and agent
        env   = self._make_env(scheduler.load_scale)
        agent = TD3Agent(
            obs_dim    = env.observation_space.shape[0],
            action_dim = env.action_space.shape[0],
            action_max = float(env.action_space.high[0]),
            cfg        = self.td3_cfg,
            device     = self.device,
        )

        logger.info("Training started.  max_episodes=%d", self.max_episodes)
        t0 = time.perf_counter()

        total_episodes = 0

        while not scheduler.is_complete and total_episodes < self.max_episodes:
            obs, _ = env.reset()
            episode_reward  = 0.0
            episode_length  = 0
            ep_c1_losses: list[float] = []
            ep_c2_losses: list[float] = []
            ep_ac_losses: list[float] = []

            # ---- One episode --------------------------------------------
            done = False
            while not done:
                action   = agent.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done     = terminated or truncated

                agent.store(obs, action, reward, next_obs, done)
                metrics = agent.train_step()

                if metrics:
                    ep_c1_losses.append(metrics.get("loss_critic1", 0.0))
                    ep_c2_losses.append(metrics.get("loss_critic2", 0.0))
                    if "loss_actor" in metrics:
                        ep_ac_losses.append(metrics["loss_actor"])

                obs             = next_obs
                episode_reward += reward
                episode_length += 1

            # ---- Record episode -----------------------------------------
            total_episodes += 1
            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(episode_length)
            if ep_c1_losses:
                result.critic1_losses.append(float(np.mean(ep_c1_losses)))
                result.critic2_losses.append(float(np.mean(ep_c2_losses)))
            if ep_ac_losses:
                result.actor_losses.append(float(np.mean(ep_ac_losses)))

            # ---- Log ----------------------------------------------------
            if total_episodes % self.log_interval == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Episode %3d | level=%s | reward=%7.2f | "
                    "best=%7.2f | steps=%d | %.1f s elapsed",
                    total_episodes,
                    scheduler.current_level.label,
                    episode_reward,
                    result.best_reward,
                    agent._total_steps,
                    elapsed,
                )

            # ---- Curriculum advancement ---------------------------------
            advanced = scheduler.record_episode(episode_reward)
            if advanced:
                result.level_boundaries.append(total_episodes)

                # Save a checkpoint at each level transition
                ckpt_path = self.checkpoint_dir / f"level_{scheduler.level_index - 1}"
                agent.save(ckpt_path)
                logger.info("Checkpoint saved → %s", ckpt_path)

                if not scheduler.is_complete:
                    # Rebuild env with new load scale; agent continues unchanged
                    env = self._make_env(scheduler.load_scale)

        # ---- Final checkpoint -------------------------------------------
        final_path = self.checkpoint_dir / "final"
        agent.save(final_path)
        elapsed = time.perf_counter() - t0

        logger.info(
            "Training complete.  %d episodes | best reward = %.2f | %.1f s",
            total_episodes, result.best_reward, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_env(self, load_scale: float) -> JointControlEnv:
        """Construct a fresh environment for the given curriculum scale."""
        logger.info("Building JointControlEnv  load_scale=%.2f", load_scale)
        return JointControlEnv(load_scale=load_scale)
