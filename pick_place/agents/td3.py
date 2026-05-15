"""Twin Delayed Deep Deterministic Policy Gradient (TD3) — from scratch.

TD3 improves on DDPG with three key tricks:
  1. Twin critics  — two independent Q-networks; TD-target uses the minimum
                     → reduces Q-value overestimation.
  2. Delayed policy update — actor and target networks are updated every
     `policy_delay` critic steps → stabilises the policy gradient.
  3. Target policy smoothing — Gaussian noise added to the target action
     before computing the TD-target → regularises the critic.

All parameter names and defaults match the project paper exactly.

Reference: Fujimoto et al., "Addressing Function Approximation Error in
Actor-Critic Methods", ICML 2018.  https://arxiv.org/abs/1802.09477
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pick_place.agents.networks import Actor, Critic
from pick_place.agents.replay_buffer import Batch, ReplayBuffer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyper-parameter container
# ---------------------------------------------------------------------------

@dataclass
class TD3Config:
    """All TD3 hyper-parameters in one place.

    Defaults are the paper values for this project.
    """
    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])

    # Optimisation
    lr_actor:  float = 3e-4   # learning rate for the actor
    lr_critic: float = 3e-4   # learning rate for each critic

    # RL
    gamma:        float = 0.99    # discount factor
    tau:          float = 0.005   # soft-update coefficient
    policy_delay: int   = 2       # update actor every N critic steps

    # TD3-specific noise
    policy_noise: float = 0.2    # std of target-policy smoothing noise
    noise_clip:   float = 0.5    # clip smoothing noise to ±noise_clip
    expl_noise:   float = 0.1    # std of exploration noise during rollout

    # Replay
    buffer_capacity: int = 500_000
    batch_size:      int = 256
    learning_starts: int = 1_000  # random actions before training begins


# ---------------------------------------------------------------------------
# TD3 Agent
# ---------------------------------------------------------------------------

class TD3Agent:
    """Off-policy TD3 agent.

    Usage
    -----
    ::
        agent = TD3Agent(obs_dim=6, action_dim=1, action_max=2.0)
        obs, _ = env.reset()
        for step in range(total_steps):
            action = agent.select_action(obs, explore=True)
            next_obs, reward, done, _, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            agent.train_step()
            obs = next_obs if not done else env.reset()[0]

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation space.
    action_dim:
        Dimensionality of the action space.
    action_max:
        Absolute upper bound on each action dimension.
    cfg:
        :class:`TD3Config` instance.  Defaults to paper values.
    device:
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_max: float,
        cfg: TD3Config | None = None,
        device: str = "cpu",
    ) -> None:
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.action_max = action_max
        self.cfg        = cfg or TD3Config()
        self.device     = torch.device(device)

        # ---- Networks -------------------------------------------------------
        self.actor  = Actor(obs_dim, action_dim, action_max,
                            self.cfg.hidden_dims).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim,
                              self.cfg.hidden_dims).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim,
                              self.cfg.hidden_dims).to(self.device)

        # Target networks — frozen copies, updated via soft polyak averaging
        self.actor_target   = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        for net in (self.actor_target, self.critic1_target, self.critic2_target):
            for p in net.parameters():
                p.requires_grad_(False)

        # ---- Optimisers -----------------------------------------------------
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(),
                                             lr=self.cfg.lr_actor)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(),
                                             lr=self.cfg.lr_critic)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(),
                                             lr=self.cfg.lr_critic)

        # ---- Replay buffer ---------------------------------------------------
        self.buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            capacity=self.cfg.buffer_capacity,
        )

        # ---- Counters -------------------------------------------------------
        self._total_steps  = 0   # environment steps taken
        self._update_steps = 0   # gradient update steps performed

        logger.info(
            "TD3Agent | obs=%d  action=%d  action_max=%.2f  device=%s",
            obs_dim, action_dim, action_max, device,
        )

    # ------------------------------------------------------------------
    # Interaction API
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, *, explore: bool = True
    ) -> np.ndarray:
        """Choose an action for the current observation.

        Parameters
        ----------
        obs:
            Current environment observation, shape (obs_dim,).
        explore:
            If True, adds Gaussian exploration noise (training mode).
            If False, returns the deterministic actor output (eval mode).

        Returns
        -------
        np.ndarray, shape (action_dim,)
        """
        # Pure random actions before learning starts — fills the buffer quickly
        if explore and self._total_steps < self.cfg.learning_starts:
            return np.random.uniform(
                -self.action_max, self.action_max, size=(self.action_dim,)
            ).astype(np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().squeeze(0)

        if explore:
            noise = np.random.normal(
                0.0, self.cfg.expl_noise * self.action_max,
                size=action.shape,
            ).astype(np.float32)
            action = np.clip(action + noise, -self.action_max, self.action_max)

        return action

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition to the replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self._total_steps += 1

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def train_step(self) -> dict[str, float] | None:
        """Perform one gradient update step (if enough data is available).

        Returns
        -------
        dict with loss values, or None if training has not started yet.

        The update sequence follows the TD3 paper exactly:
          1. Sample a mini-batch from the replay buffer.
          2. Compute the TD-target with smoothed target policy.
          3. Update both critics (minimise Bellman residual).
          4. Every `policy_delay` steps: update the actor and soft-update
             all target networks.
        """
        if len(self.buffer) < self.cfg.learning_starts:
            return None

        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        metrics: dict[str, float] = {}

        # ---- Step 1: compute TD-target with target policy smoothing --------
        with torch.no_grad():
            noise = (
                torch.randn_like(batch.action) * self.cfg.policy_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)

            next_action = (
                self.actor_target(batch.next_obs) + noise
            ).clamp(-self.action_max, self.action_max)

            q1_next = self.critic1_target(batch.next_obs, next_action)
            q2_next = self.critic2_target(batch.next_obs, next_action)
            q_next  = torch.min(q1_next, q2_next)   # conservative estimate

            td_target = batch.reward + self.cfg.gamma * (1.0 - batch.done) * q_next

        # ---- Step 2: update Critic 1 ----------------------------------------
        q1_pred = self.critic1(batch.obs, batch.action)
        loss_c1 = F.mse_loss(q1_pred, td_target)
        self.critic1_opt.zero_grad()
        loss_c1.backward()
        self.critic1_opt.step()
        metrics["loss_critic1"] = loss_c1.item()

        # ---- Step 3: update Critic 2 ----------------------------------------
        q2_pred = self.critic2(batch.obs, batch.action)
        loss_c2 = F.mse_loss(q2_pred, td_target)
        self.critic2_opt.zero_grad()
        loss_c2.backward()
        self.critic2_opt.step()
        metrics["loss_critic2"] = loss_c2.item()

        self._update_steps += 1

        # ---- Step 4: delayed actor + target network updates -----------------
        if self._update_steps % self.cfg.policy_delay == 0:
            # Actor loss: maximise Q1(s, π(s))  ↔  minimise −Q1
            loss_actor = -self.critic1(batch.obs, self.actor(batch.obs)).mean()
            self.actor_opt.zero_grad()
            loss_actor.backward()
            self.actor_opt.step()
            metrics["loss_actor"] = loss_actor.item()

            # Soft (Polyak) update of all three target networks
            self._soft_update(self.actor,   self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save all network weights and optimiser states.

        Parameters
        ----------
        path:
            Directory where the checkpoint will be written.
            Created automatically if it does not exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor":          self.actor.state_dict(),
            "critic1":        self.critic1.state_dict(),
            "critic2":        self.critic2.state_dict(),
            "actor_target":   self.actor_target.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt":      self.actor_opt.state_dict(),
            "critic1_opt":    self.critic1_opt.state_dict(),
            "critic2_opt":    self.critic2_opt.state_dict(),
            "total_steps":    self._total_steps,
            "update_steps":   self._update_steps,
        }, path / "td3_checkpoint.pt")
        logger.info("Checkpoint saved to %s", path)

    def load(self, path: Path | str) -> None:
        """Load a checkpoint saved by :meth:`save`.

        Parameters
        ----------
        path:
            Directory containing ``td3_checkpoint.pt``.
        """
        ckpt = torch.load(
            Path(path) / "td3_checkpoint.pt",
            map_location=self.device,
        )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
        self.critic2_opt.load_state_dict(ckpt["critic2_opt"])
        self._total_steps  = ckpt["total_steps"]
        self._update_steps = ckpt["update_steps"]
        logger.info("Checkpoint loaded from %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _soft_update(self, online: torch.nn.Module,
                     target: torch.nn.Module) -> None:
        """Polyak averaging: θ_target ← τ·θ_online + (1−τ)·θ_target."""
        tau = self.cfg.tau
        for p_online, p_target in zip(online.parameters(),
                                      target.parameters()):
            p_target.data.mul_(1.0 - tau)
            p_target.data.add_(tau * p_online.data)
