"""Actor and Critic neural networks for the TD3 algorithm.

Both networks are MLPs built from scratch using PyTorch — no RL library
dependency.  Architecture follows the paper:
  - Two hidden layers of 256 units each
  - ReLU activations in hidden layers
  - Actor output: tanh scaled to the action range
  - Critic: twin networks (Critic1, Critic2) to reduce Q-value overestimation

Reference: Fujimoto et al., "Addressing Function Approximation Error in
Actor-Critic Methods" (TD3 paper), ICML 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Helper: stack Linear → Activation layers, then a final Linear."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy network π_θ(s) → a.

    Outputs a continuous action in [−action_max, +action_max] by applying
    tanh and then scaling.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    action_dim:
        Dimensionality of the action vector.
    action_max:
        Absolute upper bound on each action dimension.
    hidden_dims:
        Sizes of the hidden layers.  Default: [256, 256] (paper value).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_max: float,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        self.action_max = action_max
        self.net = _build_mlp(obs_dim, hidden_dims, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action scaled to [−action_max, +action_max]."""
        return self.action_max * torch.tanh(self.net(obs))


class Critic(nn.Module):
    """Twin Q-networks Q_φ1(s, a) and Q_φ2(s, a).

    TD3 uses two independent critics to compute a conservative Q-target:
        y = r + γ · min(Q1(s', ã), Q2(s', ã))
    Both heads share the same class but are instantiated separately in TD3Agent
    so their parameters are optimised independently.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    action_dim:
        Dimensionality of the action vector.
    hidden_dims:
        Sizes of the hidden layers.  Default: [256, 256] (paper value).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        in_dim = obs_dim + action_dim
        self.net = _build_mlp(in_dim, hidden_dims, out_dim=1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Return Q-value estimate, shape (batch, 1)."""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)
