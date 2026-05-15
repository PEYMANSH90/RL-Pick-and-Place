"""Experience replay buffer for off-policy RL algorithms.

Implements a fixed-size circular buffer that stores (s, a, r, s', done)
transitions.  Insertions are O(1); random-batch sampling converts the
NumPy arrays directly to PyTorch tensors on the requested device.

Design choices
--------------
- Pre-allocated NumPy arrays — avoids per-step memory allocation.
- Circular (ring) indexing — overwrites the oldest transitions once full.
- Separate `done` flag stored as float32 so it can be multiplied directly
  into the TD-target without branching: target = r + γ·(1−done)·V(s').

Reference: Mnih et al. (2015) "Human-level control through deep RL";
           Fujimoto et al. (2018) TD3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    """A mini-batch of transitions ready for a gradient update.

    All tensors have shape (batch_size, dim) and reside on the device
    passed to :meth:`ReplayBuffer.sample`.
    """
    obs:      torch.Tensor   # (B, obs_dim)
    action:   torch.Tensor   # (B, action_dim)
    reward:   torch.Tensor   # (B, 1)
    next_obs: torch.Tensor   # (B, obs_dim)
    done:     torch.Tensor   # (B, 1)  — 1.0 if terminal, 0.0 otherwise


class ReplayBuffer:
    """Fixed-capacity circular replay buffer.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector.
    action_dim:
        Dimensionality of the action vector.
    capacity:
        Maximum number of transitions to store.
        Default: 500 000 (paper value).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 500_000,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Pre-allocate contiguous arrays — much faster than a list of dicts
        self._obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self._action   = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward   = np.zeros((capacity, 1),          dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self._done     = np.zeros((capacity, 1),          dtype=np.float32)

        self._ptr  = 0   # write pointer
        self._size = 0   # current number of valid entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition.  Overwrites the oldest entry when full."""
        self._obs[self._ptr]      = obs
        self._action[self._ptr]   = action
        self._reward[self._ptr]   = reward
        self._next_obs[self._ptr] = next_obs
        self._done[self._ptr]     = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        """Sample a random mini-batch of transitions.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.
        device:
            PyTorch device for the returned tensors (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        Batch
            Named tensors ready for a gradient update step.

        Raises
        ------
        ValueError
            If the buffer contains fewer transitions than ``batch_size``.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} transitions but batch_size={batch_size}."
            )
        idx = np.random.randint(0, self._size, size=batch_size)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr[idx], dtype=torch.float32, device=device)

        return Batch(
            obs=_t(self._obs),
            action=_t(self._action),
            reward=_t(self._reward),
            next_obs=_t(self._next_obs),
            done=_t(self._done),
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, "
            f"size={self._size}/{self.capacity})"
        )
