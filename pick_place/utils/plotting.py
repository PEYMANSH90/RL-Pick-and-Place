"""Plotting utilities for training curves and load-tracking visualisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_training_curve(
    episode_rewards: list[float],
    level_boundaries: list[int] | None = None,
    window: int = 10,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Plot episode reward over time with a smoothed rolling mean.

    Parameters
    ----------
    episode_rewards:
        List of total undiscounted reward per episode.
    level_boundaries:
        Episode indices where the curriculum advanced to the next level.
        Rendered as vertical dashed lines.
    window:
        Rolling mean window size (episodes).
    save_path:
        If provided, the figure is saved here (PNG).
    show:
        If True, calls ``plt.show()`` (blocks until the window is closed).
    """
    rewards = np.array(episode_rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Raw rewards (light)
    ax.plot(episodes, rewards, color="steelblue", alpha=0.3,
            linewidth=0.8, label="Episode reward")

    # Rolling mean (bold)
    if len(rewards) >= window:
        kernel  = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        ax.plot(
            episodes[window - 1:], smoothed,
            color="steelblue", linewidth=2.0,
            label=f"Rolling mean (w={window})",
        )

    # Curriculum level boundaries
    if level_boundaries:
        for i, ep in enumerate(level_boundaries):
            ax.axvline(
                ep, color="coral", linestyle="--", linewidth=1.2,
                label="Level advance" if i == 0 else None,
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("TD3 Training — JointControlEnv")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def plot_tracking(
    steps: np.ndarray,
    loads: np.ndarray,
    joint_torques: np.ndarray,
    errors: np.ndarray,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Three-panel plot: load vs joint torque, motor torque, tracking error.

    Parameters
    ----------
    steps:
        Timestep indices (x-axis).
    loads:
        External load torque at each step (N·m).
    joint_torques:
        Agent joint torque output = T* × gear_ratio (N·m).
    errors:
        Tracking error = load − joint_torque (N·m).
    save_path / show:
        Persistence options (same as :func:`plot_training_curve`).
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # Panel 1: load tracking
    axes[0].plot(steps, loads,        color="steelblue",  linewidth=2,
                 label="Target load (N·m)")
    axes[0].plot(steps, joint_torques, color="coral", linewidth=2,
                 linestyle="--", label="Agent joint torque (N·m)")
    axes[0].set_ylabel("Torque (N·m)")
    axes[0].set_title("Joint Torque Tracking")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: tracking error
    axes[1].plot(steps, errors, color="darkorange", linewidth=1.5,
                 label="Tracking error (N·m)")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    axes[1].set_title(f"Tracking Error  |  MSE = {mse:.3f}  |  MAE = {mae:.3f}")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Error (N·m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
