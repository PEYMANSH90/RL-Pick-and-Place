#!/usr/bin/env python3
"""Evaluation entry point.

Loads a trained TD3 checkpoint and runs one deterministic episode on the
full-load (100 %) JointControlEnv.  Prints performance metrics and saves
a tracking plot.

Usage
-----
    python scripts/evaluate.py --checkpoint checkpoints/final
    python scripts/evaluate.py --checkpoint checkpoints/level_2 --episodes 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from conf.default import cfg
from pick_place.agents.td3 import TD3Agent
from pick_place.environments.joint_control_env import JointControlEnv
from pick_place.utils.plotting import plot_tracking
from pick_place.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained TD3 agent")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to checkpoint directory (contains td3_checkpoint.pt)")
    p.add_argument("--episodes",    type=int, default=1,
                   help="Number of evaluation episodes")
    p.add_argument("--load-scale",  type=float, default=1.0,
                   help="Load difficulty (1.0 = full paper loads)")
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def run_episode(
    agent: TD3Agent,
    env: JointControlEnv,
) -> dict:
    """Run one deterministic episode and return collected metrics."""
    obs, _ = env.reset()
    done = False

    steps, loads, motor_torques, joint_torques, errors, rewards = (
        [], [], [], [], [], []
    )
    step = 0

    while not done:
        action = agent.select_action(obs, explore=False)
        t_star  = float(action[0])
        t_joint = t_star * env.gear_ratio

        steps.append(step)
        loads.append(env.current_load)
        motor_torques.append(t_star)
        joint_torques.append(t_joint)
        errors.append(env.current_load - t_joint)

        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
        step += 1

    errors_arr = np.array(errors)
    return {
        "steps":         np.array(steps),
        "loads":         np.array(loads),
        "motor_torques": np.array(motor_torques),
        "joint_torques": np.array(joint_torques),
        "errors":        errors_arr,
        "rewards":       np.array(rewards),
        "mse":           float(np.mean(errors_arr ** 2)),
        "mae":           float(np.mean(np.abs(errors_arr))),
        "max_error":     float(np.max(np.abs(errors_arr))),
        "total_reward":  float(np.sum(rewards)),
    }


def main() -> None:
    args = parse_args()
    get_logger(level=logging.INFO)
    logger = logging.getLogger(__name__)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build env and agent
    env = JointControlEnv(load_scale=args.load_scale)
    agent = TD3Agent(
        obs_dim    = env.observation_space.shape[0],
        action_dim = env.action_space.shape[0],
        action_max = float(env.action_space.high[0]),
        cfg        = cfg.td3,
        device     = cfg.trainer.device,
    )
    agent.load(args.checkpoint)
    logger.info("Checkpoint loaded from %s", args.checkpoint)

    all_rewards, all_mses = [], []

    for ep in range(args.episodes):
        metrics = run_episode(agent, env)
        all_rewards.append(metrics["total_reward"])
        all_mses.append(metrics["mse"])

        logger.info(
            "Episode %d/%d | reward=%.2f | MSE=%.4f | MAE=%.4f | max_err=%.4f",
            ep + 1, args.episodes,
            metrics["total_reward"], metrics["mse"],
            metrics["mae"], metrics["max_error"],
        )

        # Save tracking plot for every episode
        plot_path = results_dir / f"tracking_ep{ep+1:03d}.png"
        plot_tracking(
            steps        = metrics["steps"],
            loads        = metrics["loads"],
            joint_torques= metrics["joint_torques"],
            errors       = metrics["errors"],
            save_path    = plot_path,
        )
        logger.info("Tracking plot saved → %s", plot_path)

    # Summary across all episodes
    logger.info("=" * 50)
    logger.info("Mean reward : %.2f ± %.2f",
                np.mean(all_rewards), np.std(all_rewards))
    logger.info("Mean MSE    : %.4f ± %.4f",
                np.mean(all_mses), np.std(all_mses))


if __name__ == "__main__":
    main()
