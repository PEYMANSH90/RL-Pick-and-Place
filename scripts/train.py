#!/usr/bin/env python3
"""Training entry point.

Runs curriculum TD3 training on the JointControlEnv and saves a checkpoint
after each curriculum level and at completion.

Usage
-----
# Default config (paper hyperparameters):
    python scripts/train.py

# Override individual fields:
    python scripts/train.py --max-episodes 100 --device cuda --seed 7

# Minimal run for a quick smoke-test:
    python scripts/train.py --max-episodes 10 --log-interval 1
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from conf.default import cfg
from pick_place.training.trainer import Trainer
from pick_place.utils.plotting import plot_training_curve
from pick_place.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TD3 on JointControlEnv")
    p.add_argument("--max-episodes",  type=int,   default=None)
    p.add_argument("--log-interval",  type=int,   default=None)
    p.add_argument("--device",        type=str,   default=None,
                   choices=["cpu", "cuda"])
    p.add_argument("--seed",          type=int,   default=None)
    p.add_argument("--checkpoint-dir",type=str,   default=None)
    p.add_argument("--results-dir",   type=str,   default=None)
    p.add_argument("--lr",            type=float, default=None,
                   help="Learning rate for actor and critic")
    p.add_argument("--run-name",      type=str,   default=None)
    return p.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    """Patch the global cfg singleton with any CLI overrides provided."""
    if args.max_episodes  is not None: cfg.trainer.max_episodes  = args.max_episodes
    if args.log_interval  is not None: cfg.trainer.log_interval  = args.log_interval
    if args.device        is not None: cfg.trainer.device        = args.device
    if args.checkpoint_dir is not None:
        cfg.trainer.checkpoint_dir = Path(args.checkpoint_dir)
    if args.results_dir   is not None:
        cfg.trainer.results_dir   = Path(args.results_dir)
    if args.seed          is not None: cfg.seed                  = args.seed
    if args.lr            is not None:
        cfg.td3.lr_actor  = args.lr
        cfg.td3.lr_critic = args.lr
    if args.run_name      is not None: cfg.run_name              = args.run_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    apply_overrides(args)

    # Logging — INFO to stdout, DEBUG available via env var LOG_LEVEL=DEBUG
    get_logger(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Run: %s | seed=%d | device=%s",
                cfg.run_name, cfg.seed, cfg.trainer.device)

    set_seed(cfg.seed)

    # Output directories
    cfg.trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.trainer.results_dir.mkdir(parents=True, exist_ok=True)

    # Build and run trainer
    trainer = Trainer(
        td3_cfg           = cfg.td3,
        curriculum_levels = cfg.curriculum.levels,
        checkpoint_dir    = cfg.trainer.checkpoint_dir,
        max_episodes      = cfg.trainer.max_episodes,
        log_interval      = cfg.trainer.log_interval,
        device            = cfg.trainer.device,
    )
    result = trainer.train()

    # Summary
    logger.info("Best episode reward : %.2f", result.best_reward)
    logger.info("Total episodes      : %d",   result.n_episodes)
    logger.info(
        "Level boundaries    : %s",
        result.level_boundaries if result.level_boundaries else "none",
    )

    # Plot and save training curve
    plot_path = cfg.trainer.results_dir / f"{cfg.run_name}_training_curve.png"
    plot_training_curve(
        episode_rewards   = result.episode_rewards,
        level_boundaries  = result.level_boundaries,
        save_path         = plot_path,
    )
    logger.info("Training curve saved → %s", plot_path)


if __name__ == "__main__":
    main()
