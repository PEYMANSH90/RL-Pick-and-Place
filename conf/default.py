"""Default experiment configuration.

All fields are plain Python — no YAML parser needed.  To run a variant,
import this module and override individual fields:

    from conf.default import cfg
    cfg.td3.lr_actor = 1e-4
    cfg.trainer.max_episodes = 100

Or pass overrides as CLI flags — see scripts/train.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pick_place.agents.td3 import TD3Config
from pick_place.training.curriculum import CurriculumLevel


@dataclass
class EnvConfig:
    """JointControlEnv construction parameters."""
    gear_ratio:        float = 120.0
    steps_per_episode: int   = 100
    dt:                float = 0.16
    action_max:        float = 2.0
    action_penalty:    float = 0.01


@dataclass
class TrainerConfig:
    """Trainer-level settings (separate from TD3 hyperparameters)."""
    max_episodes:    int  = 200
    log_interval:    int  = 5
    device:          str  = "cpu"
    checkpoint_dir:  Path = Path("checkpoints")
    results_dir:     Path = Path("results")


@dataclass
class CurriculumConfig:
    """Three-stage curriculum matching the paper."""
    levels: list[CurriculumLevel] = field(default_factory=lambda: [
        CurriculumLevel(load_scale=0.10, episodes=30,
                        reward_threshold=-600.0, label="Level-0 (10 %)"),
        CurriculumLevel(load_scale=0.50, episodes=30,
                        reward_threshold=-550.0, label="Level-1 (50 %)"),
        CurriculumLevel(load_scale=1.00, episodes=30,
                        reward_threshold=-493.0, label="Level-2 (100 %)"),
    ])
    reward_window: int = 5


@dataclass
class Config:
    """Top-level experiment config — single object passed to all components."""
    env:        EnvConfig        = field(default_factory=EnvConfig)
    td3:        TD3Config        = field(default_factory=TD3Config)
    trainer:    TrainerConfig    = field(default_factory=TrainerConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    seed:       int              = 42
    run_name:   str              = "default"


# Singleton — scripts import this directly
cfg = Config()
