"""Training loop and curriculum learning for the TD3 agent."""

from pick_place.training.trainer import Trainer
from pick_place.training.curriculum import CurriculumScheduler

__all__ = ["Trainer", "CurriculumScheduler"]
