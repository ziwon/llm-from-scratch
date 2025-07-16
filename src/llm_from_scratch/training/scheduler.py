"""Learning rate scheduler utilities."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..config import SchedulerConfig, TrainingConfig


class WarmupLRScheduler(_LRScheduler):
    """Learning rate scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get learning rate."""
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            factor = self._step_count / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        # After warmup - override in subclasses
        return self.base_lrs


class CosineWarmupScheduler(WarmupLRScheduler):
    """Cosine annealing with linear warmup."""

    def get_lr(self):
        """Get learning rate with cosine annealing after warmup."""
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            factor = self._step_count / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        # Cosine annealing
        progress = (self._step_count - self.warmup_steps) / (
            self.max_steps - self.warmup_steps
        )
        factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs
        ]


class LinearWarmupScheduler(WarmupLRScheduler):
    """Linear decay with linear warmup."""

    def get_lr(self):
        """Get learning rate with linear decay after warmup."""
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            factor = self._step_count / self.warmup_steps
            return [base_lr * factor for base_lr in self.base_lrs]
        # Linear decay
        progress = (self._step_count - self.warmup_steps) / (
            self.max_steps - self.warmup_steps
        )
        factor = 1 - progress
        return [
            self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs
        ]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig,
    training_config: TrainingConfig,
    num_training_steps: int,
) -> _LRScheduler | None:
    """
    Create learning rate scheduler from config.

    Args:
        optimizer: Optimizer instance
        scheduler_config: Scheduler configuration
        training_config: Training configuration
        num_training_steps: Total number of training steps

    Returns:
        Scheduler instance or None
    """
    scheduler_type = scheduler_config.type.lower()

    if scheduler_type == "cosine":
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=training_config.warmup_steps,
            max_steps=num_training_steps,
            min_lr=scheduler_config.min_lr,
        )
    elif scheduler_type == "linear":
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=training_config.warmup_steps,
            max_steps=num_training_steps,
            min_lr=scheduler_config.min_lr,
        )
    elif scheduler_type == "constant":
        # No scheduler - constant learning rate
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler
