"""Module provided Cosine Annealing with Warmup LR Scheduler."""
from math import cos, pi
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmup(_LRScheduler):
    """Cosine Annealing with Warmup LR Scheduler.

    This LR scheduler combines a warm-up phase with a cosine annealing decay. It gradually increases the learning rate
    during the warm-up phase and then smoothly decays it using a cosine function.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        decay_steps: int = 10000,
        last_epoch: int = -1,
    ):
        """Initialize the CosineAnnealingWarmup LR scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The PyTorch optimizer for which to adjust the learning rate.
            min_lr (float): The minimum learning rate to be used during the cosine annealing phase. Defaults to 0.001.
            warmup_steps (int): The number of warm-up steps during which the learning rate increases linearly. D-s to 0.
            decay_steps (int): The total number of steps for the cosine annealing decay. Defaults to 10,000.
            last_epoch (int): The index of the last epoch. If not specified, it will be set to -1.
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        """Get the current learning rates for all parameter groups.

        Returns:
            List[float]: A list of learning rates for each parameter group.
        """
        warmup_corrected_epoch_state = self.last_epoch - self.warmup_steps

        if warmup_corrected_epoch_state < 0:  # check current epoch number adjusted by warmup_steps
            mult = self.last_epoch / self.warmup_steps  # warmup LR multiplier
        else:
            mult = 0.5 * (1 + cos(pi * (warmup_corrected_epoch_state) / self.decay_steps))  # starts cosine decay
        min_lr_coefs = [lr / self.base_lrs[0] for lr in self.base_lrs]
        min_lrs = [self.min_lr * min_lr_coef for min_lr_coef in min_lr_coefs]
        return [min_lr + (base_lr - min_lr) * mult for base_lr, min_lr in zip(self.base_lrs, min_lrs)]
