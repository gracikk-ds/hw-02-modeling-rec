"""
This module provides functionalities to load and manage losses used in a neural network training pipeline.

Functions:
    get_losses: Creates a list of Loss objects based on a given list of LossConfig configurations.
"""

from typing import Dict, List

from pydantic import BaseModel, ConfigDict
from torch import nn

from src.settings.config import LossConfig
from src.utils.general import load_object


class LossObj(BaseModel):
    """
    Configuration for loss functions.

    Attributes:
        loss (nn.Module): Inited loss module.
        loss_weight (float): Weight of the provided loss.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    loss: nn.Module
    loss_weight: float


def get_losses(loss_cfgs: List[LossConfig]) -> Dict[str, LossObj]:
    """Generate a list of Loss objects based on provided configurations.

    This function uses configurations provided in the form of LossConfig objects
    to create and initialize corresponding PyTorch loss modules.

    Args:
        loss_cfgs (List[LossConfig]): List of loss configurations.

    Returns:
        Dict[str, LossObj]: Dict of initialized Loss objects.
    """
    return {
        loss_cfg.name: LossObj(
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
            loss_weight=loss_cfg.loss_weight,
        )
        for loss_cfg in loss_cfgs
    }
