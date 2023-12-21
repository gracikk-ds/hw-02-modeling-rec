"""This module provides image transformation utilities using Albumentations."""

from typing import Union

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

TransformType = Union[albu.BasicTransform, albu.BaseCompose]


def get_transforms(cfg: DictConfig) -> TransformType:
    """
    Create a composition of image transformations for data augmentation and preprocessing.

    Args:
        cfg (DictConfig): Hydra configuration containing transformation parameters.

    Returns:
        TransformType: An Albumentations composition of image transformations.
    """
    preprocessing = cfg.preprocessing
    augmentations = cfg.augmentations

    transforms = []

    if preprocessing:
        transforms.extend(
            [
                albu.LongestMaxSize(max_size=cfg.max_size, always_apply=True, p=1.0),
                albu.PadIfNeeded(min_height=cfg.max_size, min_width=cfg.max_size, always_apply=True, p=1),
            ],
        )

    if augmentations:
        transforms.extend(
            [
                albu.HueSaturationValue(
                    hue_shift_limit=cfg.hue_shift_limit,
                    sat_shift_limit=cfg.sat_shift_limit,
                    val_shift_limit=cfg.val_shift_limit,
                ),
                albu.RandomBrightnessContrast(
                    brightness_limit=cfg.brightness_limit,
                    contrast_limit=cfg.contrast_limit,
                ),
                albu.ShiftScaleRotate(),
                albu.GaussianBlur(),
            ],
        )

    transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)
