"""This module provides image transformation utilities using Albumentations."""

from typing import Any, Dict, List, Union

import albumentations as albu
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from numpy import random
from omegaconf import DictConfig

TransformType = Union[albu.BasicTransform, albu.BaseCompose]


class TextEncode:
    """A class for encoding text using a specified vocabulary."""

    def __init__(self, vocab: Union[str, List[str]], target_text_size: int) -> None:
        """
        Initialize the TextEncode object.

        Args:
            vocab (Union[str, List[str]]): A string or a list of strings representing the vocabulary.
            target_text_size (int): The desired fixed size of the encoded text.
        """
        self.vocab: List[str] = vocab if isinstance(vocab, list) else list(vocab)
        self.target_text_size: int = target_text_size

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Encode the text provided in kwargs using the vocabulary and target text size.

        Args:
            kwargs: A dictionary containing the 'text' key with string value to be encoded.

        Returns:
            Dict[str, Any]: The modified kwargs dictionary with the 'text' key updated.
        """
        source_text = kwargs["text"].strip()
        postprocessed_text = [self.vocab.index(digit) + 1 for digit in source_text if digit in self.vocab]
        padding_length = (0, self.target_text_size - len(postprocessed_text))
        postprocessed_array = np.pad(postprocessed_text, padding_length, mode="constant")
        encoded_text = torch.tensor(postprocessed_array, dtype=torch.int32)
        kwargs["text"] = encoded_text
        return kwargs


class PadResizeOCR:
    """
    A class to resize and pad an image for OCR (Optical Character Recognition) processing.

    This class resizes a given image to a target height while maintaining aspect ratio, and then
    pads it to a target width using a specified padding mode.

    Attributes:
        target_width (int): The target width to which the image will be padded.
        target_height (int): The target height to which the image will be resized.
        value (int): The pixel value used for padding. Default is 0 (black).
    """

    def __init__(self, target_width: int, target_height: int, value: int = 0, mode: str = "random"):
        """
        Initialize the PadResizeOCR object.

        Args:
            target_width (int): The desired width of the image after padding.
            target_height (int): The desired height of the image after resizing.
            value (int): The padding pixel value. Defaults to 0 (black).
            mode (str): The padding mode ('random', 'left', 'center'). Defaults to 'random'.

        Raises:
            AssertionError: If the mode is not one of 'random', 'left', or 'center'.
        """
        self.target_width = target_width
        self.target_height = target_height
        self.value = value
        self.mode = mode

        assert self.mode in {"random", "left", "center"}

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Process the image for OCR by resizing and padding according to the initialized parameters.

        Args:
            kwargs: A dictionary containing the key 'image' with an image array to be processed.

        Returns:
            Dict[str, Any]: The modified kwargs dictionary with the processed 'image' key.
        """
        image = kwargs["image"].copy()

        heigth, width, _ = image.shape

        tmp_w = min(int(width * (self.target_height / heigth)), self.target_width)
        image = cv2.resize(image, (tmp_w, self.target_height))

        dw = np.round(self.target_width - tmp_w).astype(int)
        if dw > 0:
            if self.mode == "random":
                pad_left = np.random.randint(dw)
            elif self.mode == "left":
                pad_left = 0
            else:
                pad_left = dw // 2
            pad_right = dw - pad_left
            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        kwargs["image"] = image
        return kwargs


class CropPerspective:
    """
    A class to apply random perspective cropping to an image with a specified probability.

    This class randomly modifies the corners of an image to simulate a perspective change and
    then applies a perspective transformation based on these modifications.

    Attributes:
        p (float): Probability of applying the perspective transformation.
        width_ratio (float): Maximum ratio of width to be used for corner displacement.
        height_ratio (float): Maximum ratio of height to be used for corner displacement.
    """

    def __init__(self, p: float = 0.5, width_ratio: float = 0.04, height_ratio: float = 0.08):
        """
        Initialize the CropPerspective object with transformation parameters.

        Args:
            p (float): Probability of applying the perspective transformation. Defaults to 0.5.
            width_ratio (float): Maximum ratio of width for corner displacement. Defaults to 0.04.
            height_ratio (float): Maximum ratio of height for corner displacement. Defaults to 0.08.
        """
        self.p = p
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Apply a random perspective transformation to the image if the random condition is met.

        Args:
            kwargs: A dictionary containing the key 'image' with an image array to be processed.

        Returns:
            Dict[str, Any]: The modified kwargs dictionary with the processed 'image' key.
        """
        image = kwargs["image"].copy()

        if random.random() < self.p:
            height, width, _ = image.shape

            pts1 = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)
            dw = width * self.width_ratio
            dh = height * self.height_ratio
            pts2 = np.array(
                [
                    [random.uniform(-dw, dw), random.uniform(-dh, dh)],
                    [random.uniform(-dw, dw), height - random.uniform(-dh, dh)],
                    [width - random.uniform(-dw, dw), height - random.uniform(-dh, dh)],
                    [width - random.uniform(-dw, dw), random.uniform(-dh, dh)],
                ],
                dtype=np.float32,
            )

            matrix = cv2.getPerspectiveTransform(pts2, pts1)
            dst_w = (pts2[3][0] + pts2[2][0] - pts2[1][0] - pts2[0][0]) * 0.5
            dst_h = (pts2[2][1] + pts2[1][1] - pts2[3][1] - pts2[0][1]) * 0.5
            image = cv2.warpPerspective(
                image,
                matrix,
                (int(dst_w), int(dst_h)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        kwargs["image"] = image
        return kwargs


class ScaleX:
    """
    A class to apply horizontal scaling to an image with a certain probability.

    This class scales the width of a given image by a random factor within a specified range,
    with a certain probability.

    Attributes:
        p (float): The probability of applying the scaling.
        scale_min (float): The minimum scaling factor.
        scale_max (float): The maximum scaling factor.
    """

    def __init__(self, p: float = 0.5, scale_min: float = 0.8, scale_max: float = 1.2):
        """
        Initialize the ScaleX object with scaling parameters.

        Args:
            p (float): Probability of applying the scaling. Defaults to 0.5.
            scale_min (float): Minimum scaling factor. Defaults to 0.8.
            scale_max (float): Maximum scaling factor. Defaults to 1.2.
        """
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Apply horizontal scaling to the image if the random condition is met.

        Args:
            kwargs: A dictionary containing the key 'image' with an image array to be processed.

        Returns:
            Dict[str, Any]: The modified kwargs dictionary with the processed 'image' key.
        """
        image = kwargs["image"].copy()

        if random.random() < self.p:
            heigth, width, _ = image.shape
            width = int(width * random.uniform(self.scale_min, self.scale_max))
            image = cv2.resize(image, (width, heigth), interpolation=cv2.INTER_LINEAR)

        kwargs["image"] = image
        return kwargs


def get_transforms(cfg: DictConfig) -> TransformType:
    """
    Create a composition of image and text transformations for data augmentation and preprocessing.

    Args:
        cfg (DictConfig): Hydra configuration containing transformation parameters.

    Returns:
        TransformType: An Albumentations composition of image transformations.
    """
    preprocessing = cfg.preprocessing
    augmentations = cfg.augmentations

    transforms = []

    if augmentations:
        transforms.extend(
            [
                CropPerspective(),
                ScaleX(),
            ],
        )

    if preprocessing:
        transforms.append(
            PadResizeOCR(
                target_height=cfg.img_height,
                target_width=cfg.img_width,
                mode="random" if augmentations else "left",
            ),
        )

    if augmentations:
        transforms.extend(
            [
                albu.GaussianBlur(),
                albu.CLAHE(),
                albu.HueSaturationValue(
                    hue_shift_limit=cfg.hue_shift_limit,
                    sat_shift_limit=cfg.sat_shift_limit,
                    val_shift_limit=cfg.val_shift_limit,
                ),
                albu.RandomBrightnessContrast(
                    brightness_limit=cfg.brightness_limit,
                    contrast_limit=cfg.contrast_limit,
                ),
                albu.Downscale(scale_min=cfg.downscale_min, scale_max=cfg.downscale_max),
            ],
        )

    transforms.extend(
        [
            albu.Normalize(),
            TextEncode(vocab=cfg.vocab, target_text_size=cfg.text_size),
            ToTensorV2(),
        ],
    )

    return albu.Compose(transforms)
