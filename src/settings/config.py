"""
This module provides configuration models for training and data transformation setups.

It contains models for:
- Loss configurations, defining different losses and their properties.
- Data transformation configurations, detailing various augmentation and preprocessing steps.
- Data loading configurations, specifying dataset paths and related properties.
- The main configuration model, bringing together all the aforementioned configurations for a cohesive training setup.
"""

import string
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class GeneralSettings(BaseModel):
    """
    General train settings.

    Attributes:
        project_name (str): Name of the project.
        experiment_name (str): Name of the experiment.
        num_sanity_val_steps (int): Number of validation steps for sanity check (e.g., 1).
        max_steps (int): Number of training steps.
        seed (int): Random seed.
        dotenv_path (str): dotenv path.
    """

    project_name: str
    experiment_name: str
    num_sanity_val_steps: int
    max_steps: int
    seed: int
    dotenv_path: str = ".env"


class ModelConfig(BaseModel):
    """
    Configuration for Highlight Detector.

    Attributes:
        encoder_name (str): Name of the encoder to use.
        num_classes (int): Number of classes.
        rnn_features_dim (int): Input rnn features.
        rnn_features_num (int): Input rnn features.
        rnn_hidden_size (int): hidden size of the rnn model.
        rnn_num_layers (int): number of lstm layers to use.
    """

    encoder_name: str
    num_classes: int
    rnn_features_dim: int
    rnn_features_num: int
    rnn_hidden_size: int
    rnn_num_layers: int


class HardwareConfiguration(BaseModel):
    """Hardware configuration settings.

    Attributes:
        accelerator (str): Type of accelerator (e.g., "CPU", "GPU").
        devices (Union[List[int], int, str]): Which device to use if there many. Or just use "auto".
        precision (int): Training precision.
    """

    accelerator: str
    devices: Union[List[int], int, str]
    precision: int = 32


class CallbacksConfiguration(BaseModel):
    """Callbacks Configuration settings.

    Attributes:
        monitor_metric (str): Metric to monitor during training.
        monitor_mode (str): Mode for monitoring the metric (e.g., "min", "max").
        early_stopping_patience (int): early stopping patience steps.
        progress_bar_refresh_rate (int): progress bar refresh rate for Lightning callback.
    """

    monitor_metric: str
    monitor_mode: str
    early_stopping_patience: int
    progress_bar_refresh_rate: int


class LossConfig(BaseModel):
    """
    Configuration for loss functions.

    Attributes:
        name (str): Name of the loss.
        loss_weight (float): Weight of the loss.
        loss_fn (str): Loss function.
        loss_kwargs (Dict[str, Any]): Additional keyword arguments for the loss function.
    """

    name: str
    loss_weight: float
    loss_fn: str
    loss_kwargs: Dict[str, Any]


class TransformsConfig(BaseModel):
    """
    Configuration for data transformations.

    Attributes:
        preprocessing (bool): Whether to apply preprocessing.
        augmentations (bool): Whether to apply augmentations.
        vocab (str): Model vocab.
        text_size (int): Target text size.
        img_width (int): Target img width.
        img_height (int): Target img height.
        crop_persp_prob (float): Probability of applying a perspective transformation through cropping to the images.
        scalex_prob (float): Probability of scaling the images along the x-axis.
        rbc_prob (float): Probability of applying random brightness and contrast adjustments.
        clahe_prob (float): Probability of applying CLAHE for contrast enhancement.
        blur_limit (float): The maximum extent to which the blur is applied.
        blur_prob (float): Probability of applying a blur effect to the images.
        noise_prob (float): Probability of adding Gaussian noise to the images.
        downscale_min (float): The range for downscaling the image resolution.
        downscale_max (float): The range for downscaling the image resolution.
        downscale_prob (float): Probability of applying the downscaling.
        max_holes (int): The range for the number of holes in the CoarseDropout augmentation.
        min_holes (int): The range for the number of holes in the CoarseDropout augmentation.
        coarse_prob (float): Probability of applying CoarseDropout, which randomly removes regions of the image.
    """

    preprocessing: bool = True
    augmentations: bool = True
    text_size: int = 13
    vocab: str = string.digits
    img_width: int = 416
    img_height: int = 96
    crop_persp_prob: float
    scalex_prob: float
    rbc_prob: float
    clahe_prob: float
    blur_limit: float
    blur_prob: float
    noise_prob: float
    downscale_min: float
    downscale_max: float
    downscale_prob: float
    max_holes: int
    min_holes: int
    coarse_prob: float


class DataConfig(BaseModel):
    """
    Configuration for data loading.

    Attributes:
        data_path (str): Path to the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        train_size (float): Proportion of the dataset used for training.
        seed (int): Random seed.
    """

    data_path: str
    batch_size: int
    num_workers: Optional[int]
    train_size: float
    seed: int


class Config(BaseModel):
    """
    Main configuration class for the project.

    Attributes:
        general (str): General project settings.
        hardware (HardwareConfiguration): Hardware configuration settings.
        callbacks (CallbacksConfiguration): Callbacks configuration settings.
        base_data_settings (DataConfig): Data loading configuration.
        transforms_settings (TransformsConfig): Data transformation configuration.
        model (ModelConfig): Model configuration.
        losses (List[LossConfig]): List of loss configurations.
        optimizer (str): Optimizer for training.
        optimizer_kwargs (Dict[str, Any]): Additional keyword arguments for the optimizer.
        scheduler (str): Learning rate scheduler.
        scheduler_kwargs (Dict[str, Any]): Additional keyword arguments for the scheduler.

    """

    general: GeneralSettings
    hardware: HardwareConfiguration
    callbacks: CallbacksConfiguration
    base_data_settings: DataConfig
    transforms_settings: TransformsConfig
    model: ModelConfig
    losses: List[LossConfig]
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    scheduler: str
    scheduler_kwargs: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> DictConfig:
        """
        Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML file.

        Returns:
            DictConfig: An instance of the Config class.
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)  # type: ignore
