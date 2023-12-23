# pylint:disable=arguments-differ,unused-argument
"""
BarcodeRunner Module.

This module contains the `BarcodeRunner` class, a subclass of `pytorch_lightning.LightningModule`, specifically designed
for ocr tasks. The runner handles model initialization, metric computation, optimizer and scheduler
configuration, and the main training, validation, and testing loops.
"""

from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.models.crnn import CRNN
from src.utils.general import load_object
from src.utils.losses import get_losses
from src.utils.metrics import get_metrics


class BarcodeRunner(pl.LightningModule):
    """The main LightningModule for image ocr tasks.

    Attributes:
        config (DictConfig): Configuration object with parameters for model, optimizer, scheduler, etc.
        model (torch.nn.Module): The image ocr model.
        valid_metrics (pytorch_lightning.Metric): Metrics to track during validation.
        test_metrics (pytorch_lightning.Metric): Metrics to track during testing.
        losses (list): List of loss functions.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the BarcodeRunner with a given configuration.

        Args:
            config (DictConfig): Configuration object containing necessary parameters.
        """
        super().__init__()
        self.config = config

        self._init_model()
        self._init_metrics()
        self.losses = get_losses(self.config.losses)

        self.save_hyperparameters()

    def _init_model(self) -> None:
        """Initialize the model."""
        self.model = CRNN(
            encoder=self.config.model.encoder_name,
            num_classes=self.config.model.num_classes,
            rnn_hidden_size=self.config.model.rnn_hidden_size,
            rnn_num_layers=self.config.model.rnn_num_layers,
            rnn_features_num=self.config.model.rnn_features_num,
        )

    def _init_metrics(self) -> None:
        """Initialize metrics for validation and testing."""
        metrics = get_metrics()
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and the learning rate scheduler.
        """
        optimizer = load_object(self.config.optimizer)(
            self.model.parameters(),
            **self.config.optimizer_kwargs,
        )

        scheduler = load_object(self.config.scheduler)(optimizer, **self.config.scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.callbacks.monitor_metric,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            images (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Model's output tensor.
        """
        return self.model(images)

    def _calculate_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        """
        Calculate the total loss and logs individual and total losses.

        Args:
            log_probs (torch.Tensor): Predicted log_probs from the model.
            targets (torch.Tensor): Ground truth labels.
            input_lengths (torch.Tensor): Length of input log_probs.
            target_lengths (torch.Tensor): Length of traget labels.
            prefix (str): Prefix indicating the phase.

        Returns:
            torch.Tensor: Computed total loss.
        """
        ctc_loss = self.losses["CTCLoss"].loss(  # type: ignore
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )
        self.log(f"{prefix}CTCLoss", ctc_loss.item(), sync_dist=True)
        return ctc_loss

    def _process_batch(self, batch, prefix: str) -> Optional[torch.Tensor]:
        """
        Process a batch of images and labels for either training, validation, or testing.

        Args:
            batch (tuple): A tuple containing images and ground truth labels.
            prefix (str): Prefix indicating the phase.

        Returns:
            Optional[torch.Tensor]: Computed total loss for train step.
        """
        images, targets, target_length = batch
        log_probs = self(images)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(0),
            dtype=torch.int,
        )

        if "train" in prefix:
            self.train_metrics(log_probs, targets)
            return self._calculate_loss(log_probs, targets, input_lengths, target_length, prefix)

        self._calculate_loss(log_probs, targets, input_lengths, target_length, prefix)

        if "val" in prefix:
            self.valid_metrics(log_probs, targets)
        elif "test" in prefix:
            self.test_metrics(log_probs, targets)
        return None

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Process a batch during training.

        Args:
            batch (tuple): A tuple containing images and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed Loss

        """
        return self._process_batch(batch, "train_")  # type: ignore

    def validation_step(self, batch, batch_idx) -> None:
        """
        Process a batch during validation.

        Args:
            batch (tuple): A tuple containing images and labels.
            batch_idx (int): Index of the current batch.
        """
        self._process_batch(batch, "val_")

    def test_step(self, batch, batch_idx) -> None:
        """
        Process a batch during testing.

        Args:
            batch (tuple): A tuple containing images and labels.
            batch_idx (int): Index of the current batch.
        """
        self._process_batch(batch, "test_")

    def on_train_epoch_start(self) -> None:
        """Reset the train metrics at the start of a validation epoch."""
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        """Log the computed train metrics at the end of a validation epoch."""
        self.log_dict(self.train_metrics.compute(), on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        """Reset the validation metrics at the start of a validation epoch."""
        self.valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log the computed validation metrics at the end of a validation epoch."""
        self.log_dict(self.valid_metrics.compute(), on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        """Log the computed test metrics at the end of a testing epoch."""
        self.log_dict(self.test_metrics.compute(), on_epoch=True, sync_dist=True)
