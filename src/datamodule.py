"""This module defines the BarcodeDataModule class, which is used to prepare and load data."""

import os
from typing import Literal, Type

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info  # type: ignore
from torch.utils.data import DataLoader

from src.dataset.augmentations import get_transforms
from src.dataset.dataset import BarcodeDataset
from src.utils.data import read_df, split_and_save_datasets
from src.utils.general import guess_num_workers

Segment = Literal["train", "val", "test"]


class BarcodeDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for the Barcode dataset.

    This class provides methods for preparing the data, creating datasets, and
    defining data loaders for training, validation, and testing.

    Attributes:
        data_config: Configuration object containing dataset parameters.
        transforms_config: Configuration object containing augs parameters.
    """

    def __init__(self, data_config: DictConfig, transforms_config: DictConfig):
        """
        Initialize the BarcodeDataModule.

        Args:
            data_config (DictConfig): Configuration object containing dataset parameters.
            transforms_config (DictConfig): Configuration object containing augs parameters.
        """
        super().__init__()
        self.config = data_config

        # For training, we keep augmentations enabled
        train_transforms = get_transforms(transforms_config)

        # For validation, we disable augmentations
        valid_config = transforms_config.model_copy()
        valid_config.augmentations = False
        valid_transforms = get_transforms(valid_config)

        self.transforms = {"train": train_transforms, "val": valid_transforms, "test": valid_transforms}

        # Image folder
        self.image_folder = os.path.join(data_config.data_path, "barcodes")

    def prepare_data(self):
        """Prepare and split the datasets for train, validation, and test."""
        split_and_save_datasets(self.config.data_path, self.config.train_size)

    def get_dataset(self, dataset_cl: Type[BarcodeDataset], segment: Segment) -> BarcodeDataset:
        """
        Get the dataset object for the given segment.

        Args:
            dataset_cl (BarcodeDataset): Dataset class.
            segment (Segment): Which segment to fetch (train, val, or test).

        Returns:
            BarcodeDataset: Dataset object for the given segment.
        """
        return dataset_cl(
            dataframe=read_df(self.config.data_path, segment),
            image_folder=self.config.data_path,
            transforms=self.transforms[segment],
        )

    def get_dataloader(self, dataset_cl: Type[BarcodeDataset], segment: Segment) -> DataLoader[BarcodeDataset]:
        """
        Get the data loader for the given segment.

        Args:
            dataset_cl (Type[BarcodeDataset]): Dataset class.
            segment (Segment): Which segment to fetch data loader for (train, val, or test).

        Returns:
            DataLoader: DataLoader object for the given segment.
        """
        dataset = self.get_dataset(dataset_cl, segment)
        num_workers = self.config.num_workers if self.config.num_workers is not None else guess_num_workers()
        rank_zero_info(f"\nUsing {num_workers} workers for {segment}")

        return DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=(segment == "train"),
            shuffle=(segment == "train"),
        )

    def train_dataloader(self) -> DataLoader[BarcodeDataset]:
        """
        Get the data loader for the training set.

        Returns:
            DataLoader[BarcodeDataset]: Data loader for the training set.
        """
        return self.get_dataloader(BarcodeDataset, "train")

    def val_dataloader(self) -> DataLoader[BarcodeDataset]:
        """
        Get the data loader for the validation set.

        Returns:
            DataLoader[BarcodeDataset]: Data loader for the val set.
        """
        return self.get_dataloader(BarcodeDataset, "val")

    def test_dataloader(self) -> DataLoader[BarcodeDataset]:
        """
        Get the data loader for the testing set.

        Returns:
            DataLoader[BarcodeDataset]: Data loader for the testing set.
        """
        return self.get_dataloader(BarcodeDataset, "test")
