"""Module to define a custom dataset for the Barcode segmentation task."""

import os
from typing import Tuple, Union

import albumentations as albu
import cv2
import jpeg4py as jpeg
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

TransformType = Union[albu.BasicTransform, albu.BaseCompose]
DataAnnotation = Union[NDArray[np.uint8], NDArray[np.float32]]


class BarcodeDataset(Dataset):  # type: ignore
    """
    Custom dataset for the Barcode segmaentation task.

    Attributes:
        dataframe (pd.DataFrame): The dataset's metadata including image IDs and labels.
        image_folder (str): Path to the folder containing the images.
        transforms (TransformType): Albumentations transformations to be applied on the images.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_folder: str,
        transforms: TransformType,
    ) -> None:
        """
        Initialize a new instance of BarcodeDataset.

        Args:
            dataframe (pd.DataFrame): Dataset's metadata.
            image_folder (str): Path to the folder containing the images.
            transforms (TransformType): Albumentations transformations to apply on the images.
        """
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transforms = transforms

    @staticmethod
    def load_image(image_path: str) -> NDArray[np.uint8]:
        """Load an image from the given path.

        Args:
            image_path (str): Path to the image.

        Returns:
            NDArray[np.uint8]: The loaded image.
        """
        try:
            image: NDArray[np.uint8] = jpeg.JPEG(image_path).decode()
        except RuntimeError:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Fetch the image and its label based on the provided index.

        Args:
            idx (int): Index of the desired dataset item.

        Returns:
            tuple: A tuple containing the RGB image (np.ndarray) and its labels (np.ndarray).
        """
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_folder, row["filename"]).replace("images", "barcodes")
        image = self.load_image(image_path)
        transformed_image = self.transforms(image=image)
        return transformed_image, row["code"]

    def __len__(self) -> int:
        """
        Return the total number of items in the dataset.

        Returns:
            int: Total number of items.
        """
        return len(self.dataframe)
