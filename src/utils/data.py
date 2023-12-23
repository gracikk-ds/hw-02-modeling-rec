"""Utility module for reading dataframes from a specified path based on a given mode."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_VAL_PROPORTION: float = 0.5
RANDOM_STATE: int = 42


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """
    Read the dataframe for the given mode.

    Args:
        data_path (str): Path to the data folder.
        mode (str): Mode specifying which data to read (train, val, or test).

    Returns:
        pd.DataFrame: Dataframe containing data for the specified mode.
    """
    return pd.read_csv(os.path.join(data_path, f"{mode}_data.csv"))


def save_df(dataframe: pd.DataFrame, data_path: str, mode: str) -> None:
    """
    Save the given dataframe to a CSV file.

    Args:
        dataframe (pd.DataFrame): Dataframe to save.
        data_path (str): Path to the data folder.
        mode (str): Mode specifying which data to save (train, val, or test).
    """
    path_to_save = os.path.join(data_path, f"{mode}_data.csv")
    dataframe.to_csv(path_to_save, index=False)


def split_and_save_datasets(data_path: str, train_size: float) -> None:
    """
    Split dataset from given CSV file into training, validation, and test datasets.

    Parameters:
        data_path (str): The path to the input CSV file.
        train_size (float): Proportion of the dataset to include in the training split (0.0 to 1.0).
    """
    dataframe = pd.read_csv(os.path.join(data_path, "annotations_updated.tsv"), sep="\t")

    train_data, temp_data = train_test_split(dataframe, train_size=train_size, random_state=RANDOM_STATE)
    val_data, test_data = train_test_split(temp_data, test_size=TEST_VAL_PROPORTION, random_state=RANDOM_STATE)

    save_df(train_data, data_path, "train")
    save_df(val_data, data_path, "val")
    save_df(test_data, data_path, "test")
    save_df(test_data, data_path, "test")
