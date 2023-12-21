"""
Command-line script for training models using PyTorch Lightning.

This module facilitates training by setting up logging, callbacks, and hardware configurations based on
a provided configuration. It integrates with ClearML for experiment tracking and utilizes various PyTorch
Lightning callbacks for functionalities like early stopping, learning rate monitoring, and progress bar display.

Main Functions:
    - hardware_trainer_params: Setup training hardware parameters based on the configuration.
    - train: Train and test a model using PyTorch Lightning.
    - main: Main entry point, handling command-line arguments and initiating training.

Example usage:
    python src/cli/train.py --config /path/to/config --is_local_run

Note:
    Ensure the necessary configurations and settings are in place before executing the script.
"""

import os
from typing import Any, Dict, List

import click
import pytorch_lightning as pl
import torch
from clearml import Task
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from src.datamodule import BarcodeDataModule
from src.lightning_module import BarcodeRunner
from src.settings.config import Config
from src.utils.reproducibility import git_status

SEED: int = 157
PROJECT_PATH: str = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."),
)
EXPERIMENTS_PATH: str = os.path.join(PROJECT_PATH, "experiments")
MAX_PRECISION: int = 32


def setup_clearml_task(config: DictConfig) -> Task:
    """
    Initialize a logger task using given configuration.

    Args:
        config (DictConfig): The configuration object with project and experiment names.

    Returns:
        Task: An initialized Task object.
    """
    task = Task.init(
        project_name=config.general.project_name,
        task_name=config.general.experiment_name,
        auto_connect_frameworks={
            "tensorboard": {"report_hparams": False},
            "pytorch": "*.ckpt",
            "detect_repository": True,
            "jsonargparse": True,
        },
    )
    task.connect(config.model_dump())
    return task


def setup_callback(config: DictConfig) -> List[Callback]:
    """
    Set up the necessary callbacks based on the configuration.

    Args:
        config (DictConfig): Configuration object containing the experiment details.

    Returns:
        List[Callback]: List of initialized callbacks.
    """
    experiment_save_path = os.path.join(EXPERIMENTS_PATH, config.general.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_save_path,
        filename=f"epoch_{{epoch:02d}}-{{{config.callbacks.monitor_metric}:.3f}}",
        monitor=config.callbacks.monitor_metric,
        save_top_k=5,
        mode=config.callbacks.monitor_mode,
        save_weights_only=False,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    progress_bar = TQDMProgressBar(refresh_rate=config.callbacks.progress_bar_refresh_rate)
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.monitor_metric,
        patience=config.callbacks.early_stopping_patience,
        mode=config.callbacks.monitor_mode,
    )

    return [checkpoint_callback, early_stopping_callback, lr_monitor_callback, progress_bar]


def hardware_trainer_params(config: DictConfig) -> Dict[str, Any]:
    """
    Construct trainer parameters based on the provided configuration and available hardware.

    Args:
        config (DictConfig): Configuration object containing training parameters.

    Returns:
        Dict[str, Any]: Dictionary of trainer parameters.
    """
    callbacks = setup_callback(config)
    use_cuda = torch.cuda.is_available() and config.hardware.accelerator == "gpu"
    trainer_params = {
        "accelerator": "gpu" if use_cuda else "cpu",
        "devices": config.hardware.devices,
        "benchmark": False,
        "max_steps": config.general.max_steps,
        "num_sanity_val_steps": 5,
        "precision": config.hardware.precision if use_cuda else MAX_PRECISION,
        "callbacks": callbacks,
        "log_every_n_steps": 1,
    }

    if use_cuda and torch.cuda.device_count() > 1 and not isinstance(config.hardware.devices, (int, str)):
        trainer_params["strategy"] = "ddp_find_unused_parameters_false"

    return trainer_params


def train(
    config: DictConfig,
    trainer_params: Dict[str, Any],
    is_local_run: bool = False,
):
    """
    Train and test a model based on the provided configuration and trainer parameters.

    Args:
        config (DictConfig): Configuration object containing details for the data module and model runner.
        trainer_params (Dict[str, Any]): Dictionary of trainer parameters to customize the training process.
        is_local_run (bool): Indicator whether the training is being run locally.

    Note:
        The function will set global seeds for reproducibility and will also run the test phase on
        the best checkpoint after the training phase is complete.
    """
    # Set reproducibility
    pl.seed_everything(SEED, workers=True)

    datamodule = BarcodeDataModule(config.base_data_settings, config.transforms_settings)
    model = BarcodeRunner(config)

    if not is_local_run:
        load_dotenv(config.general.dotenv_path)
        setup_clearml_task(config)

    trainer = Trainer(**trainer_params)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")


def inform_and_wait_for_key(message: str, prompt: str = "Press any key to continue..."):
    """
    Display a message to the user and wait for a key press to continue.

    Args:
        message (str): The primary message to display to the user.
        prompt (str): A prompt message indicating the user should press a key.
    """
    print(message)
    input(prompt)


@click.command()
@click.option("--config_path", type=str, required=True)
@click.option("--is_local_run", is_flag=True)
def main(config_path: str, is_local_run: bool) -> None:
    """
    Initialize training based on the provided configuration and git status.

    Args:
        config_path (str): Configuration object containing details for training.
        is_local_run (bool): Indicator whether the training is being run locally. Defaults to False.

    Note:
        The function checks the current git status to determine if the current branch is up-to-date.
        If the branch is not updated, it will disable remote experiment registration and prompt the user.
    """
    config: DictConfig = Config.from_yaml(config_path)
    trainer_params = hardware_trainer_params(config)

    commit = git_status()
    if not is_local_run and (commit is None):
        is_local_run = True
        inform_and_wait_for_key("The branch is not up to date with remote. Disabling remote experiment registration.")

    train(config=config, trainer_params=trainer_params, is_local_run=is_local_run)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
