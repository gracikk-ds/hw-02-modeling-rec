"""Convert torch to torchscript."""
import os

import click
import torch
from omegaconf import DictConfig

from src.models.crnn import CRNN
from src.settings.config import Config


def extract_value_from_string(string: str) -> float:
    """
    Extract the metric value from a checkpoint path.

    Args:
        string: checkpoint file name.

    Returns:
        float: returns the numeric value after the '=' and before the '.ckpt'.
    """
    # Splitting the string to get the part after the last '='
    last_part = string.split("=")[-1]
    # Extracting the numeric part before '.ckpt'
    numeric_value = last_part.split(".ckpt")[0]
    return float(numeric_value)


def get_checkpoint(checkpoint_path: str) -> str:
    """
    Get the checkpoint with the highest metric value.

    Args:
        checkpoint_path: path to the directory containing the checkpoints or to checkpoint itself.

    Returns:
        str: path to the checkpoint with the highest metric value.
    """
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    # Getting the list of all the checkpoints
    checkpoints = os.listdir(checkpoint_path)
    # Getting the metric values from the checkpoint names
    metric_values = [extract_value_from_string(checkpoint) for checkpoint in checkpoints]
    # Getting the index of the checkpoint with the highest metric value
    max_index = metric_values.index(max(metric_values))
    return os.path.join(checkpoint_path, checkpoints[max_index])


def load_model_weights(model, checkpoint_path: str):
    """Load model weights.

    Args:
        model: pytorch model class.
        checkpoint_path (str): path to model .ckpt checkpoits.

    Raises:
        KeyError: if no state dict found.

    Returns:
        HighlightDetector: model with updated state_dict
    """
    # Load the checkpoint dictionary
    checkpoint_path = get_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Check if 'state_dict' is in the checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint.get("state_dict")
    else:
        raise KeyError("No state_dict found in the checkpoint file")

    # Adjust keys to remove module. prefix if it exists
    # This is necessary if the original model was saved in an nn.DataParallel wrapper
    new_state_dict = {}
    for key, value in state_dict.items():
        name = key[6:] if key.startswith("model.") else key  # remove `module.` prefix if present
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict, strict=True)

    return model


@click.command()
@click.option("--config_path", type=str, default="configs/config.yaml")
@click.option("--checkpoints_path", type=str, default="experiments/")
@click.option("--output_path", type=str, default="weights/seg_model.pt")
@click.option("--batch_size", type=int)
@click.option("--channels", type=int)
@click.option("--img_dim", type=int)
def convert_torch_to_scripted(
    config_path: str,
    checkpoints_path: str,
    output_path: str,
    batch_size: int,
    channels: int,
    img_dim: int,
):
    """Convert torch to scripted module.

    Args:
        config_path (str): path to cfg file.
        checkpoints_path (str): path to model checkpoint's dir or checkpoint itself.
        output_path (str): path to save model.
        batch_size (int): batch size for model input.
        channels (int): number of channels for model input.
        img_dim (int): image dimension for model input.

    Raises:
        RuntimeError: outputs of torch and compiled models are mismatch.
    """
    config: DictConfig = Config.from_yaml(config_path)
    model = CRNN(
        encoder=config.model.encoder_name,
        num_classes=config.model.num_classes,
        rnn_hidden_size=config.model.rnn_hidden_size,
        rnn_num_layers=config.model.rnn_num_layers,
        rnn_features_num=config.model.rnn_features_num,
    )

    model.eval()
    model = load_model_weights(model, checkpoints_path)
    scripted_model = torch.jit.script(model)  # type: ignore

    check_tensor = torch.randn(batch_size, channels, img_dim, img_dim)

    with torch.no_grad():
        torch_output = model(check_tensor)
        scripted_output = scripted_model(check_tensor)

    if not torch.allclose(torch_output, scripted_output):
        raise RuntimeError("torch_output != scripted_output")

    torch.jit.save(scripted_model, output_path)  # type: ignore


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    convert_torch_to_scripted()
