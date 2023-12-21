"""General utility functions."""
import importlib
import os
from typing import Any

import torch


def guess_num_workers() -> int:
    """
    Guesses the number of workers based on the available CPU count and distributed training settings.

    Returns:
        int: The estimated number of workers. If distributed training is enabled, the number of workers is
            determined by dividing the CPU count by the world size. Otherwise, the number of CPUs is returned.
    """
    num_cpus = os.cpu_count()
    if num_cpus is None:
        return 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return num_cpus // torch.distributed.get_world_size()
    return num_cpus


def load_object(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Load a Python object from a specified path.

    The `obj_path` parameter is a dot-separated string indicating the import path
    to the object. If the `obj_path` only contains the object's name and not its
    module path, the `default_obj_path` will be used as the module path.

    Args:
        obj_path (str): The path to the object to be loaded.
        default_obj_path (str): The default module path to use if only an object name is provided in `obj_path`.

    Returns:
        Any: The Python object loaded from the specified path.

    Raises:
        AttributeError: If the object cannot be found in the specified path.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)
