"""Module to decoded predictions."""
import itertools
import operator
from typing import List, Tuple

import numpy as np
import torch


def matrix_to_string(model_output: torch.Tensor, vocab: str) -> Tuple[List[str], List[np.ndarray]]:
    """Convert the output of a neural network model to a list of strings and their corresponding confidence levels.

    Args:
        model_output (torch.Tensor): The output tensor from the model.
        vocab (str): The vocabulary string where each character corresponds to a label in the model's output.

    Returns:
        Tuple[List[str], List[np.ndarray]]: A tuple containing:
            - A list of strings representing the decoded predictions.
            - A list of numpy arrays, each array containing the confidence levels of the corresponding string.
    """
    labels, confs = postprocess(model_output)
    labels_decoded, conf_decoded = decode(labels_raw=labels, conf_raw=confs)
    string_pred = labels_to_strings(labels_decoded, vocab)
    return string_pred, conf_decoded


def postprocess(model_output: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Process the raw model output to obtain predicted labels and their confidence levels.

    Args:
        model_output (torch.Tensor): The raw output tensor from the model.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - A numpy array of predicted labels.
            - A numpy array of corresponding confidence levels.
    """
    output = model_output.permute(1, 0, 2)
    output = torch.nn.Softmax(dim=2)(output)
    confidences, labels = output.max(dim=2)
    confidences = confidences.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return labels, confidences


def decode(labels_raw: np.ndarray, conf_raw: np.ndarray) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Decode raw labels and confidences into a more structured format.

    Args:
        labels_raw (np.ndarray): Raw label data as numpy array.
        conf_raw (np.ndarray): Raw confidence data as numpy array.

    Returns:
        Tuple[List[List[int]], List[np.ndarray]]: A tuple containing:
            - A list of lists, where each inner list contains decoded label indices.
            - A list of numpy arrays, each containing the confidence levels for the corresponding list of labels.
    """
    result_labels = []
    result_confidences = []
    for labels, conf in zip(labels_raw, conf_raw):
        result_one_labels = []
        result_one_confidences = []
        for label, group in itertools.groupby(zip(labels, conf), operator.itemgetter(0)):
            if label > 0:
                result_one_labels.append(label)
                result_one_confidences.append(max(list(zip(*group))[1]))
        result_labels.append(result_one_labels)
        result_confidences.append(np.array(result_one_confidences))

    return result_labels, result_confidences


def labels_to_strings(labels: List[List[int]], vocab: str) -> List[str]:
    """Convert lists of label indices into strings using the provided vocabulary.

    Args:
        labels (List[List[int]]): A list of lists, where each inner list contains label indices.
        vocab (str): The vocabulary string used for converting label indices to characters.

    Returns:
        List[str]: A list of strings, each string corresponding to the decoded label indices.
    """
    strings = []
    for single_str_labels in labels:
        output_str = "".join(vocab[char_index - 1] if char_index > 0 else "_" for char_index in single_str_labels)
        strings.append(output_str)
    return strings
