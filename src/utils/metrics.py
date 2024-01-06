"""This module provides functionalities to set up evaluation metrics used in a neural network evaluation process."""
import itertools

import numpy as np
import torch
from nltk.metrics.distance import edit_distance as ed
from torchmetrics import Metric, MetricCollection


def get_metrics() -> MetricCollection:
    """
    Create and returns a collection of metrics for evaluating string-based data.

    This function initializes a MetricCollection object, which is a container for multiple metrics.
    It specifically includes two types of metrics:

    1. StringMatchMetric: This metric evaluates the exact string match between two strings.
    2. EditDistanceMetric: This metric calculates the edit distance between two strings.
        Edit distance is the number of edits (insertions, deletions, or substitutions) needed to convert one string
        into another. It is valuable for scenarios where close, but not exact, matches are significant.

    Returns:
        MetricCollection: An object containing the specified metrics.
    """
    return MetricCollection(
        {
            "string_match": StringMatchMetric(),
            "edit_distance": EditDistanceMetric(),
        },
    )


# pylint: disable=no-member,arguments-differ
class StringMatchMetric(Metric):
    """
    A metric class for computing the string match accuracy.

    Attributes:
        correct (torch.Tensor): A tensor that keeps track of the total number of correct predictions.
        total (torch.Tensor): A tensor that keeps track of the total number of predictions.
    """

    def __init__(self) -> None:
        """Initialize the StringMatchMetric object with default state values."""
        super().__init__()
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    @staticmethod
    def process_sequence(seq: np.ndarray) -> list:
        """
        Process a sequence by removing consecutive duplicates and filtering out zeros.

        Args:
            seq (Iterable): The sequence to be processed.

        Returns:
            list: A processed sequence with consecutive duplicates removed and zeros filtered out.
        """
        return [key for key, _ in itertools.groupby(seq) if key > 0]

    def string_match(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate the string match accuracy between predictions and targets values.

        This function compares predicted sequences with the ground truth sequences. The prediction and truth
        are tensors where each sequence is represented as a series of integer tokens. The comparison is based on
        whether the entire sequence of tokens matches exactly.

        The function first converts the predicted tensor to a numpy array of token indices, representing the most
        likely token at each position in each sequence. It then compares these sequences to the ground truth,
        considering a match valid only if the entire sequence matches.

        Note: Tokens with an index greater than 0 are considered in the comparison, assuming that 0 represents a
        padding or non-significant token.

        Args:
            preds (torch.Tensor): The predictions tensor. Tensor shape is expected to be [Slength, Batch, N_classes].
            targets (torch.Tensor): The ground truth tensor. Tensor shape is expected to be [Batch, sequence length].

        Returns:
            float: The proportion of sequences in the batch that match exactly, as a float.
        """
        preds = preds.permute(1, 0, 2)
        preds = torch.argmax(preds, dim=2)
        preds = preds.detach().cpu().numpy()

        targets = targets.detach().cpu().numpy()

        processed_preds = [self.process_sequence(seq) for seq in preds]
        processed_targets = [[element for element in seq if element != 0] for seq in targets]
        valid_matches = sum(
            float(np.array_equal(pred, target)) for pred, target in zip(processed_preds, processed_targets)
        )
        return valid_matches / len(processed_preds)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the state of the metric with new predictions and targets.

        Args:
            preds (torch.Tensor): The predictions made by the model.
            target (torch.Tensor): The ground truth values.
        """
        batch_size = torch.tensor(target.shape[0])
        metric = torch.tensor(self.string_match(preds, target))
        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self):
        """Compute the string match accuracy over all predictions.

        Returns:
            torch.Tensor: The accuracy of the predictions.
        """
        return self.correct / self.total  # type: ignore


class EditDistanceMetric(Metric):
    """
    A class for computing the Edit Distance metric in a batch-wise fashion.

    This metric computes the sum of edit distances between pairs of prediction and target tensors,
    and normalizes the sum by the total number of samples to get an average edit distance.

    Attributes:
        correct (torch.Tensor): A tensor that accumulates the total edit distance.
        total (torch.Tensor): A tensor that counts the total number of samples processed.
    """

    def __init__(self) -> None:
        """Initialize the EditDistanceMetric object, setting up the state variables for correct and total samples."""
        super().__init__()
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    @staticmethod
    def process_sequence(seq: np.ndarray):
        """
        Process a sequence by removing consecutive duplicates and filtering out zeros.

        Args:
            seq (Iterable): The sequence to be processed.

        Returns:
            list: A processed sequence with consecutive duplicates removed and zeros filtered out.
        """
        return [key for key, _ in itertools.groupby(seq) if key > 0]

    def edit_distance(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate the average edit distance between predicted and target sequences.

        Args:
            preds (torch.Tensor): The predictions tensor.
            targets (torch.Tensor): The ground truth targets tensor.

        Returns:
            float: The average edit distance.
        """
        preds = preds.permute(1, 0, 2)
        preds = torch.Tensor.argmax(preds, dim=2)
        preds = preds.detach().cpu().numpy()

        targets = targets.detach().cpu().numpy()

        processed_preds = [self.process_sequence(seq) for seq in preds]
        processed_targets = [[element for element in seq if element != 0] for seq in targets]

        total_distance = 0
        for pred_sequence, target_sequence in zip(processed_preds, processed_targets):
            s_pred = "".join(map(chr, pred_sequence))
            s_target = "".join(map(chr, target_sequence))
            distance = ed(s_pred, s_target)
            total_distance += distance

        return total_distance / len(processed_preds)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the metric's state with the edit distance for a batch of predictions and targets.

        The function calculates the edit distance for each pair in the batch and updates the total
        distance and total samples accordingly.

        Args:
            preds (torch.Tensor): The predictions tensor.
            target (torch.Tensor): The ground truth targets tensor.
        """
        batch_size = torch.tensor(target.shape[0])
        metric = torch.tensor(self.edit_distance(preds, target))
        self.correct += metric * batch_size
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        """
        Compute the average edit distance over all samples.

        Returns:
            torch.Tensor: The average edit distance computed over all samples.
        """
        return self.correct / self.total  # type: ignore
