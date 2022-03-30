"""
ConfusionMatrix module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


# TODO: Output format pandas dataframe
# TODO: Output format numpy array
# TODO: Output format table (dict of dicts)
class ConfusionMatrix:
    """
    Confusion matrix class with support for vectorized confusion matrices.

    >>> labels      = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    >>> predictions = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    >>> cm = ConfusionMatrix(labels=labels, predictions=predictions)
    >>> cm.classes
    [0, 1, 2]
    >>> cm.table
    {0: {0: 3, 1: 0, 2: 0}, 1: {0: 0, 1: 1, 2: 2}, 2: {0: 2, 1: 1, 2: 3}}
    >>> cm2 = ConfusionMatrix(matrix={"a": {"a": 1, "b":2}, "b": {"a": 0, "b": 5}})
    >>> cm2.classes
    ["a", "b"]
    """

    def __init__(
        self,
        labels=None,
        predictions=None,
        *,
        weights=None,
        matrix=None,
        classes=None,
    ):
        """
        A confusion matrix can be created in two ways:
        * From labels, predictions and (optionally) weights
        * From a matrix already containing necessary information

        Creating a confusion matrix from labels and predictions

        If classes are not provided, we use all values that appear as either labels or
        predictions, in sorted order, as classes.

        The dtype of the confusion matrix is int if no weights vector is provided and
        the same as the dtype of the weights vector otherwise. The default weights
        are 1 for all samples.

        Creating a confusion matrix from a matrix

        If matrix is a dict of dicss, we expect all entries to have the same set of
        keys, which will be used as classes. In this case the classes parameter can
        only be used to reorder the classes.

        If matrix is a pandas DataFrame, the index and column names are used as classes.
        We expect them to be the same (up to reordering). The classes parameter can
        only be used to reorder the classes.

        All other input types we attempt to convert to numpy arrays via np.asarray. The
        class names are either taken from the provided parameter or set to 0, ..., n.

        Vectorized confusion matrices

        When creating the confusion matrix from a matrix, we can use a matrix of shape
        (..., n, n) to represent a vectorized confusion matrix.
        """
        if matrix is not None:
            if labels is not None:
                raise ValueError("Cannot provide labels and matrix.")
            if predictions is not None:
                raise ValueError("Cannot provide predictions and matrix.")
            if weights is not None:
                raise ValueError("Cannot provide sample_weight and matrix.")

            self.matrix, self.classes = self._assign_from_matrix(matrix, classes)
        else:
            if labels is None:
                raise ValueError("Must provide labels.")
            if predictions is None:
                raise ValueError("Must provide predictions.")

            self.matrix, self.classes = self._assign_from_predictions(
                labels, predictions, weights, classes
            )

        # Input checks
        if self.matrix.ndim < 2:
            raise ValueError("Matrix must be at least two-dimensional.")
        if self.matrix.shape[-1] != self.matrix.shape[-2]:
            raise ValueError("Last two matrix dimensions must be equal.")
        if len(self.classes) < 2:
            raise ValueError("At least two classes must be provided.")
        if len(self.classes) != self.matrix.shape[-1]:
            raise ValueError("Number of classes must be compatible with matrix.")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Class names must be unique.")

    @staticmethod
    def _assign_from_matrix(matrix, classes):
        """Creates confusion matrix from a matrix represenations."""
        if isinstance(matrix, dict):  # Dict of dicts
            if classes is not None:
                classes = np.asarray(classes)
                if set(classes) != set(matrix.keys()):
                    raise ValueError("Matrix keys not same as provided classes.")
            else:
                classes = np.asarray(list(matrix.keys()))

            classes_set = set(classes)
            for row in matrix.values():
                if set(row.keys()) != classes_set:
                    raise ValueError("All entries in matrix must have same keys.")

            matrix = [[matrix[r][c] for c in classes] for r in classes]
            matrix = np.asarray(matrix)

        elif isinstance(matrix, pd.DataFrame):
            rows = list(matrix.index)
            cols = list(matrix.columns)
            if set(rows) != set(cols):
                raise ValueError("Matrix must have same rows as columns")
            if len(set(rows)) != len(rows):
                raise ValueError("Matrix row names must be unique")
            if len(set(cols)) != len(cols):
                raise ValueError("Matrix column names must be unique")
            if classes is not None:
                classes = np.asarray(classes)
                if set(classes) != set(rows):
                    raise ValueError("Row names not same as provided classes.")
            else:
                classes = np.asarray(rows)  # We take class names from labels

            matrix = matrix.loc[classes, classes]  # Reorder if necessary
            matrix = np.asarray(matrix.values)

        else:
            # Try to coerce matrix as array; this should handle other input types such
            # as lists of lists, etc. And of course numpy arrays. This is the only
            # way to create vectorized confusion matrices.
            matrix = np.asarray(matrix)
            if classes is not None:
                classes = np.asarray(classes)
            else:
                classes = list(range(matrix.shape[-1]))

        return matrix, classes

    @staticmethod
    def _assign_from_predictions(labels, predictions, weights, classes):
        """Creates confusion matrix from labels and predictions."""
        labels = np.asarray(labels)
        predictions = np.asarray(predictions)

        if classes is None:
            classes = np.unique(
                np.concatenate([np.unique(labels), np.unique(predictions)])
            )
        else:
            classes = np.asarray(classes)
        idx_map = {c: i for i, c in enumerate(classes)}

        if weights is not None:
            if len(weights) != len(labels):
                raise ValueError("sample_weight must have same length as labels.")
            weights = np.asarray(weights)
        else:
            weights = np.ones_like(labels, dtype=int)

        matrix = np.zeros((len(classes), len(classes)), dtype=weights.dtype)
        for label, pred, weight in zip(labels, predictions, weights):
            matrix[idx_map[label]][idx_map[pred]] += weight

        return matrix, classes

    def __getitem__(self, item) -> ConfusionMatrix:
        """
        Returns confusion matrix at given index.
        """
        return ConfusionMatrix(matrix=self.matrix[item], classes=self.classes)

    @property
    def nb_classes(self) -> int:
        """Number of classes for confusion matrix"""
        return self.matrix.shape[-1]

    def one_vs_all(self) -> BinaryConfusionMatrix:
        """
        Binarizes the confusion matrix using one-vs-all strategy.

        For an input confusion matrix of shape (..., N, N) with N classes, the output
        is a vectorized binary confusion matrix of shape (..., N, 2, 2), where
        cm[..., j] is the confusion matrix of class j (pos) against all other classes
        (neg).

        The one-vs-all operation is _not_ idempotent. If we start with a binary
        confusion matrix, we will generate a vector of two matrices, so each class gets
        to play the role of the positive class.

        Returns:
            Vectorized binary confusion matrix of shape (..., N, 2, 2).
        """
        dims = self.matrix.shape[:-2]  # Extra dimensions
        matrix = np.zeros((*dims, self.nb_classes, 2, 2), dtype=self.matrix.dtype)
        for j in range(self.nb_classes):
            matrix[..., j, 0, 0] = self.matrix[..., j, j]
            matrix[..., j, 0, 1] = (
                np.sum(self.matrix[..., j, :], axis=-1) - matrix[..., j, 0, 0]
            )
            matrix[..., j, 1, 0] = (
                np.sum(self.matrix[..., :, j], axis=-1) - matrix[..., j, 0, 0]
            )
            matrix[..., j, 1, 1] = (
                np.sum(self.matrix, axis=(-1, -2))
                - np.sum(matrix[..., j, :, :], axis=(-1, -2))
            )
        return BinaryConfusionMatrix(matrix=matrix)

    def accuracy(self):
        """Accuracy"""
        return metrics.accuracy(self.matrix)


class BinaryConfusionMatrix(ConfusionMatrix):
    """
    Confusion matrix for two-class classifications.
    """

    def __init__(
        self,
        labels=None,
        predictions=None,
        *,
        weights=None,
        matrix=None,
        pos_label=1,
        neg_label=0,
    ):
        super().__init__(
            labels=labels,
            predictions=predictions,
            weights=weights,
            matrix=matrix,
            classes=[pos_label, neg_label],
        )

    def tpr(self) -> float:
        """True Positve Rate"""
        return metrics.tpr(self.matrix)

    def fnr(self) -> float:
        """False Negative Rate"""
        return metrics.fnr(self.matrix)

    def tnr(self) -> float:
        """True Negative Rate"""
        return metrics.tnr(self.matrix)

    def fpr(self) -> float:
        """False Positive Rate"""
        return metrics.fpr(self.matrix)
