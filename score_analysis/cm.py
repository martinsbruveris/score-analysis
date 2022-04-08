"""
ConfusionMatrix module.
"""
from __future__ import annotations

from typing import Union

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
            matrix[..., j, 1, 1] = np.sum(self.matrix, axis=(-1, -2)) - np.sum(
                matrix[..., j, :, :], axis=(-1, -2)
            )
        return BinaryConfusionMatrix(matrix=matrix)

    def _class_metric_as_dict(self, arr: np.ndarray) -> dict:
        """
        Converts per-class metrics in array form to dict.

        We assume that arr is array of shape (..., N). The dictionary has classes as
        keys and arrays of shape (...) as values.
        """
        res = {c: arr[..., j] for j, c in enumerate(self.classes)}
        return res

    def pop(self) -> float:
        """Population"""
        return metrics.pop(self.matrix)

    def accuracy(self) -> float:
        """Accuracy"""
        return metrics.accuracy(self.matrix)

    def error_rate(self) -> float:
        """Error Rate"""
        return metrics.error_rate(self.matrix)

    def tp(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """True Positves"""
        res = metrics.tp(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def tn(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """True Negatives"""
        res = metrics.tn(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def fp(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """False Positives"""
        res = metrics.fp(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def fn(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """False Negatives"""
        res = metrics.fn(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def p(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """Condition Positve"""
        res = metrics.p(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def n(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """Condition Negative"""
        res = metrics.n(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def top(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """Test Outcome Positive"""
        res = metrics.top(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def ton(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """Test Outcome Negative"""
        res = metrics.ton(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def tpr(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """True Positve Rate"""
        res = metrics.tpr(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def tnr(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """True Negative Rate"""
        res = metrics.tnr(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def fpr(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """False Positive Rate"""
        res = metrics.fpr(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def fnr(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """False Negative Rate"""
        res = metrics.fnr(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def tpr_ci(
        self, alpha: float = 0.05, as_dict: bool = False
    ) -> Union[dict, np.ndarray]:
        """True Positive Rate confidence interval"""
        res = metrics.tpr_ci(self.one_vs_all().matrix, alpha=alpha)  # (..., N, 2)
        if as_dict:
            res = np.swapaxes(res, -1, -2)  # (..., 2, N)
            return self._class_metric_as_dict(res)
        else:
            return res

    def tnr_ci(
        self, alpha: float = 0.05, as_dict: bool = False
    ) -> Union[dict, np.ndarray]:
        """True Negative Rate confidence interval"""
        res = metrics.tnr_ci(self.one_vs_all().matrix, alpha=alpha)  # (..., N, 2)
        if as_dict:
            res = np.swapaxes(res, -1, -2)  # (..., 2, N)
            return self._class_metric_as_dict(res)
        else:
            return res

    def fpr_ci(
        self, alpha: float = 0.05, as_dict: bool = False
    ) -> Union[dict, np.ndarray]:
        """False Positive Rate confidence interval"""
        res = metrics.fpr_ci(self.one_vs_all().matrix, alpha=alpha)  # (..., N, 2)
        if as_dict:
            res = np.swapaxes(res, -1, -2)  # (..., 2, N)
            return self._class_metric_as_dict(res)
        else:
            return res

    def fnr_ci(
        self, alpha: float = 0.05, as_dict: bool = False
    ) -> Union[dict, np.ndarray]:
        """False Negative Rate confidence interval"""
        res = metrics.fnr_ci(self.one_vs_all().matrix, alpha=alpha)  # (..., N, 2)
        if as_dict:
            res = np.swapaxes(res, -1, -2)  # (..., 2, N)
            return self._class_metric_as_dict(res)
        else:
            return res

    def ppv(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """Positive Predictive Value"""
        res = metrics.ppv(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def npv(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """Negative Predictive Value"""
        res = metrics.npv(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def fdr(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """False Discovery Rate"""
        res = metrics.fdr(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def for_(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """False Omission Rate"""
        res = metrics.for_(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def class_accuracy(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """Class Accuracy"""
        res = metrics.accuracy(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res

    def class_error_rate(self, as_dict: bool = False) -> Union[dict, np.ndarray]:
        """Class Error Rate"""
        res = metrics.error_rate(self.one_vs_all().matrix)
        return self._class_metric_as_dict(res) if as_dict else res


# noinspection PyMethodOverriding
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

    def tp(self) -> float:
        """True Positives"""
        return metrics.tp(self.matrix)

    def tn(self) -> float:
        """True Negatives"""
        return metrics.tn(self.matrix)

    def fp(self) -> float:
        """False Positives"""
        return metrics.fp(self.matrix)

    def fn(self) -> float:
        """False Negatives"""
        return metrics.fn(self.matrix)

    def p(self) -> float:
        """Condition Positive"""
        return metrics.p(self.matrix)

    def n(self) -> float:
        """Condition Negative"""
        return metrics.n(self.matrix)

    def top(self) -> float:
        """Test Outcome Positive"""
        return metrics.top(self.matrix)

    def ton(self) -> float:
        """Test Outcome Negative"""
        return metrics.ton(self.matrix)

    def tpr(self) -> float:
        """True Positve Rate"""
        return metrics.tpr(self.matrix)

    def tnr(self) -> float:
        """True Negative Rate"""
        return metrics.tnr(self.matrix)

    def fpr(self) -> float:
        """False Positive Rate"""
        return metrics.fpr(self.matrix)

    def fnr(self) -> float:
        """False Negative Rate"""
        return metrics.fnr(self.matrix)

    def tpr_ci(self, alpha: float = 0.05) -> np.ndarray:
        """True Positve Rate confidence interval"""
        return metrics.tpr_ci(self.matrix, alpha=alpha)

    def tnr_ci(self, alpha: float = 0.05) -> np.ndarray:
        """True Negative Rate confidence interval"""
        return metrics.tnr_ci(self.matrix, alpha=alpha)

    def fpr_ci(self, alpha: float = 0.05) -> np.ndarray:
        """False Positve Rate confidence interval"""
        return metrics.fpr_ci(self.matrix, alpha=alpha)

    def fnr_ci(self, alpha: float = 0.05) -> np.ndarray:
        """False Negative Rate confidence interval"""
        return metrics.fnr_ci(self.matrix, alpha=alpha)

    def ppv(self) -> float:
        """Positive Predictive Value"""
        return metrics.ppv(self.matrix)

    def npv(self) -> float:
        """Negative Predictive Value"""
        return metrics.npv(self.matrix)

    def fdr(self) -> float:
        """False Discovery Rate"""
        return metrics.fdr(self.matrix)

    def for_(self) -> float:
        """False Omission Rate"""
        return metrics.for_(self.matrix)
