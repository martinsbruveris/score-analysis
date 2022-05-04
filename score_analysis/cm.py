"""
ConfusionMatrix module.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from . import metrics


def cm_class_metric(metric=None, axis: int = -1):
    """
    Wrapper for a per-class metric to ensure that
    - We simply call the metric for binary confusion matrices
    - We call the metric on the one_vs_all() matrix for multi-class matrices
    - We correctly apply the as_dict parameter.

    Args:
        metric: Callable with signature metric(self, as_dict), implementing the metric
            on a binary confusion matrix.
        axis: Axis in which

    Returns:
        Callable with the same signature implementing the metric for general confusion
        matrices and the as_dict paramter.
    """

    def decorator(_metric):
        def wrapper(self: ConfusionMatrix, *args, as_dict: bool = False, **kwargs):
            if self.binary and as_dict:
                raise ValueError("Cannot return as dict with binary matrices.")
            cm = self if self.binary else self.one_vs_all()
            res = _metric(cm, *args, **kwargs)
            return self._class_metric_as_dict(res, axis=axis) if as_dict else res

        return wrapper

    if metric is not None:
        return decorator(metric)
    else:
        return decorator


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
        binary: bool = False,
    ):
        """
        A confusion matrix can be created in two ways:

          * From labels, predictions and (optionally) weights
          * From a matrix already containing necessary information

        *Creating a confusion matrix from labels and predictions*

        If classes are not provided, we use all values that appear as either labels or
        predictions, in sorted order, as classes.

        The dtype of the confusion matrix is int if no weights vector is provided and
        the same as the dtype of the weights vector otherwise. The default weight is
        1 for all samples.

        *Creating a confusion matrix from a matrix*

        If matrix is a dict of dicts, we expect all entries to have the same set of
        keys, which will be used as classes. In this case the classes parameter can
        only be used to reorder the classes.

        If matrix is a pandas DataFrame, the index and column names are used as classes.
        We expect them to be the same (up to reordering). The classes parameter can
        only be used to reorder the classes.

        All other input types we attempt to convert to numpy arrays via np.asarray. The
        class names are either taken from the provided parameter or set to 0, ..., N.

        *Vectorized confusion matrices*

        When creating the confusion matrix from a matrix, we can use a matrix of shape
        (..., N, N) to represent a vectorized confusion matrix.

        *Binary confusion matrices*

        Binary confusion matrices are two-class confusion matrices, with the classes in
        the order [positive, negative]. The default labels are [1, 0]. Note that this
        is different from the default labels for a two-class non-binary confusion
        matrix, for which the default labels would be [0, 1].

        The main difference is that for a binary confusion matrix metrics, such as TPR,
        etc. return *scalar* values, while for general confusion matrices they return
        one value *per class*.
        """
        if matrix is not None:
            if labels is not None:
                raise ValueError("Cannot provide labels and matrix.")
            if predictions is not None:
                raise ValueError("Cannot provide predictions and matrix.")
            if weights is not None:
                raise ValueError("Cannot provide sample_weight and matrix.")

            self.matrix, self.classes = self._assign_from_matrix(
                matrix=matrix, classes=classes, binary=binary
            )
        else:
            if labels is None:
                raise ValueError("Must provide labels.")
            if predictions is None:
                raise ValueError("Must provide predictions.")

            self.matrix, self.classes = self._assign_from_predictions(
                labels, predictions, weights, classes, binary
            )

        self.binary = binary

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
        if self.binary and len(self.classes) != 2:
            raise ValueError("Binary confusion matrices must have exactly two classes.")

    @staticmethod
    def _assign_from_matrix(matrix, classes, binary):
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
                if not binary:
                    classes = np.asarray(list(range(matrix.shape[-1])))
                else:
                    # Binary CMs order classes as [positive, negative]
                    classes = np.array([1, 0])

        return matrix, classes

    @staticmethod
    def _assign_from_predictions(labels, predictions, weights, classes, binary):
        """Creates confusion matrix from labels and predictions."""
        labels = np.asarray(labels)
        predictions = np.asarray(predictions)

        if classes is None:
            if not binary:
                classes = np.unique(
                    np.concatenate([np.unique(labels), np.unique(predictions)])
                )
            else:
                # Binary CMs order classes as [positive, negative]
                classes = np.array([1, 0])
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
        return ConfusionMatrix(
            matrix=self.matrix[item], classes=self.classes, binary=self.binary
        )

    @property
    def nb_classes(self) -> int:
        """Number of classes of confusion matrix"""
        return len(self.classes)

    def one_vs_all(self) -> ConfusionMatrix:
        """
        Binarizes the confusion matrix using one-vs-all strategy.

        For an input confusion matrix of shape (..., N, N) with N classes, the output
        is a vectorized binary confusion matrix of shape (..., N, 2, 2), where
        cm[..., j] is the confusion matrix of class j (pos) against all other classes
        (neg).

        The one-vs-all operation is *not* idempotent. If we start with a binary
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
        return ConfusionMatrix(matrix=matrix, binary=True)

    def _class_metric_as_dict(self, arr: np.ndarray, axis: int = -1) -> dict:
        """
        Converts per-class metrics in array form to dict.

        We assume that arr is array of shape (X, N, Y), where the metric is present
        in the defined axis. The dictionary has classes as keys and arrays of shape
        (X, Y) as values.
        """
        res = {c: np.take(arr, j, axis=axis) for j, c in enumerate(self.classes)}
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

    @cm_class_metric
    def tp(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """True Positves"""
        return metrics.tp(self.matrix)

    @cm_class_metric
    def tn(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """True Negatives"""
        return metrics.tn(self.matrix)

    @cm_class_metric
    def fp(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """False Positives"""
        return metrics.fp(self.matrix)

    @cm_class_metric
    def fn(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """False Negatives"""
        return metrics.fn(self.matrix)

    @cm_class_metric
    def p(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """Condition Positve"""
        return metrics.p(self.matrix)

    @cm_class_metric
    def n(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """Condition Negative"""
        return metrics.n(self.matrix)

    @cm_class_metric
    def top(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """Test Outcome Positive"""
        return metrics.top(self.matrix)

    @cm_class_metric
    def ton(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """Test Outcome Negative"""
        return metrics.ton(self.matrix)

    @cm_class_metric
    def tpr(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """True Positve Rate"""
        return metrics.tpr(self.matrix)

    @cm_class_metric
    def tnr(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """True Negative Rate"""
        return metrics.tnr(self.matrix)

    @cm_class_metric
    def fpr(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """False Positive Rate"""
        return metrics.fpr(self.matrix)

    @cm_class_metric
    def fnr(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """False Negative Rate"""
        return metrics.fnr(self.matrix)

    @cm_class_metric(axis=-2)
    def tpr_ci(
        self, alpha: float = 0.05, *, as_dict: bool = False
    ) -> Union[dict, float, np.ndarray]:
        """True Positive Rate confidence interval"""
        return metrics.tpr_ci(self.matrix, alpha=alpha)

    @cm_class_metric(axis=-2)
    def tnr_ci(
        self, alpha: float = 0.05, *, as_dict: bool = False
    ) -> Union[dict, float, np.ndarray]:
        """True Negative Rate confidence interval"""
        return metrics.tnr_ci(self.matrix, alpha=alpha)

    @cm_class_metric(axis=-2)
    def fpr_ci(
        self, alpha: float = 0.05, *, as_dict: bool = False
    ) -> Union[dict, float, np.ndarray]:
        """False Positive Rate confidence interval"""
        return metrics.fpr_ci(self.matrix, alpha=alpha)

    @cm_class_metric(axis=-2)
    def fnr_ci(
        self, alpha: float = 0.05, *, as_dict: bool = False
    ) -> Union[dict, float, np.ndarray]:
        """False Negative Rate confidence interval"""
        return metrics.fnr_ci(self.matrix, alpha=alpha)

    @cm_class_metric
    def ppv(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """Positive Predictive Value"""
        return metrics.ppv(self.matrix)

    @cm_class_metric
    def npv(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """Negative Predictive Value"""
        return metrics.npv(self.matrix)

    @cm_class_metric
    def fdr(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """False Discovery Rate"""
        return metrics.fdr(self.matrix)

    @cm_class_metric
    def for_(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """False Omission Rate"""
        return metrics.for_(self.matrix)

    @cm_class_metric
    def class_accuracy(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """Class Accuracy"""
        return metrics.accuracy(self.matrix)

    @cm_class_metric
    def class_error_rate(self, as_dict: bool = False) -> Union[dict, float, np.ndarray]:
        """Class Error Rate"""
        return metrics.error_rate(self.matrix)
