from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Union

import numpy as np

from score_analysis import Scores
from score_analysis.scores import BinaryLabel


class DocLabel(Enum):
    """Enum for genuine/fraud labels."""

    pos = "genuine"
    neg = "fraud"


def doc_to_binary_label(label: Union[DocLabel, str]) -> BinaryLabel:
    """Converts DocLabel to BinaryLabel."""
    return BinaryLabel(DocLabel(label).name)


def binary_to_doc_label(label: Union[BinaryLabel, str]) -> DocLabel:
    """Converts BinaryLabel to DocLabel."""
    label = BinaryLabel(label)
    return DocLabel.pos if label == BinaryLabel.pos else DocLabel.neg


# noinspection PyMethodOverriding
class FraudScores(Scores):
    def __init__(
        self,
        *,
        genuines,
        frauds,
        nb_easy_genuines: int = 0,
        nb_easy_frauds: int = 0,
        score_class: Union[DocLabel, str] = "genuine",
    ):
        """
        Args:
            genuines: Scores for genuine samples
            frauds: Scores for fraud samples
            nb_easy_genuines: Number of genuine samples we assume are always correctly
                classified.
            nb_easy_frauds: Number of fraud samples we assume are always correctly
                classified.
        """
        super().__init__(
            pos=genuines,
            neg=frauds,
            nb_easy_pos=nb_easy_genuines,
            nb_easy_neg=nb_easy_frauds,
            score_class=doc_to_binary_label(score_class),
            equal_class=doc_to_binary_label("genuine"),
        )

        # Input checks
        if np.any(self.genuines < 0) or np.any(self.genuines > 1):
            raise ValueError("Genuine scores must be between 0.0 and 1.0.")
        if np.any(self.frauds < 0) or np.any(self.frauds > 1):
            raise ValueError("Fraud scores must be between 0.0 and 1.0.")

        # This is a heuristic to check that score_class has been set correctly
        if len(self.genuines) > 0 and len(self.frauds) > 0:
            genuines_larger = np.median(self.genuines) >= np.median(self.frauds)
            score_class_guess = "genuine" if genuines_larger else "fraud"
            if self.score_class != doc_to_binary_label(score_class_guess):
                warnings.warn(
                    "Warning: score_class parameter might be set incorrectly.\n"
                    "Medians of genuine and fraud are ordered other than expected."
                )

    @property
    def genuines(self):
        """Alias for positive scores."""
        return self.pos

    @genuines.setter
    def genuines(self, value):
        self.pos = value

    @property
    def frauds(self):
        """Alias for negative scores."""
        return self.neg

    @frauds.setter
    def frauds(self, value):
        self.neg = value

    @staticmethod
    def from_labels(
        labels,
        scores,
        *,
        genuine_label: Any = 1,
        nb_easy_genuines: int = 0,
        nb_easy_frauds: int = 0,
        score_class: Union[DocLabel, str] = "genuine",
    ) -> FraudScores:
        """
        Args:
            labels: Array with sample labels
            scores: Array with sample scores
            genuine_label: The label of genuine samples. All other labels are treated as
                fraud labels.
            nb_easy_genuines: Number of genuine samples we assume are always correctly
                classified.
            nb_easy_frauds: Number of fraud samples we assume are always correctly
                classified.
            score_class: Do scores indicate membership of the positive or the negative
                class?

        Returns:
            A FraudScores instance.
        """
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        genuines = scores[labels == genuine_label]
        frauds = scores[labels != genuine_label]
        return FraudScores(
            genuines=genuines,
            frauds=frauds,
            nb_easy_genuines=nb_easy_genuines,
            nb_easy_frauds=nb_easy_frauds,
            score_class=score_class,
        )
