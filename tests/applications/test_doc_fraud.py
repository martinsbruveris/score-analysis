import numpy as np
import pytest

from score_analysis import Scores
from score_analysis.applications import (
    DocLabel,
    FraudScores,
    binary_to_doc_label,
    doc_to_binary_label,
)


def test_enum():
    for value in ["genuine", "fraud"]:
        value = DocLabel(value)
        assert binary_to_doc_label(doc_to_binary_label(value)) == value


def test_create_fraud_scores():
    scores = FraudScores(genuines=[0.1, 0.2, 0.3], frauds=[0.1, 0.2])
    assert isinstance(scores, FraudScores)
    assert isinstance(scores, Scores)

    # Test we can use aliases
    scores.genuines[0] = 0.3
    scores.genuines = np.array([0.8])
    scores.frauds = np.array([0.2, 0.4])

    np.testing.assert_equal(scores.genuines, scores.pos)
    np.testing.assert_equal(scores.frauds, scores.neg)

    # Invalid score range
    with pytest.raises(ValueError):
        FraudScores(genuines=[-0.1], frauds=[])
    with pytest.raises(ValueError):
        FraudScores(genuines=[1.1], frauds=[])
    with pytest.raises(ValueError):
        FraudScores(genuines=[], frauds=[-0.1])
    with pytest.raises(ValueError):
        FraudScores(genuines=[], frauds=[1.2])

    # Check score ordering heuristic works
    with pytest.warns(UserWarning):
        FraudScores(genuines=[0.1], frauds=[0.3])

    # Create empty score object
    scores = FraudScores(genuines=[], frauds=[])


def test_from_labels():
    scores = FraudScores.from_labels(labels=[0, 1, 1], scores=[0.0, 1.0, 1.0])
    assert isinstance(scores, FraudScores)
