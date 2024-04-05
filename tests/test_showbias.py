import numpy as np
import pandas as pd
import pytest

from score_analysis import BootstrapConfig, showbias


@pytest.mark.parametrize(
    "threshold, normalize, expected_values",
    [
        [[0.5], None, {"A": [0.0], "B": [1.0]}],
        [[0.0, 0.5, 1.0], None, {"A": [0.0, 0.0, 1.0], "B": [0.0, 1.0, 1.0]}],
        [[0.5], "by_overall", {"A": [0.0], "B": [2.0]}],
        [[0.5], "by_min", {"A": [0.0], "B": [1.0]}],
        [[0.0, 0.5, 1.0], "by_overall", {"A": [0.0, 0.0, 1.0], "B": [0.0, 2.0, 1.0]}],
    ],
)
def test_showbias(threshold, normalize, expected_values):
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "score": [0.8, 0.6, 0.4, 0.2],
            "label": [1, 1, 1, 1],
        }
    )
    result = showbias(
        data,
        metric="fnr",
        threshold=threshold,
        group_columns="group",
        label_column="label",
        score_column="score",
        normalize=normalize,
    )
    assert result.values.equals(
        pd.DataFrame.from_dict(expected_values, orient="index", columns=threshold)
    )


@pytest.mark.parametrize(
    "threshold, normalize, expected_values, expected_lower, expected_upper",
    [
        [[0.5], None, {"A": 0.0, "B": 1.0}, {"A": 0.0, "B": 1.0}, {"A": 0.0, "B": 1.0}],
        [
            [0.0, 0.5, 1.0],
            None,
            {"A": [0.0, 0.0, 1.0], "B": [0.0, 1.0, 1.0]},
            {"A": [0.0, 0.0, 1.0], "B": [0.0, 1.0, 1.0]},
            {"A": [0.0, 0.0, 1.0], "B": [0.0, 1.0, 1.0]},
        ],
        [
            [0.5],
            "by_overall",
            {"A": [0.0], "B": [2.0]},
            {"A": [0.0], "B": [2.0]},
            {"A": [0.0], "B": [2.0]},
        ],
        [
            [0.5],
            "by_min",
            {"A": [0.0], "B": [1.0]},
            {"A": [np.nan], "B": [np.nan]},
            {"A": [np.nan], "B": [np.nan]},
        ],
    ],
)
def test_showbias_bootstrap_ci(
    threshold, normalize, expected_values, expected_lower, expected_upper
):
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "score": [0.8, 0.6, 0.4, 0.2],
            "label": [1, 1, 1, 1],
        }
    )
    alpha = 0.05
    bootstrap_config = BootstrapConfig(bootstrap_method="quantile", nb_samples=500)
    result = showbias(
        data,
        metric="fnr",
        threshold=threshold,
        group_columns="group",
        label_column="label",
        score_column="score",
        normalize=normalize,
        bootstrap_ci=True,
        bootstrap_config=bootstrap_config,
        alpha=alpha,
    )
    assert result.values.equals(
        pd.DataFrame.from_dict(expected_values, orient="index", columns=threshold)
    )
    assert result.lower.equals(
        pd.DataFrame.from_dict(expected_lower, orient="index", columns=threshold)
    )
    assert result.upper.equals(
        pd.DataFrame.from_dict(expected_upper, orient="index", columns=threshold)
    )
    assert result.alpha == alpha


@pytest.mark.parametrize(
    "threshold, normalize, group1, group2, score, label, expected_data",
    [
        [
            [0.5],
            None,
            ["A", "A", "B", "B"],
            ["X", "Y", "X", "Y"],
            [0.95, 0.2, 0.55, 0.45],
            [1, 1, 1, 1],
            {
                ("A", "X"): 0.0,
                ("A", "Y"): 1.0,
                ("B", "X"): 0.0,
                ("B", "Y"): 1.0,
            },
        ],
        [
            [0.5],
            "by_overall",
            ["A", "A", "B", "B"],
            ["X", "Y", "X", "Y"],
            [0.95, 0.2, 0.55, 0.45],
            [1, 1, 1, 1],
            {
                ("A", "X"): 0.0,
                ("A", "Y"): 2.0,
                ("B", "X"): 0.0,
                ("B", "Y"): 2.0,
            },
        ],
        [
            [0.0, 0.25, 0.75, 1.0],
            None,
            ["C", "B", "A"],
            ["X", "Y", "Z"],
            [0.1, 0.5, 0.9],
            [1, 1, 1],
            {
                ("A", "Z"): [0.0, 0.0, 0.0, 1.0],
                ("B", "Y"): [0.0, 0.0, 1.0, 1.0],
                ("C", "X"): [0.0, 1.0, 1.0, 1.0],
            },
        ],
    ],
)
def test_showbias_multiple_group_columns(
    threshold, normalize, group1, group2, score, label, expected_data
):
    data = pd.DataFrame(
        {
            "group1": group1,
            "group2": group2,
            "score": score,
            "label": label,
        }
    )
    result = showbias(
        data,
        metric="fnr",
        threshold=threshold,
        normalize=normalize,
        group_columns=["group1", "group2"],
        label_column="label",
        score_column="score",
    )
    assert result.values.equals(
        pd.DataFrame.from_dict(expected_data, orient="index", columns=threshold)
    )


@pytest.mark.parametrize(
    "threshold, bootstrap_ci, number_decimal_points, expected",
    [
        [
            [0.3, 0.5, 0.7],
            False,
            2,
            (
                "| age_group   |   0.3 |   0.5 |   0.7 |\n"
                "|:------------|------:|------:|------:|\n"
                "| female      |  0.36 |  0.6  |  0.76 |\n"
                "| male        |  0.16 |  0.32 |  0.53 |"
            ),
        ],
        [
            [0.5],
            True,
            4,
            (
                "| age_group   | 0.5               |\n"
                "|:------------|:------------------|\n"
                "| female      | 0.6               |\n"
                "|             | (0.4127 - 0.7795) |\n"
                "| male        | 0.3158            |\n"
                "|             | (0.0952 - 0.5263) |"
            ),
        ],
    ],
)
def test_bias_frame_to_markdown(
    threshold, bootstrap_ci, number_decimal_points, expected
):
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age_group": np.random.choice(["male", "female"], size=100),
            "labels": np.random.choice([0, 1], size=100),
            "scores": np.random.uniform(0.0, 1.0, 100),
        }
    )
    bootstrap_config = BootstrapConfig(bootstrap_method="quantile", nb_samples=500)
    bias_frame = showbias(
        data=df,
        group_columns="age_group",
        label_column="labels",
        score_column="scores",
        metric="fnr",
        threshold=threshold,
        bootstrap_ci=bootstrap_ci,
        bootstrap_config=bootstrap_config,
    )

    bias_frame_markdown = bias_frame.to_markdown(
        reset_display_index=False, number_decimal_points=number_decimal_points
    )

    assert bias_frame_markdown == expected


def test_invalid_column_input():
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "score": [0.8, 0.6, 0.4, 0.2],
            "label": [1, 1, 1, 1],
        }
    )
    with pytest.raises(AssertionError):
        showbias(
            data,
            metric="fnr",
            threshold=[0.5],
            group_columns="",
            label_column="label",
            score_column="score",
        )
    with pytest.raises(AssertionError):
        showbias(
            data,
            metric="fnr",
            threshold=[0.5],
            group_columns="group",
            label_column="",
            score_column="score",
        )
    with pytest.raises(AssertionError):
        showbias(
            data,
            metric="fnr",
            threshold=[0.5],
            group_columns="group",
            label_column="label",
            score_column="",
        )


def test_invalid_normalize_input():
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "score": [0.8, 0.6, 0.4, 0.2],
            "label": [1, 1, 1, 1],
        }
    )
    with pytest.raises(ValueError):
        showbias(
            data,
            metric="fnr",
            threshold=[0.5],
            group_columns="group",
            label_column="label",
            score_column="score",
            normalize="unsupported",
        )
