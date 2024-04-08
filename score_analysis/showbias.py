from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from .group_scores import GroupScores
from .scores import DEFAULT_BOOTSTRAP_CONFIG, BinaryLabel, BootstrapConfig
from .utils import bootstrap_ci as get_bootstrap_ci


@dataclass
class BiasFrame:
    """
    A class for returning data from the `showbias` function.

    Args:
        values: A pandas DataFrame containing the observed metrics per group. This
            DataFrame is the primary result that the `showbias` function returns.
        alpha: The significance level for computing confidence intervals. (1 - alpha)
            is the probability that the true parameter is in the interval, assuming
            the experiment is repeated many times.
        lower: A pandas DataFrame with lower bounds of confidence intervals for the
            observed values. Set only if a bootstrap confidence interval was requested.
        upper: A pandas DataFrame with upper bounds of confidence intervals for the
            observed values. Set only if a bootstrap confidence interval was requested.
    """

    values: pd.DataFrame
    alpha: Optional[float] = None
    lower: Optional[pd.DataFrame] = None
    upper: Optional[pd.DataFrame] = None

    def to_markdown(
        self, number_decimal_points: int = 3, reset_display_index: bool = False
    ) -> str:
        """
        Prints a markdown-formatted table of the BiasFrame's values. If confidence
        intervals (lower and upper bounds) are present, each value is accompanied by
        its interval. Otherwise, only the observed values are displayed.

        The method optionally allows for resetting the DataFrame index before printing
        and customizing the number of decimal points for the numerical values.

        Args:
            number_decimal_points: The number of decimal points for formatting numerical
                values in the table. Defaults to 3, meaning values are shown with three
                decimal places.
            reset_display_index: If True, resets the DataFrame index before printing the
                table. This makes the index information a separate column in the table.
                Defaults to False, meaning the index is not reset unless specified.
        """
        if self.lower is None and self.upper is None:
            display_values = (
                self.values.reset_index() if reset_display_index else self.values
            )
            display_values = display_values.round(number_decimal_points)
            return display_values.to_markdown(index=not reset_display_index)
        else:
            display_values = []
            for column in self.values.columns:
                display_values.append(
                    self.values[column].round(number_decimal_points).astype(str)
                    + "\n"
                    + "("
                    + self.lower[column].round(number_decimal_points).astype(str)
                    + " - "
                    + self.upper[column].round(number_decimal_points).astype(str)
                    + ")"
                )
            display_values = pd.concat(display_values, axis=1)
            display_values = (
                display_values.reset_index() if reset_display_index else display_values
            )
            return display_values.to_markdown(index=not reset_display_index)


def showbias(
    data: pd.DataFrame,
    group_columns: Union[str, List[str]],
    label_column: str,
    score_column: str,
    metric: str,
    normalize: Optional[str] = None,
    bootstrap_ci: bool = False,
    bootstrap_config: BootstrapConfig = DEFAULT_BOOTSTRAP_CONFIG,
    alpha: float = 0.05,
    pos_label: int = 1,
    score_class: str = "pos",
    equal_class: Union[str, BinaryLabel] = "pos",
    **metric_kwargs,
) -> BiasFrame:
    """
    Calculates bias metrics for specific groups in a dataset, optionally
    normalising metrics and calculating confidence intervals.

    Args:
        data: Dataset with scores and actual labels.
        group_columns: Column(s) for dataset grouping.
        label_column: Column name for actual labels.
        score_column: Column name for predicted scores.
        metric: Bias metric to compute, must be a method of `ConfusionMatrix`.
        normalize: Normalisation method for computed group metrics.
            Possible values are:

             * "by_overall" normalizes by the overall metric computed
              on the entire dataset.
             * "by_min" normalizes by the minimum metric value observed
              across all groups

        bootstrap_ci: If True, enables bootstrapping for confidence intervals.
        bootstrap_config: Configuration for bootstrap sampling.
            Uses `DEFAULT_BOOTSTRAP_CONFIG` by default.
        alpha: Significance level for confidence intervals.
        pos_label: Label of the positive class.
        score_class: Indicates if scores denote positive or negative class membership.
        equal_class: Assignment of samples at threshold.
        **metric_kwargs: Additional keyword arguments for the metric computation, e.g.,
            `threshold=[0.5]` if metric requires a threshold argument.

    Returns:
        BiasFrame: Contains computed bias values for each group, including confidence
        intervals if calculated.

    Raises:
        AssertionError: If any column validations fail.
    """
    _validate_column_inputs(data, group_columns, label_column, score_column)

    if isinstance(group_columns, str):
        group_column = group_columns
    elif isinstance(group_columns, Iterable):
        group_column = "_".join(group_columns)
        data[group_column] = data.apply(
            lambda row: "_".join(row[col] for col in group_columns), axis=1
        )
    else:
        raise TypeError(
            f"Got unexpected type {type(group_columns)} value for `group_columns`"
        )

    score_object = GroupScores.from_labels(
        scores=data[score_column].values,
        labels=data[label_column].values,
        groups=data[group_column].values,
        pos_label=pos_label,
        score_class=score_class,
        equal_class=equal_class,
    )

    def calculate_metric(sample: GroupScores, **kwargs):
        return getattr(sample.cm(**kwargs), metric)()

    def calculate_group_metric(sample: GroupScores, **kwargs):
        return getattr(sample.group_cm(**kwargs), metric)()

    group_names = score_object.groups
    group_index = _get_group_index(group_names, group_columns)
    group_metrics = calculate_group_metric(score_object, **metric_kwargs)

    if normalize is not None:
        group_metrics = _apply_normalisation(
            group_metrics, score_object, calculate_metric, normalize, **metric_kwargs
        )

    if bootstrap_ci:

        samples = score_object.bootstrap_metric(
            calculate_group_metric,
            config=bootstrap_config,
            **metric_kwargs,
        )

        if normalize is not None:
            samples = _apply_normalisation(
                samples,
                score_object,
                calculate_metric,
                normalize,
                **metric_kwargs,
            )
        bootstrap_ci = get_bootstrap_ci(
            theta=samples,
            theta_hat=calculate_group_metric(score_object, **metric_kwargs),
            alpha=alpha,
            method=bootstrap_config.bootstrap_method,
        )
        return BiasFrame(
            values=pd.DataFrame(
                group_metrics.tolist(),
                index=group_index,
                columns=metric_kwargs.get("threshold"),
            ),
            alpha=alpha,
            lower=pd.DataFrame(
                np.squeeze(bootstrap_ci[..., 0]).tolist(),
                index=group_index,
                columns=metric_kwargs.get("threshold"),
            ),
            upper=pd.DataFrame(
                np.squeeze(bootstrap_ci[..., 1]).tolist(),
                index=group_index,
                columns=metric_kwargs.get("threshold"),
            ),
        )

    group_metrics = pd.DataFrame(
        group_metrics.tolist(),
        index=group_index,
        columns=metric_kwargs.get("threshold"),
    )
    return BiasFrame(values=group_metrics)


def _apply_normalisation(
    group_metrics: np.ndarray,
    score_object: GroupScores,
    metric: Callable,
    normalize: str,
    **kwargs,
):
    """
    Normalizes group metrics based on a specified method.

    Adjusts the metric values for each group by dividing them by a denominator
    determined by the `normalize` parameter: either the overall metric across
    the entire dataset ("by_overall") or the minimum metric value across groups
    ("by_min"). If the denominator is 0, the original group metrics are returned.

    Args:
        group_metrics: Array of metric values for each group.
        score_object: Object encapsulating score data and methods.
        metric: Function to compute the metric.
        normalize: Method of normalisation, "by_overall" or "by_min".
        **kwargs: Additional keyword arguments passed to the metric function.

    Returns:
        np.ndarray: Normalized group metric values.

    Raises:
        ValueError: If an unsupported `normalize` value is provided.
    """

    if normalize == "by_overall":
        denominator_metric = metric(score_object, **kwargs)
    elif normalize == "by_min":
        denominator_metric = np.min(group_metrics, axis=0)
    else:
        raise ValueError(f"Unsupported value for {normalize=}.")

    return np.where(
        denominator_metric != 0,
        np.divide(
            group_metrics,
            denominator_metric,
            out=np.zeros_like(group_metrics, dtype=float),
            where=denominator_metric != 0,
        ),
        group_metrics,
    )


def _validate_column_inputs(
    data: pd.DataFrame,
    group_columns: Union[str, List[str]],
    label_column: str,
    score_column: str,
):
    """
    Validate inputs for the showbias function.

    Checks that `data` is a DataFrame and `group_columns`, `label_column`, and
    `score_column` are columns in the DataFrame.

    Args:
        data: Dataset with scores and actual labels.
        group_columns: Column name(s) for grouping the dataset. Can be a single name
            or a list of names.
        label_column: Column name containing the actual labels.
        score_column: Column name containing the predicted scores.

    Raises:
        AssertionError: If any input validations fail.
    """
    assert isinstance(data, pd.DataFrame), "`data` needs to be a pandas data frame"
    if isinstance(group_columns, list):
        assert all(
            [group in data.columns for group in group_columns]
        ), "`group_columns` not found in `data`"
    else:
        assert group_columns in data.columns, "`group_columns` not found in `data`"
    assert label_column in data.columns, "`label_column` not found in `data`"
    assert score_column in data.columns, "`score_column` not found in `data`"


def _get_group_index(group_names: np.ndarray, group_columns: Union[str, List[str]]):
    """
    Creates a pandas index object for group identifiers.

    Args:
        group_names: Array of strings identifying groups. For MultiIndex, strings should
            be concatenated values separated by underscores.
        group_columns: List of column names for grouping. The list's length should match
            the number of elements in each group identifier when split by underscores.
            Can be a single string for a simple Index.

    Returns:
        pd.Index or pd.MultiIndex: A pandas Index or MultiIndex object representing the
        group identifiers, suitable for indexing or grouping operations.

    Raises:
        ValueError: If `group_columns` is a list and the length of any group identifier
        (when split) does not match the length of `group_columns`.
    """
    if isinstance(group_columns, list):
        group_index = list(zip(*[group_name.split("_") for group_name in group_names]))
        return pd.MultiIndex.from_arrays(group_index, names=group_columns)
    else:
        return pd.Index(group_names, name=group_columns)
