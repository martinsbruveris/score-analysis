from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from score_analysis import BiasFrame


def plot_single_threshold(
    bias_frame: BiasFrame, threshold: float, title: Optional[str] = None
) -> plt.Figure:
    """
    Plots the observed values at a specific threshold, optionally with their
    confidence intervals if they are available.

    This method visualizes the relationship between different groups and a specific
    metric at the given threshold. If confidence intervals are available (i.e.,
    the `lower` and `upper` attributes are not None), they are displayed as error
    bars around each point, providing a visual indication of the variability or
    uncertainty around the observed values. If confidence intervals are not
    available, a simple scatter plot of the observed values is shown.

    Args:
        threshold: The specific threshold value for plotting observed values (and
            optionally, their confidence intervals). Selects the corresponding
            column in the `values` DataFrame, and if available, in the `lower` and
            `upper` DataFrames.
        title: Custom title for the plot. If not provided, a default title is
            generated that includes the DataFrame's index name and the specified
            threshold. Defaults to None.

    Returns:
        A matplotlib Figure object displaying the observed values at a specific
        threshold, with error bars for confidence intervals if available.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    if bias_frame.lower is None and bias_frame.upper is None:
        plt.scatter(
            bias_frame.values[threshold],
            (
                [" x ".join(item) for item in bias_frame.values.index]
                if isinstance(bias_frame.values.index, pd.core.indexes.multi.MultiIndex)
                else bias_frame.values.index
            ),
            marker="D",
            color="#999999",
            edgecolor="#ca0020",
            s=50,
        )
        max_value = bias_frame.values[threshold].max()

    else:
        error_bars = np.vstack(
            (
                bias_frame.values[threshold].values
                - bias_frame.lower[threshold].values,
                bias_frame.upper[threshold].values
                - bias_frame.values[threshold].values,
            )
        )
        plt.errorbar(
            bias_frame.values[threshold],
            bias_frame.values.index,
            xerr=error_bars,
            ecolor="#999999",
            capsize=8,
            capthick=3,
            elinewidth=3,
            marker="D",
            markersize=8,
            mfc="#999999",
            mec="#ca0020",
            linestyle="None",
        )
        max_value = max(
            bias_frame.values[threshold].max(),
            (bias_frame.values[threshold] + error_bars[1]).max(),
        )

    plt.xlim(0, max_value * 1.1)
    plt.xlabel("Metric values")
    plt.ylabel(bias_frame.values.index.name)
    title_with_ci = (
        ""
        if bias_frame.lower is None and bias_frame.upper is None
        else "with Confidence Intervals"
    )
    title_category = (
        bias_frame.values.index.names
        if isinstance(bias_frame.values.index, pd.core.indexes.multi.MultiIndex)
        else bias_frame.values.index.name
    )
    title = (
        f"Metric values {title_with_ci} by {title_category}, threshold: {threshold}"
        if title is None
        else title
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    return plt.gcf()


def plot_multiple_thresholds(
    bias_frame, log_scale: bool = False, title: Optional[str] = None
) -> plt.Figure:
    """
    Plots metric values for different groups across a range of thresholds,
    optionally with their confidence intervals.

    This method generates a line plot for each group, showing how metric
    values change across different threshold values. If confidence intervals are
    available (i.e., the `lower` and `upper` attributes are not None), these
    intervals are visualized as shaded areas around the lines, providing a visual
    representation of uncertainty for each group's metric values across thresholds.

    Args:
        log_scale: If True, plots both x-axis (threshold values) and y-axis (metric
            values) on a logarithmic scale.
        title: Custom title for the plot. If not provided, a default title is generated
            that includes the DataFrame's index name and indicates whether confidence
            intervals are included in the plot.

    Returns:
        A matplotlib Figure object visualizing metric values for different groups across
        a range of thresholds, with shaded areas representing confidence intervals if
        available.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    thresholds = bias_frame.values.columns.astype(float)

    for group in bias_frame.values.index:
        observed_values = bias_frame.values.loc[group]
        plt.plot(thresholds, observed_values, label=group)

        if bias_frame.lower is not None and bias_frame.upper is not None:
            lower_bound = bias_frame.lower.loc[group]
            upper_bound = bias_frame.upper.loc[group]
            plt.fill_between(thresholds, lower_bound, upper_bound, alpha=0.3)

    plt.xlabel("Threshold values")
    plt.ylabel("Metric values")
    title_with_ci = (
        ""
        if bias_frame.lower is None and bias_frame.upper is None
        else "with Confidence Intervals"
    )
    title = (
        f"Metric values {title_with_ci} by {bias_frame.values.index.name}"
        if title is None
        else title
    )
    plt.title(title, fontsize=16)
    plt.legend(title="Group")
    plt.grid(True)
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    return plt.gcf()
