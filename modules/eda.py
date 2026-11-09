"""Exploratory data analysis utilities for Adult Census Income data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

sns.set_theme(style="whitegrid", context="notebook")

DEFAULT_TARGET = "income"


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Thiếu cột bắt buộc: {missing_columns}")


def missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages, sorted from high to low."""

    missing_count = frame.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_count.div(len(frame)).mul(100),
        }
    )
    return summary.sort_values(
        ["missing_count", "missing_percent"], ascending=False
    )


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric features."""

    return frame.select_dtypes(include="number").describe().T


def categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return cardinality, most frequent value and frequency for text columns."""

    categorical = frame.select_dtypes(include=["object", "string", "category"])
    if categorical.empty:
        return pd.DataFrame(columns=["count", "unique", "top", "freq"])
    return categorical.describe().T


def dataset_overview(
    frame: pd.DataFrame, target_column: str = DEFAULT_TARGET
) -> dict[str, Any]:
    """Compute the core tables required for the EDA section."""

    _require_columns(frame, [target_column])
    target_counts = frame[target_column].value_counts(dropna=False)
    target_percent = frame[target_column].value_counts(
        dropna=False, normalize=True
    ).mul(100)
    target_distribution = pd.DataFrame(
        {"count": target_counts, "percent": target_percent}
    )

    return {
        "shape": frame.shape,
        "duplicate_rows": int(frame.duplicated().sum()),
        "data_types": frame.dtypes.astype(str).to_frame("dtype"),
        "missing_values": missing_value_summary(frame),
        "target_distribution": target_distribution,
        "numeric_summary": numeric_summary(frame),
        "categorical_summary": categorical_summary(frame),
    }


def plot_target_distribution(
    frame: pd.DataFrame, target_column: str = DEFAULT_TARGET
) -> Figure:
    _require_columns(frame, [target_column])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    sns.countplot(
        data=frame,
        x=target_column,
        hue=target_column,
        legend=False,
        order=frame[target_column].value_counts().index,
        ax=axis,
    )
    total = len(frame)
    for container in axis.containers:
        labels = [
            f"{int(bar.get_height()):,}\n({bar.get_height() / total:.1%})"
            for bar in container
        ]
        axis.bar_label(
            container,
            labels=labels,
            padding=3,
        )
    axis.set(title="Phân bố mức thu nhập", xlabel="Thu nhập", ylabel="Số mẫu")
    figure.tight_layout()
    return figure


def plot_numeric_distributions(
    frame: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    columns: tuple[str, ...] = (
        "age",
        "education_num",
        "hours_per_week",
        "capital_gain",
        "capital_loss",
    ),
) -> Figure:
    selected_columns = [column for column in columns if column in frame.columns]
    _require_columns(frame, [target_column])
    if not selected_columns:
        raise ValueError("Không tìm thấy thuộc tính số để trực quan hóa")

    number_of_columns = 2
    number_of_rows = (len(selected_columns) + number_of_columns - 1) // number_of_columns
    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(13, 4 * number_of_rows),
        squeeze=False,
    )
    for axis, column in zip(axes.flat, selected_columns):
        sns.histplot(
            data=frame,
            x=column,
            hue=target_column,
            bins=30,
            element="step",
            stat="density",
            common_norm=False,
            ax=axis,
        )
        axis.set_title(f"Phân bố {column}")

    for axis in axes.flat[len(selected_columns) :]:
        axis.set_visible(False)
    figure.suptitle("Phân bố các thuộc tính số theo mức thu nhập", y=1.01)
    figure.tight_layout()
    return figure


def plot_categorical_distributions(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = (
        "workclass",
        "education",
        "occupation",
        "marital_status",
        "sex",
        "race",
    ),
    *,
    top_n: int = 12,
) -> Figure:
    selected_columns = [column for column in columns if column in frame.columns]
    if not selected_columns:
        raise ValueError("Không tìm thấy thuộc tính phân loại để trực quan hóa")

    figure, axes = plt.subplots(
        len(selected_columns),
        1,
        figsize=(13, 4 * len(selected_columns)),
        squeeze=False,
    )
    for axis, column in zip(axes.flat, selected_columns):
        order = frame[column].value_counts(dropna=False).head(top_n).index
        sns.countplot(data=frame, y=column, order=order, color="#4C72B0", ax=axis)
        axis.set_title(f"Phân bố {column} (tối đa {top_n} nhóm)")
        axis.set(xlabel="Số mẫu", ylabel=column)
    figure.tight_layout()
    return figure


def plot_numeric_correlation(frame: pd.DataFrame) -> Figure:
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("Cần ít nhất hai thuộc tính số để tính tương quan")

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        numeric.corr(),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        ax=axis,
    )
    axis.set_title("Ma trận tương quan các thuộc tính số")
    figure.tight_layout()
    return figure


def run_eda(
    frame: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> tuple[dict[str, Any], dict[str, Figure]]:
    """Run required EDA and optionally save all figures as PNG files."""

    overview = dataset_overview(frame, target_column)
    figures = {
        "target_distribution": plot_target_distribution(frame, target_column),
        "numeric_distributions": plot_numeric_distributions(frame, target_column),
        "categorical_distributions": plot_categorical_distributions(frame),
        "numeric_correlation": plot_numeric_correlation(frame),
    }

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(destination / f"{name}.png", dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)

    return overview, figures
