"""
visualizations.py — Matplotlib / Seaborn chart builders for LearnTrack.

Each function creates a Matplotlib *Figure* and returns it so the caller
(``app.py``) can display it via ``st.pyplot(fig)``.  No Streamlit imports
are used here, keeping the module testable in isolation.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import NUMERIC_COLUMNS
from src.analytics import GRADE_LABELS

# ── Shared style ─────────────────────────────────────────────────────────────
_PALETTE = "coolwarm"
_FIG_BG = "#0E1117"
_AXES_BG = "#1A1D23"
_TEXT_COLOR = "#FAFAFA"
_GRID_COLOR = "#2D3139"

# Grade colours (low → high brightness)
_GRADE_COLORS = {
    "F": "#FF4B4B",
    "D": "#FF8C42",
    "C": "#FFD166",
    "B": "#06D6A0",
    "A": "#118AB2",
    "A+": "#7B2D8E",
}


def _apply_dark_style(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply a dark background and light text to *fig* and *ax*."""
    fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_AXES_BG)
    ax.tick_params(colors=_TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.4, alpha=0.5)


# ── Chart builders ───────────────────────────────────────────────────────────

def attendance_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of attendance percentages with KDE overlay."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _apply_dark_style(fig, ax)

    if not df.empty and "attendance_percentage" in df.columns:
        sns.histplot(
            df["attendance_percentage"],
            bins=15,
            kde=True,
            color="#06D6A0",
            edgecolor="#1A1D23",
            ax=ax,
        )
    ax.set_title("Attendance Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Number of Students")
    fig.tight_layout()
    return fig


def study_hours_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of daily study hours with KDE overlay."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _apply_dark_style(fig, ax)

    if not df.empty and "study_hours_per_day" in df.columns:
        sns.histplot(
            df["study_hours_per_day"],
            bins=12,
            kde=True,
            color="#118AB2",
            edgecolor="#1A1D23",
            ax=ax,
        )
    ax.set_title("Study Hours Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Study Hours per Day")
    ax.set_ylabel("Number of Students")
    fig.tight_layout()
    return fig


def internal_vs_final_marks(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot comparing internal marks against final marks."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_style(fig, ax)

    if (
        not df.empty
        and "internal_marks" in df.columns
        and "final_marks" in df.columns
    ):
        scatter = ax.scatter(
            df["internal_marks"],
            df["final_marks"],
            c=df["final_marks"],
            cmap="viridis",
            edgecolors="#1A1D23",
            s=60,
            alpha=0.85,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Final Marks", color=_TEXT_COLOR)
        cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_COLOR)

        # Trend line
        valid = df[["internal_marks", "final_marks"]].dropna()
        if len(valid) >= 2:
            z = np.polyfit(valid["internal_marks"], valid["final_marks"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(
                valid["internal_marks"].min(), valid["internal_marks"].max(), 100
            )
            ax.plot(x_line, p(x_line), "--", color="#FF8C42", linewidth=1.5, label="Trend")
            ax.legend(facecolor=_AXES_BG, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    ax.set_title("Internal vs Final Marks", fontsize=14, fontweight="bold")
    ax.set_xlabel("Internal Marks")
    ax.set_ylabel("Final Marks")
    fig.tight_layout()
    return fig


def grade_distribution_chart(grade_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of grade distribution.

    Parameters
    ----------
    grade_df : pd.DataFrame
        Must have columns ``Grade`` and ``Count`` (as returned by
        ``analytics.grade_distribution``).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _apply_dark_style(fig, ax)

    if not grade_df.empty and "Grade" in grade_df.columns:
        colors = [_GRADE_COLORS.get(g, "#888888") for g in grade_df["Grade"]]
        bars = ax.bar(
            grade_df["Grade"],
            grade_df["Count"],
            color=colors,
            edgecolor="#1A1D23",
            width=0.55,
        )
        # Value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.3,
                    str(int(height)),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color=_TEXT_COLOR,
                )

    ax.set_title("Grade Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Number of Students")
    fig.tight_layout()
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """Annotated heatmap of the correlation matrix.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square correlation matrix (e.g. from ``analytics.correlation_matrix``).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(_FIG_BG)

    if not corr_matrix.empty:
        # Nicer column labels
        label_map = {
            "attendance_percentage": "Attendance %",
            "study_hours_per_day": "Study Hours",
            "internal_marks": "Internal Marks",
            "final_marks": "Final Marks",
        }
        display_corr = corr_matrix.rename(index=label_map, columns=label_map)

        sns.heatmap(
            display_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.8,
            linecolor=_GRID_COLOR,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            annot_kws={"fontsize": 11, "fontweight": "bold"},
        )
        ax.set_facecolor(_AXES_BG)
        ax.tick_params(colors=_TEXT_COLOR)
        plt.setp(ax.get_xticklabels(), color=_TEXT_COLOR, rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), color=_TEXT_COLOR, rotation=0)
        # Colorbar text
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(colors=_TEXT_COLOR)

    ax.set_title(
        "Correlation Heatmap",
        fontsize=14,
        fontweight="bold",
        color=_TEXT_COLOR,
    )
    fig.tight_layout()
    return fig


def student_comparison_chart(
    df: pd.DataFrame,
    student_ids: list[str],
    metrics: Optional[list[str]] = None,
) -> plt.Figure:
    """Grouped bar chart comparing selected students across metrics.

    Parameters
    ----------
    df : pd.DataFrame
    student_ids : list[str]
        Student IDs to include.
    metrics : list[str], optional
        Numeric columns to compare.  Defaults to all numeric columns.
    """
    metrics = metrics or [c for c in NUMERIC_COLUMNS if c in df.columns]
    subset = df[df["student_id"].isin(student_ids)] if "student_id" in df.columns else df

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, ax)

    if not subset.empty and metrics:
        x = np.arange(len(metrics))
        width = 0.8 / max(len(subset), 1)
        colors = sns.color_palette("Set2", n_colors=len(subset))

        for i, (_, row) in enumerate(subset.iterrows()):
            label = str(row.get("student_name", row.get("student_id", f"Student {i+1}")))
            values = [float(row.get(m, 0)) for m in metrics]
            ax.bar(x + i * width, values, width, label=label, color=colors[i % len(colors)], edgecolor="#1A1D23")

        nice_labels = {
            "attendance_percentage": "Attendance %",
            "study_hours_per_day": "Study Hrs/Day",
            "internal_marks": "Internal Marks",
            "final_marks": "Final Marks",
        }
        ax.set_xticks(x + width * (len(subset) - 1) / 2)
        ax.set_xticklabels([nice_labels.get(m, m) for m in metrics])
        ax.legend(
            facecolor=_AXES_BG,
            edgecolor=_GRID_COLOR,
            labelcolor=_TEXT_COLOR,
            fontsize=9,
            loc="upper right",
        )

    ax.set_title("Student Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig
