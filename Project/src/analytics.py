"""
analytics.py — Descriptive statistics, grading, and correlation analysis.

All functions accept a DataFrame (or Series) and return computed results.
They handle empty datasets and missing values safely.

SciPy is used for Pearson correlation with p-values and for normality
testing (Shapiro–Wilk), which go beyond what Pandas provides natively.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import NUMERIC_COLUMNS

# ── Grade thresholds ─────────────────────────────────────────────────────────
# Based on *final_marks* out of 100.
GRADE_BINS: list[float] = [0, 40, 50, 60, 75, 90, 100]
GRADE_LABELS: list[str] = ["F", "D", "C", "B", "A", "A+"]


# ── Descriptive statistics ───────────────────────────────────────────────────

def total_students(df: pd.DataFrame) -> int:
    """Return the number of student records."""
    return len(df)


def column_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the mean of *col*, or ``None`` if the column is empty/missing."""
    if col not in df.columns or df[col].dropna().empty:
        return None
    return float(df[col].mean())


def column_median(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the median of *col*."""
    if col not in df.columns or df[col].dropna().empty:
        return None
    return float(df[col].median())


def column_std(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the sample standard deviation of *col*."""
    if col not in df.columns or df[col].dropna().empty:
        return None
    return float(df[col].std())


def column_min(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the minimum of *col*."""
    if col not in df.columns or df[col].dropna().empty:
        return None
    return float(df[col].min())


def column_max(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the maximum of *col*."""
    if col not in df.columns or df[col].dropna().empty:
        return None
    return float(df[col].max())


def descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a descriptive-statistics table for all numeric columns.

    Columns: mean, median, std, min, max, count.
    """
    if df.empty:
        return pd.DataFrame()

    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    rows = []
    for col in numeric_cols:
        rows.append(
            {
                "Column": col,
                "Count": int(df[col].count()),
                "Mean": column_mean(df, col),
                "Median": column_median(df, col),
                "Std Dev": column_std(df, col),
                "Min": column_min(df, col),
                "Max": column_max(df, col),
            }
        )
    return pd.DataFrame(rows)


# ── Grading ──────────────────────────────────────────────────────────────────

def assign_grades(
    df: pd.DataFrame,
    col: str = "final_marks",
    bins: Optional[list[float]] = None,
    labels: Optional[list[str]] = None,
) -> pd.Series:
    """Assign letter grades based on *col* using bin thresholds.

    Grading scale (default, based on ``final_marks`` out of 100):
        A+ : 90–100
        A  : 75–89
        B  : 60–74
        C  : 50–59
        D  : 40–49
        F  : 0–39

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    bins : list[float], optional
    labels : list[str], optional

    Returns
    -------
    pd.Series
        Categorical series of grade labels.
    """
    bins = bins or GRADE_BINS
    labels = labels or GRADE_LABELS

    if col not in df.columns or df.empty:
        return pd.Series(dtype="category")

    return pd.cut(
        df[col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )


def grade_distribution(
    df: pd.DataFrame,
    col: str = "final_marks",
) -> pd.DataFrame:
    """Return a DataFrame with grade labels and their counts/percentages.

    Parameters
    ----------
    df : pd.DataFrame
    col : str

    Returns
    -------
    pd.DataFrame
        Columns: Grade, Count, Percentage.
    """
    grades = assign_grades(df, col=col)
    if grades.empty:
        return pd.DataFrame(columns=["Grade", "Count", "Percentage"])

    counts = grades.value_counts().reindex(GRADE_LABELS, fill_value=0)
    total = counts.sum()
    pct = (counts / total * 100).round(1) if total > 0 else counts * 0
    return pd.DataFrame(
        {"Grade": counts.index, "Count": counts.values, "Percentage": pct.values}
    )


# ── Correlation ──────────────────────────────────────────────────────────────

def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute the Pearson correlation matrix for *columns*.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str], optional
        Defaults to :data:`NUMERIC_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix.
    """
    columns = columns or [c for c in NUMERIC_COLUMNS if c in df.columns]
    if df.empty or not columns:
        return pd.DataFrame()
    return df[columns].corr(method="pearson")


def pairwise_correlation_with_pvalue(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict[str, Optional[float]]:
    """Return Pearson *r* and its *p*-value for two columns (via SciPy).

    Uses ``scipy.stats.pearsonr`` which provides the two-tailed p-value
    for testing non-correlation — information Pandas' ``.corr()`` does not
    supply.

    Parameters
    ----------
    df : pd.DataFrame
    col_a, col_b : str

    Returns
    -------
    dict
        ``{"r": float | None, "p_value": float | None}``
    """
    if col_a not in df.columns or col_b not in df.columns:
        return {"r": None, "p_value": None}

    valid = df[[col_a, col_b]].dropna()
    if len(valid) < 3:
        return {"r": None, "p_value": None}

    r, p = stats.pearsonr(valid[col_a], valid[col_b])
    return {"r": round(float(r), 4), "p_value": round(float(p), 6)}


def full_correlation_report(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Correlation matrix with p-values for every pair.

    Returns a DataFrame with columns: Variable 1, Variable 2, r, p-value.
    """
    columns = columns or [c for c in NUMERIC_COLUMNS if c in df.columns]
    if df.empty or len(columns) < 2:
        return pd.DataFrame(columns=["Variable 1", "Variable 2", "r", "p-value"])

    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            result = pairwise_correlation_with_pvalue(df, a, b)
            rows.append(
                {
                    "Variable 1": a,
                    "Variable 2": b,
                    "r": result["r"],
                    "p-value": result["p_value"],
                }
            )
    return pd.DataFrame(rows)


# ── Normality test (SciPy) ──────────────────────────────────────────────────

def shapiro_wilk_test(
    df: pd.DataFrame,
    col: str,
) -> dict[str, Optional[float]]:
    """Shapiro–Wilk test of normality for *col*.

    Useful for checking whether parametric assumptions (used in Pearson r)
    are reasonable for the sample.

    Parameters
    ----------
    df : pd.DataFrame
    col : str

    Returns
    -------
    dict
        ``{"statistic": float | None, "p_value": float | None}``
    """
    if col not in df.columns:
        return {"statistic": None, "p_value": None}

    data = df[col].dropna()
    if len(data) < 3:
        return {"statistic": None, "p_value": None}

    stat, p = stats.shapiro(data)
    return {"statistic": round(float(stat), 4), "p_value": round(float(p), 6)}
