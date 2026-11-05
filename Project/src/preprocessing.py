"""
preprocessing.py — Data-cleaning and quality-check utilities for LearnTrack.

All functions operate on a DataFrame and return information about data quality
or a cleaned copy.  The original DataFrame is never modified in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import NUMERIC_COLUMNS

# ── Valid numeric ranges ─────────────────────────────────────────────────────
# Mapping of column name → (min_allowed, max_allowed).
VALID_RANGES: dict[str, tuple[float, float]] = {
    "attendance_percentage": (0.0, 100.0),
    "study_hours_per_day": (0.0, 24.0),
    "internal_marks": (0.0, 50.0),
    "final_marks": (0.0, 100.0),
}


@dataclass
class DataQualityReport:
    """Summary of data-quality issues found during preprocessing."""

    total_rows: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    total_missing: int = 0
    duplicate_count: int = 0
    out_of_range: dict[str, int] = field(default_factory=dict)
    total_out_of_range: int = 0
    rows_after_cleaning: int = 0


# ── Detection helpers ────────────────────────────────────────────────────────

def detect_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return a dict mapping column name → count of missing (NaN/None) values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict[str, int]
        Only columns with at least one missing value are included.
    """
    counts = df.isna().sum()
    return {col: int(n) for col, n in counts.items() if n > 0}


def detect_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> int:
    """Return the number of fully duplicated rows (or duplicates on *subset*).

    Parameters
    ----------
    df : pd.DataFrame
    subset : list[str], optional
        Columns to consider for duplicate detection.  Defaults to all columns.

    Returns
    -------
    int
    """
    return int(df.duplicated(subset=subset).sum())


def detect_out_of_range(
    df: pd.DataFrame,
    ranges: Optional[dict[str, tuple[float, float]]] = None,
) -> dict[str, int]:
    """Identify rows with numeric values outside allowed ranges.

    Parameters
    ----------
    df : pd.DataFrame
    ranges : dict, optional
        ``{column: (min, max)}``.  Defaults to :data:`VALID_RANGES`.

    Returns
    -------
    dict[str, int]
        Column → count of out-of-range values.
    """
    ranges = ranges or VALID_RANGES
    result: dict[str, int] = {}
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        bad = ((series < lo) | (series > hi)) & series.notna()
        count = int(bad.sum())
        if count > 0:
            result[col] = count
    return result


# ── Cleaning ─────────────────────────────────────────────────────────────────

def clean_dataset(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    coerce_numeric: bool = True,
    clip_ranges: bool = True,
    fill_missing_numeric: str = "median",
) -> pd.DataFrame:
    """Return a cleaned copy of *df*.

    Steps (in order):
    1. Drop fully duplicated rows (optional).
    2. Coerce numeric columns to proper types (optional).
    3. Clip values to allowed ranges (optional).
    4. Fill remaining missing numeric values with column median/mean (optional).

    The original DataFrame is **not** modified.

    Parameters
    ----------
    df : pd.DataFrame
    drop_duplicates : bool
    coerce_numeric : bool
    clip_ranges : bool
    fill_missing_numeric : str
        ``"median"`` or ``"mean"``.

    Returns
    -------
    pd.DataFrame
        Cleaned copy.
    """
    cleaned = df.copy()

    # 1 — duplicates
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2 — coerce
    if coerce_numeric:
        for col in NUMERIC_COLUMNS:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # 3 — clip
    if clip_ranges:
        for col, (lo, hi) in VALID_RANGES.items():
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].clip(lower=lo, upper=hi)

    # 4 — fill missing
    if fill_missing_numeric:
        for col in NUMERIC_COLUMNS:
            if col in cleaned.columns and cleaned[col].isna().any():
                if fill_missing_numeric == "median":
                    fill_val = cleaned[col].median()
                else:
                    fill_val = cleaned[col].mean()
                cleaned[col] = cleaned[col].fillna(fill_val)

    return cleaned


def generate_quality_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> DataQualityReport:
    """Create a :class:`DataQualityReport` comparing raw and cleaned data.

    Parameters
    ----------
    raw_df : pd.DataFrame
    cleaned_df : pd.DataFrame

    Returns
    -------
    DataQualityReport
    """
    missing = detect_missing_values(raw_df)
    oor = detect_out_of_range(raw_df)
    return DataQualityReport(
        total_rows=len(raw_df),
        missing_values=missing,
        total_missing=sum(missing.values()),
        duplicate_count=detect_duplicates(raw_df),
        out_of_range=oor,
        total_out_of_range=sum(oor.values()),
        rows_after_cleaning=len(cleaned_df),
    )
