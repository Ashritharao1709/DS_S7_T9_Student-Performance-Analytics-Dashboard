"""
data_loader.py — Dataset loading and schema validation for LearnTrack.

Loads the student performance CSV file and validates that it contains the
required columns with the expected data types.  Validation logic is kept
independent of any Streamlit interface so it can be tested and reused
in scripts or notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# ── Required schema ──────────────────────────────────────────────────────────
REQUIRED_COLUMNS: list[str] = [
    "student_id",
    "student_name",
    "attendance_percentage",
    "study_hours_per_day",
    "internal_marks",
    "final_marks",
]

NUMERIC_COLUMNS: list[str] = [
    "attendance_percentage",
    "study_hours_per_day",
    "internal_marks",
    "final_marks",
]

# Default path relative to this file's parent → <project>/data/
_DEFAULT_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV_PATH: Path = _DEFAULT_DATA_DIR / "student_performance.csv"


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or fails validation."""


class SchemaValidationError(Exception):
    """Raised when the loaded dataset does not match the expected schema."""


# ── Public API ───────────────────────────────────────────────────────────────

def load_csv(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load a CSV file and return a raw *DataFrame*.

    Parameters
    ----------
    filepath : Path, optional
        Path to the CSV file.  Defaults to ``data/student_performance.csv``
        relative to the project root.

    Returns
    -------
    pd.DataFrame
        The raw dataframe as read from disk (no cleaning applied).

    Raises
    ------
    DataLoadError
        If the file does not exist, is empty, or cannot be parsed.
    """
    filepath = Path(filepath) if filepath is not None else DEFAULT_CSV_PATH

    if not filepath.exists():
        raise DataLoadError(f"File not found: {filepath}")

    if filepath.stat().st_size == 0:
        raise DataLoadError(f"File is empty: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        raise DataLoadError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise DataLoadError(
            "CSV file was parsed but contains no data rows."
        )

    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that *df* contains the required columns and numeric types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate (typically the output of :func:`load_csv`).

    Returns
    -------
    pd.DataFrame
        The same DataFrame, passed through for chaining convenience.

    Raises
    ------
    SchemaValidationError
        If required columns are missing or numeric columns cannot be coerced.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required columns: {missing}"
        )

    # Attempt to coerce numeric columns; flag failures.
    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_numeric_count = coerced.isna().sum() - df[col].isna().sum()
        if non_numeric_count > 0:
            raise SchemaValidationError(
                f"Column '{col}' contains {non_numeric_count} non-numeric "
                f"value(s) that could not be converted."
            )

    return df


def load_and_validate(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Convenience wrapper: load CSV then validate schema.

    Parameters
    ----------
    filepath : Path, optional
        CSV file path (see :func:`load_csv`).

    Returns
    -------
    pd.DataFrame
        Validated raw DataFrame.
    """
    df = load_csv(filepath)
    validate_schema(df)
    return df
