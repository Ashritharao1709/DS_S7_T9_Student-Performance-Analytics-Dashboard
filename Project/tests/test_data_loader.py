"""
test_data_loader.py — Tests for src.data_loader module.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    DataLoadError,
    SchemaValidationError,
    load_csv,
    validate_schema,
    load_and_validate,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    """Create a minimal valid CSV file."""
    fp = tmp_path / "valid.csv"
    rows = [
        {
            "student_id": "S001",
            "student_name": "Alice",
            "attendance_percentage": 85.0,
            "study_hours_per_day": 5.0,
            "internal_marks": 40,
            "final_marks": 80,
        },
        {
            "student_id": "S002",
            "student_name": "Bob",
            "attendance_percentage": 70.5,
            "study_hours_per_day": 3.5,
            "internal_marks": 30,
            "final_marks": 60,
        },
    ]
    with fp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return fp


@pytest.fixture
def empty_csv(tmp_path: Path) -> Path:
    """Create an empty file (0 bytes)."""
    fp = tmp_path / "empty.csv"
    fp.write_text("")
    return fp


@pytest.fixture
def headers_only_csv(tmp_path: Path) -> Path:
    """CSV with headers but no data rows."""
    fp = tmp_path / "headers_only.csv"
    fp.write_text(",".join(REQUIRED_COLUMNS) + "\n")
    return fp


@pytest.fixture
def missing_column_csv(tmp_path: Path) -> Path:
    """CSV missing the 'final_marks' column."""
    fp = tmp_path / "missing_col.csv"
    cols = [c for c in REQUIRED_COLUMNS if c != "final_marks"]
    fp.write_text(",".join(cols) + "\nS001,Alice,80,5,35\n")
    return fp


@pytest.fixture
def non_numeric_csv(tmp_path: Path) -> Path:
    """CSV with non-numeric value in a numeric column."""
    fp = tmp_path / "bad_num.csv"
    fp.write_text(
        ",".join(REQUIRED_COLUMNS) + "\n"
        "S001,Alice,eighty,5,35,70\n"
    )
    return fp


# ── Tests ────────────────────────────────────────────────────────────────────

class TestLoadCSV:
    """Tests for the load_csv function."""

    def test_load_valid_csv(self, valid_csv: Path) -> None:
        df = load_csv(valid_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="File not found"):
            load_csv(tmp_path / "nonexistent.csv")

    def test_empty_file(self, empty_csv: Path) -> None:
        with pytest.raises(DataLoadError, match="empty"):
            load_csv(empty_csv)

    def test_headers_only(self, headers_only_csv: Path) -> None:
        with pytest.raises(DataLoadError, match="no data rows"):
            load_csv(headers_only_csv)


class TestValidateSchema:
    """Tests for the validate_schema function."""

    def test_valid_schema(self, valid_csv: Path) -> None:
        df = load_csv(valid_csv)
        result = validate_schema(df)
        assert list(result.columns) == list(df.columns)

    def test_missing_required_column(self, missing_column_csv: Path) -> None:
        df = load_csv(missing_column_csv)
        with pytest.raises(SchemaValidationError, match="Missing required columns"):
            validate_schema(df)

    def test_non_numeric_column(self, non_numeric_csv: Path) -> None:
        df = load_csv(non_numeric_csv)
        with pytest.raises(SchemaValidationError, match="non-numeric"):
            validate_schema(df)


class TestLoadAndValidate:
    """Integration tests for load_and_validate."""

    def test_success(self, valid_csv: Path) -> None:
        df = load_and_validate(valid_csv)
        assert len(df) == 2
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_default_dataset_loads(self) -> None:
        """The bundled synthetic dataset should always load successfully."""
        df = load_and_validate()
        assert len(df) >= 10  # we created 50 records
        for col in REQUIRED_COLUMNS:
            assert col in df.columns
