"""
test_preprocessing.py — Tests for src.preprocessing module.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.preprocessing import (
    detect_missing_values,
    detect_duplicates,
    detect_out_of_range,
    clean_dataset,
    generate_quality_report,
    VALID_RANGES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small clean DataFrame."""
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002", "S003"],
            "student_name": ["Alice", "Bob", "Carol"],
            "attendance_percentage": [85.0, 70.5, 92.0],
            "study_hours_per_day": [5.0, 3.5, 6.0],
            "internal_marks": [40.0, 30.0, 45.0],
            "final_marks": [80.0, 60.0, 90.0],
        }
    )


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002", "S003"],
            "student_name": ["Alice", None, "Carol"],
            "attendance_percentage": [85.0, np.nan, 92.0],
            "study_hours_per_day": [5.0, 3.5, np.nan],
            "internal_marks": [40.0, 30.0, 45.0],
            "final_marks": [80.0, 60.0, 90.0],
        }
    )


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["S001", "S001", "S002"],
            "student_name": ["Alice", "Alice", "Bob"],
            "attendance_percentage": [85.0, 85.0, 70.5],
            "study_hours_per_day": [5.0, 5.0, 3.5],
            "internal_marks": [40.0, 40.0, 30.0],
            "final_marks": [80.0, 80.0, 60.0],
        }
    )


@pytest.fixture
def df_with_out_of_range() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002"],
            "student_name": ["Alice", "Bob"],
            "attendance_percentage": [105.0, 70.5],  # 105 > 100
            "study_hours_per_day": [-1.0, 3.5],       # -1 < 0
            "internal_marks": [40.0, 55.0],            # 55 > 50
            "final_marks": [80.0, 60.0],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "student_id",
            "student_name",
            "attendance_percentage",
            "study_hours_per_day",
            "internal_marks",
            "final_marks",
        ]
    )


# ── Tests: detect_missing_values ────────────────────────────────────────────

class TestDetectMissing:
    def test_no_missing(self, sample_df: pd.DataFrame) -> None:
        assert detect_missing_values(sample_df) == {}

    def test_with_missing(self, df_with_missing: pd.DataFrame) -> None:
        result = detect_missing_values(df_with_missing)
        assert result["student_name"] == 1
        assert result["attendance_percentage"] == 1
        assert result["study_hours_per_day"] == 1

    def test_empty(self, empty_df: pd.DataFrame) -> None:
        assert detect_missing_values(empty_df) == {}


# ── Tests: detect_duplicates ────────────────────────────────────────────────

class TestDetectDuplicates:
    def test_no_duplicates(self, sample_df: pd.DataFrame) -> None:
        assert detect_duplicates(sample_df) == 0

    def test_with_duplicates(self, df_with_duplicates: pd.DataFrame) -> None:
        assert detect_duplicates(df_with_duplicates) == 1

    def test_empty(self, empty_df: pd.DataFrame) -> None:
        assert detect_duplicates(empty_df) == 0


# ── Tests: detect_out_of_range ──────────────────────────────────────────────

class TestDetectOutOfRange:
    def test_all_in_range(self, sample_df: pd.DataFrame) -> None:
        assert detect_out_of_range(sample_df) == {}

    def test_out_of_range(self, df_with_out_of_range: pd.DataFrame) -> None:
        result = detect_out_of_range(df_with_out_of_range)
        assert result["attendance_percentage"] == 1
        assert result["study_hours_per_day"] == 1
        assert result["internal_marks"] == 1

    def test_empty(self, empty_df: pd.DataFrame) -> None:
        assert detect_out_of_range(empty_df) == {}


# ── Tests: clean_dataset ────────────────────────────────────────────────────

class TestCleanDataset:
    def test_removes_duplicates(self, df_with_duplicates: pd.DataFrame) -> None:
        cleaned = clean_dataset(df_with_duplicates)
        assert len(cleaned) == 2

    def test_clips_out_of_range(self, df_with_out_of_range: pd.DataFrame) -> None:
        cleaned = clean_dataset(df_with_out_of_range)
        assert cleaned["attendance_percentage"].max() <= 100.0
        assert cleaned["study_hours_per_day"].min() >= 0.0
        assert cleaned["internal_marks"].max() <= 50.0

    def test_fills_missing(self, df_with_missing: pd.DataFrame) -> None:
        cleaned = clean_dataset(df_with_missing)
        assert cleaned["attendance_percentage"].isna().sum() == 0
        assert cleaned["study_hours_per_day"].isna().sum() == 0

    def test_does_not_modify_original(self, sample_df: pd.DataFrame) -> None:
        original_len = len(sample_df)
        _ = clean_dataset(sample_df)
        assert len(sample_df) == original_len

    def test_empty(self, empty_df: pd.DataFrame) -> None:
        cleaned = clean_dataset(empty_df)
        assert cleaned.empty


# ── Tests: generate_quality_report ──────────────────────────────────────────

class TestQualityReport:
    def test_clean_data(self, sample_df: pd.DataFrame) -> None:
        cleaned = clean_dataset(sample_df)
        report = generate_quality_report(sample_df, cleaned)
        assert report.total_rows == 3
        assert report.total_missing == 0
        assert report.duplicate_count == 0
        assert report.total_out_of_range == 0

    def test_dirty_data(self, df_with_missing: pd.DataFrame) -> None:
        cleaned = clean_dataset(df_with_missing)
        report = generate_quality_report(df_with_missing, cleaned)
        assert report.total_missing > 0
