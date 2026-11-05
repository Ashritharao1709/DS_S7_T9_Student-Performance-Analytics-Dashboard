"""
test_analytics.py — Tests for src.analytics module.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.analytics import (
    total_students,
    column_mean,
    column_median,
    column_std,
    column_min,
    column_max,
    descriptive_summary,
    assign_grades,
    grade_distribution,
    correlation_matrix,
    pairwise_correlation_with_pvalue,
    full_correlation_report,
    shapiro_wilk_test,
    GRADE_BINS,
    GRADE_LABELS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002", "S003", "S004", "S005"],
            "student_name": ["A", "B", "C", "D", "E"],
            "attendance_percentage": [90.0, 80.0, 70.0, 60.0, 50.0],
            "study_hours_per_day": [6.0, 5.0, 4.0, 3.0, 2.0],
            "internal_marks": [45.0, 38.0, 30.0, 22.0, 15.0],
            "final_marks": [92.0, 78.0, 65.0, 48.0, 35.0],
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


@pytest.fixture
def df_with_nan() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002", "S003"],
            "student_name": ["A", "B", "C"],
            "attendance_percentage": [90.0, np.nan, 70.0],
            "study_hours_per_day": [6.0, 5.0, np.nan],
            "internal_marks": [45.0, 38.0, 30.0],
            "final_marks": [92.0, 78.0, 65.0],
        }
    )


# ── Tests: basic statistics ─────────────────────────────────────────────────

class TestBasicStats:
    def test_total_students(self, sample_df: pd.DataFrame) -> None:
        assert total_students(sample_df) == 5

    def test_total_students_empty(self, empty_df: pd.DataFrame) -> None:
        assert total_students(empty_df) == 0

    def test_column_mean(self, sample_df: pd.DataFrame) -> None:
        result = column_mean(sample_df, "attendance_percentage")
        assert result is not None
        assert abs(result - 70.0) < 1e-6

    def test_column_mean_empty(self, empty_df: pd.DataFrame) -> None:
        assert column_mean(empty_df, "attendance_percentage") is None

    def test_column_mean_missing_col(self, sample_df: pd.DataFrame) -> None:
        assert column_mean(sample_df, "nonexistent") is None

    def test_column_median(self, sample_df: pd.DataFrame) -> None:
        result = column_median(sample_df, "attendance_percentage")
        assert result is not None
        assert abs(result - 70.0) < 1e-6

    def test_column_std(self, sample_df: pd.DataFrame) -> None:
        result = column_std(sample_df, "attendance_percentage")
        assert result is not None
        assert result > 0

    def test_column_min(self, sample_df: pd.DataFrame) -> None:
        assert column_min(sample_df, "final_marks") == 35.0

    def test_column_max(self, sample_df: pd.DataFrame) -> None:
        assert column_max(sample_df, "final_marks") == 92.0

    def test_stats_with_nan(self, df_with_nan: pd.DataFrame) -> None:
        """Stats should ignore NaN values gracefully."""
        result = column_mean(df_with_nan, "attendance_percentage")
        assert result is not None
        assert abs(result - 80.0) < 1e-6  # (90 + 70) / 2


# ── Tests: descriptive_summary ──────────────────────────────────────────────

class TestDescriptiveSummary:
    def test_summary_shape(self, sample_df: pd.DataFrame) -> None:
        summary = descriptive_summary(sample_df)
        assert len(summary) == 4  # 4 numeric columns
        expected_cols = {"Column", "Count", "Mean", "Median", "Std Dev", "Min", "Max"}
        assert set(summary.columns) == expected_cols

    def test_summary_empty(self, empty_df: pd.DataFrame) -> None:
        summary = descriptive_summary(empty_df)
        assert summary.empty


# ── Tests: grading ──────────────────────────────────────────────────────────

class TestGrading:
    def test_assign_grades(self, sample_df: pd.DataFrame) -> None:
        grades = assign_grades(sample_df)
        assert len(grades) == 5
        # 92 → A+, 78 → A, 65 → B, 48 → D, 35 → F
        expected = ["A+", "A", "B", "D", "F"]
        assert list(grades) == expected

    def test_assign_grades_empty(self, empty_df: pd.DataFrame) -> None:
        grades = assign_grades(empty_df)
        assert grades.empty

    def test_grade_distribution(self, sample_df: pd.DataFrame) -> None:
        dist = grade_distribution(sample_df)
        assert "Grade" in dist.columns
        assert "Count" in dist.columns
        assert "Percentage" in dist.columns
        assert dist["Count"].sum() == 5

    def test_grade_distribution_percentages(self, sample_df: pd.DataFrame) -> None:
        dist = grade_distribution(sample_df)
        total_pct = dist["Percentage"].sum()
        assert abs(total_pct - 100.0) < 0.5  # allow rounding tolerance


# ── Tests: correlation ──────────────────────────────────────────────────────

class TestCorrelation:
    def test_correlation_matrix_shape(self, sample_df: pd.DataFrame) -> None:
        corr = correlation_matrix(sample_df)
        assert corr.shape == (4, 4)

    def test_correlation_diagonal(self, sample_df: pd.DataFrame) -> None:
        corr = correlation_matrix(sample_df)
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-6

    def test_correlation_empty(self, empty_df: pd.DataFrame) -> None:
        corr = correlation_matrix(empty_df)
        assert corr.empty

    def test_pairwise_correlation(self, sample_df: pd.DataFrame) -> None:
        result = pairwise_correlation_with_pvalue(
            sample_df, "attendance_percentage", "final_marks"
        )
        assert result["r"] is not None
        assert result["p_value"] is not None
        assert -1.0 <= result["r"] <= 1.0
        assert result["p_value"] >= 0.0

    def test_pairwise_missing_column(self, sample_df: pd.DataFrame) -> None:
        result = pairwise_correlation_with_pvalue(
            sample_df, "attendance_percentage", "nonexistent"
        )
        assert result["r"] is None
        assert result["p_value"] is None

    def test_pairwise_insufficient_data(self) -> None:
        """With fewer than 3 valid observations, return None."""
        tiny = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = pairwise_correlation_with_pvalue(tiny, "a", "b")
        assert result["r"] is None

    def test_full_correlation_report(self, sample_df: pd.DataFrame) -> None:
        report = full_correlation_report(sample_df)
        assert len(report) == 6  # C(4,2) = 6 pairs
        assert "r" in report.columns
        assert "p-value" in report.columns


# ── Tests: normality ────────────────────────────────────────────────────────

class TestNormality:
    def test_shapiro_wilk(self, sample_df: pd.DataFrame) -> None:
        result = shapiro_wilk_test(sample_df, "final_marks")
        assert result["statistic"] is not None
        assert result["p_value"] is not None
        assert 0 <= result["statistic"] <= 1

    def test_shapiro_missing_col(self, sample_df: pd.DataFrame) -> None:
        result = shapiro_wilk_test(sample_df, "nonexistent")
        assert result["statistic"] is None

    def test_shapiro_insufficient_data(self) -> None:
        tiny = pd.DataFrame({"x": [1.0]})
        result = shapiro_wilk_test(tiny, "x")
        assert result["statistic"] is None
