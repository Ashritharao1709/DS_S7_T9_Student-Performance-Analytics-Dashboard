"""
test_smart_search.py — Tests for src.smart_search module.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.smart_search import (
    parse_query,
    apply_conditions,
    smart_search,
    ParsedCondition,
    SearchResult,
    EXAMPLE_QUERIES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def student_df() -> pd.DataFrame:
    """A small student dataset with grades assigned."""
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002", "S003", "S004", "S005"],
            "student_name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "attendance_percentage": [92.0, 70.0, 85.0, 55.0, 96.0],
            "study_hours_per_day": [6.0, 3.0, 5.0, 2.0, 7.0],
            "internal_marks": [44.0, 28.0, 38.0, 20.0, 46.0],
            "final_marks": [90.0, 58.0, 78.0, 38.0, 95.0],
            "grade": ["A+", "C", "A", "F", "A+"],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "student_id", "student_name", "attendance_percentage",
            "study_hours_per_day", "internal_marks", "final_marks", "grade",
        ]
    )


# ── Tests: parse_query ──────────────────────────────────────────────────────

class TestParseQuery:
    """Tests for the query parser."""

    def test_marks_above(self) -> None:
        conditions = parse_query("students with marks above 80")
        assert len(conditions) == 1
        assert conditions[0].column == "final_marks"
        assert conditions[0].operator == ">"
        assert conditions[0].value == 80.0

    def test_attendance_below(self) -> None:
        conditions = parse_query("attendance below 75")
        assert len(conditions) == 1
        assert conditions[0].column == "attendance_percentage"
        assert conditions[0].operator == "<"
        assert conditions[0].value == 75.0

    def test_study_hours_more_than(self) -> None:
        conditions = parse_query("study hours more than 5")
        assert len(conditions) == 1
        assert conditions[0].column == "study_hours_per_day"
        assert conditions[0].operator == ">"
        assert conditions[0].value == 5.0

    def test_marks_between(self) -> None:
        conditions = parse_query("marks between 60 and 80")
        assert len(conditions) == 1
        assert conditions[0].column == "final_marks"
        assert conditions[0].operator == "between"
        assert conditions[0].value == 60.0
        assert conditions[0].value_high == 80.0

    def test_grade_a(self) -> None:
        conditions = parse_query("students with grade A")
        assert len(conditions) == 1
        assert conditions[0].column == "grade"
        assert conditions[0].operator == "=="
        assert conditions[0].value == "A"

    def test_grade_a_plus(self) -> None:
        conditions = parse_query("grade A+")
        assert len(conditions) == 1
        assert conditions[0].value == "A+"

    def test_internal_marks(self) -> None:
        conditions = parse_query("internal marks above 40")
        assert len(conditions) == 1
        assert conditions[0].column == "internal_marks"
        assert conditions[0].operator == ">"

    def test_equal_to(self) -> None:
        conditions = parse_query("final marks equal to 80")
        assert len(conditions) == 1
        assert conditions[0].operator == "=="
        assert conditions[0].value == 80.0

    def test_at_least(self) -> None:
        conditions = parse_query("attendance at least 90")
        assert len(conditions) == 1
        assert conditions[0].operator == ">="
        assert conditions[0].value == 90.0

    def test_multiple_conditions(self) -> None:
        conditions = parse_query(
            "attendance above 85 and marks below 60"
        )
        assert len(conditions) == 2
        cols = {c.column for c in conditions}
        assert "attendance_percentage" in cols
        assert "final_marks" in cols

    def test_between_reversed_values(self) -> None:
        """If user says 'between 80 and 60', values should be swapped."""
        conditions = parse_query("marks between 80 and 60")
        assert len(conditions) == 1
        assert conditions[0].value == 60.0
        assert conditions[0].value_high == 80.0

    def test_invalid_query_returns_empty(self) -> None:
        conditions = parse_query("hello world")
        assert conditions == []

    def test_empty_query(self) -> None:
        conditions = parse_query("")
        assert conditions == []

    def test_decimal_values(self) -> None:
        conditions = parse_query("study hours above 4.5")
        assert len(conditions) == 1
        assert conditions[0].value == 4.5


# ── Tests: apply_conditions ─────────────────────────────────────────────────

class TestApplyConditions:
    """Tests for applying parsed conditions to a DataFrame."""

    def test_greater_than(self, student_df: pd.DataFrame) -> None:
        conds = [ParsedCondition(column="final_marks", operator=">", value=80.0)]
        result = apply_conditions(student_df, conds)
        assert len(result) == 2  # Alice(90), Eve(95)

    def test_less_than(self, student_df: pd.DataFrame) -> None:
        conds = [ParsedCondition(column="final_marks", operator="<", value=60.0)]
        result = apply_conditions(student_df, conds)
        assert len(result) == 2  # Bob(58), Dave(38)

    def test_equal(self, student_df: pd.DataFrame) -> None:
        conds = [ParsedCondition(column="final_marks", operator="==", value=78.0)]
        result = apply_conditions(student_df, conds)
        assert len(result) == 1  # Carol

    def test_between(self, student_df: pd.DataFrame) -> None:
        conds = [ParsedCondition(
            column="final_marks", operator="between",
            value=60.0, value_high=80.0,
        )]
        result = apply_conditions(student_df, conds)
        # Bob is 58 which is < 60, so excluded.  Only Carol(78) matches.
        assert len(result) == 1  # Carol(78)

    def test_grade_filter(self, student_df: pd.DataFrame) -> None:
        conds = [ParsedCondition(column="grade", operator="==", value="A+")]
        result = apply_conditions(student_df, conds)
        assert len(result) == 2  # Alice, Eve

    def test_multiple_conditions_and(self, student_df: pd.DataFrame) -> None:
        conds = [
            ParsedCondition(column="attendance_percentage", operator=">", value=80.0),
            ParsedCondition(column="final_marks", operator=">", value=80.0),
        ]
        result = apply_conditions(student_df, conds)
        # Alice(att=92,fm=90), Carol(att=85,fm=78 not>80), Eve(att=96,fm=95)
        assert len(result) == 2  # Alice, Eve

    def test_no_matches(self, student_df: pd.DataFrame) -> None:
        conds = [ParsedCondition(column="final_marks", operator=">", value=99.0)]
        result = apply_conditions(student_df, conds)
        assert len(result) == 0

    def test_empty_conditions(self, student_df: pd.DataFrame) -> None:
        result = apply_conditions(student_df, [])
        assert len(result) == len(student_df)

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        conds = [ParsedCondition(column="final_marks", operator=">", value=50.0)]
        result = apply_conditions(empty_df, conds)
        assert len(result) == 0

    def test_missing_column_ignored(self, student_df: pd.DataFrame) -> None:
        conds = [ParsedCondition(column="nonexistent_col", operator=">", value=50.0)]
        result = apply_conditions(student_df, conds)
        # Missing column condition is skipped, so all rows match
        assert len(result) == len(student_df)


# ── Tests: smart_search (integration) ───────────────────────────────────────

class TestSmartSearch:
    """End-to-end tests for the smart_search function."""

    def test_successful_search(self, student_df: pd.DataFrame) -> None:
        result = smart_search(student_df, "marks above 80")
        assert result.success is True
        assert result.match_count == 2
        assert result.matches is not None
        assert len(result.conditions) == 1
        assert result.description != ""

    def test_no_matches(self, student_df: pd.DataFrame) -> None:
        result = smart_search(student_df, "marks above 99")
        assert result.success is True
        assert result.match_count == 0

    def test_invalid_query(self, student_df: pd.DataFrame) -> None:
        result = smart_search(student_df, "tell me a joke")
        assert result.success is False
        assert result.error_message != ""

    def test_empty_query(self, student_df: pd.DataFrame) -> None:
        result = smart_search(student_df, "")
        assert result.success is False
        assert "enter" in result.error_message.lower()

    def test_whitespace_only_query(self, student_df: pd.DataFrame) -> None:
        result = smart_search(student_df, "   ")
        assert result.success is False

    def test_grade_search(self, student_df: pd.DataFrame) -> None:
        result = smart_search(student_df, "students with grade F")
        assert result.success is True
        assert result.match_count == 1
        assert result.matches is not None
        assert result.matches.iloc[0]["student_name"] == "Dave"

    def test_multi_condition_search(self, student_df: pd.DataFrame) -> None:
        result = smart_search(
            student_df,
            "attendance above 85 and marks below 95"
        )
        assert result.success is True
        assert result.match_count >= 1

    def test_between_search(self, student_df: pd.DataFrame) -> None:
        result = smart_search(student_df, "marks between 50 and 90")
        assert result.success is True
        assert result.match_count >= 1

    def test_description_contains_and(self, student_df: pd.DataFrame) -> None:
        result = smart_search(
            student_df,
            "attendance above 80 and marks above 70"
        )
        assert result.success is True
        assert "AND" in result.description

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = smart_search(empty_df, "marks above 80")
        assert result.success is True
        assert result.match_count == 0

    def test_example_queries_parse(self, student_df: pd.DataFrame) -> None:
        """All example queries listed in the UI should parse successfully."""
        for query in EXAMPLE_QUERIES:
            result = smart_search(student_df, query)
            assert result.success is True, (
                f"Example query failed to parse: '{query}'"
            )


# ── Tests: Streamlit UI App Integration ─────────────────────────────────────

class TestSmartSearchAppIntegration:
    """Verify that quick-example buttons safely update widget session state."""

    def test_example_button_click_updates_search_input(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=30).run()
        assert len(at.exception) == 0

        # Click the first quick-example button
        btn = at.button(key="ex_btn_0")
        assert btn is not None
        at = btn.click().run()

        # Must not raise StreamlitWidgetAlreadyInstantiatedError
        assert len(at.exception) == 0

        # Input widget must be populated with the example query
        input_widget = at.text_input(key="smart_search_input")
        assert input_widget.value == EXAMPLE_QUERIES[0]

    def test_more_examples_button_click(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=30).run()
        assert len(at.exception) == 0

        # Click an example from the expanded section
        btn = at.button(key="ex_btn_6")
        assert btn is not None
        at = btn.click().run()

        assert len(at.exception) == 0
        input_widget = at.text_input(key="smart_search_input")
        assert input_widget.value == EXAMPLE_QUERIES[6]

    def test_manual_text_input_executes(self) -> None:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("app.py", default_timeout=30).run()
        assert len(at.exception) == 0

        input_widget = at.text_input(key="smart_search_input")
        at = input_widget.input("marks above 80").run()

        assert len(at.exception) == 0
        assert at.text_input(key="smart_search_input").value == "marks above 80"
