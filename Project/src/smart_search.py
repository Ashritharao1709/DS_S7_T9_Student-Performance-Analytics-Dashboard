"""
smart_search.py — Lightweight natural-language query parser for LearnTrack.

Parses plain-English search queries like "students with marks above 80"
into Pandas filter conditions using Python's built-in ``re`` module.

Security: **No** ``eval()``, ``exec()``, or arbitrary code execution.
All filtering is done through safe Pandas boolean indexing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ── Column alias mapping ────────────────────────────────────────────────────
# Maps user-friendly phrases to actual DataFrame column names.
_COLUMN_ALIASES: dict[str, str] = {
    # attendance
    "attendance": "attendance_percentage",
    "attendance percentage": "attendance_percentage",
    "attendance_percentage": "attendance_percentage",
    "attend": "attendance_percentage",
    # study hours
    "study hours": "study_hours_per_day",
    "study_hours": "study_hours_per_day",
    "study_hours_per_day": "study_hours_per_day",
    "study hour": "study_hours_per_day",
    "hours": "study_hours_per_day",
    "study": "study_hours_per_day",
    # internal marks
    "internal marks": "internal_marks",
    "internal_marks": "internal_marks",
    "internal": "internal_marks",
    # final marks  (also the default for bare "marks")
    "final marks": "final_marks",
    "final_marks": "final_marks",
    "final": "final_marks",
    "marks": "final_marks",
    "score": "final_marks",
    "scores": "final_marks",
    # grade
    "grade": "grade",
    "grades": "grade",
}

# Sorted longest-first so multi-word aliases match before single-word ones.
_ALIAS_KEYS_SORTED: list[str] = sorted(
    _COLUMN_ALIASES.keys(), key=len, reverse=True
)

# Operator keywords → symbolic operator
_OPERATOR_MAP: dict[str, str] = {
    "above": ">",
    "over": ">",
    "greater than": ">",
    "more than": ">",
    "higher than": ">",
    "exceeds": ">",
    "exceed": ">",
    "below": "<",
    "under": "<",
    "less than": "<",
    "lower than": "<",
    "equal to": "==",
    "equals": "==",
    "equal": "==",
    "exactly": "==",
    "is": "==",
    "at least": ">=",
    "minimum": ">=",
    "at most": "<=",
    "maximum": "<=",
}

_OPERATOR_KEYS_SORTED: list[str] = sorted(
    _OPERATOR_MAP.keys(), key=len, reverse=True
)

# Valid grade labels (uppercase)
_VALID_GRADES = {"A+", "A", "B", "C", "D", "F"}


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class ParsedCondition:
    """A single parsed filter condition."""

    column: str           # actual DataFrame column name
    operator: str         # one of >, <, >=, <=, ==, between
    value: float | str | None = None
    value_high: float | None = None  # only for "between"
    raw_text: str = ""    # the original fragment that produced this

    def describe(self) -> str:
        """Human-readable description of the condition."""
        col_label = self.column.replace("_", " ").title()
        if self.operator == "between" and self.value_high is not None:
            return f"{col_label} between {self.value} and {self.value_high}"
        if isinstance(self.value, str):
            return f"{col_label} {self.operator} '{self.value}'"
        return f"{col_label} {self.operator} {self.value}"


@dataclass
class SearchResult:
    """Result of a smart search query."""

    success: bool = False
    conditions: list[ParsedCondition] = field(default_factory=list)
    description: str = ""
    error_message: str = ""
    match_count: int = 0
    matches: Optional[pd.DataFrame] = None


# ── Parsing ──────────────────────────────────────────────────────────────────

def _normalise(query: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation edges."""
    query = query.lower().strip().rstrip(".!?")
    query = re.sub(r"\s+", " ", query)
    return query


def _extract_between(query: str) -> list[tuple[str, float, float, str]]:
    """Extract "column between X and Y" patterns.

    Returns list of (column_name, low, high, matched_text).
    """
    results: list[tuple[str, float, float, str]] = []

    # Build a regex-safe alias alternation (longest first)
    alias_pattern = "|".join(re.escape(k) for k in _ALIAS_KEYS_SORTED)

    pattern = re.compile(
        rf"({alias_pattern})\s+(?:between|from)\s+"
        rf"(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )

    for m in pattern.finditer(query):
        alias = m.group(1).lower()
        col = _COLUMN_ALIASES.get(alias)
        if col:
            lo, hi = float(m.group(2)), float(m.group(3))
            if lo > hi:
                lo, hi = hi, lo
            results.append((col, lo, hi, m.group(0)))

    return results


def _extract_grade_conditions(query: str) -> list[tuple[str, str, str]]:
    """Extract grade equality conditions like 'grade A+' or 'grade is B'.

    Returns list of (column_name, grade_value, matched_text).
    """
    results: list[tuple[str, str, str]] = []

    pattern = re.compile(
        r"grade[s]?\s+(?:is\s+|equals?\s+|equal\s+to\s+)?"
        r"([A-Fa-f]\+?)",
        re.IGNORECASE,
    )

    for m in pattern.finditer(query):
        grade_val = m.group(1).upper()
        if grade_val in _VALID_GRADES:
            results.append(("grade", grade_val, m.group(0)))

    return results


def _extract_numeric_conditions(
    query: str,
    already_matched: list[str],
) -> list[tuple[str, str, float, str]]:
    """Extract "column operator value" patterns.

    Parameters
    ----------
    query : str
    already_matched : list[str]
        Substrings already consumed by between/grade parsers (skipped).

    Returns list of (column_name, operator, value, matched_text).
    """
    results: list[tuple[str, str, float, str]] = []

    alias_pattern = "|".join(re.escape(k) for k in _ALIAS_KEYS_SORTED)
    op_pattern = "|".join(re.escape(k) for k in _OPERATOR_KEYS_SORTED)

    pattern = re.compile(
        rf"({alias_pattern})\s+({op_pattern})\s+(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )

    for m in pattern.finditer(query):
        # Skip if this overlaps with an already-matched span
        if any(m.group(0) in am for am in already_matched):
            continue

        alias = m.group(1).lower()
        op_text = m.group(2).lower()
        col = _COLUMN_ALIASES.get(alias)
        op = _OPERATOR_MAP.get(op_text)
        if col and op and col != "grade":
            results.append((col, op, float(m.group(3)), m.group(0)))

    # Also try reversed order: "operator value column"
    # e.g. "above 80 marks" / "more than 5 hours"
    rev_pattern = re.compile(
        rf"({op_pattern})\s+(\d+(?:\.\d+)?)\s+({alias_pattern})",
        re.IGNORECASE,
    )
    for m in rev_pattern.finditer(query):
        if any(m.group(0) in am for am in already_matched):
            continue
        op_text = m.group(1).lower()
        alias = m.group(3).lower()
        col = _COLUMN_ALIASES.get(alias)
        op = _OPERATOR_MAP.get(op_text)
        if col and op and col != "grade":
            # Avoid duplicate if same column+operator+value already found
            entry = (col, op, float(m.group(2)), m.group(0))
            if not any(r[:3] == entry[:3] for r in results):
                results.append(entry)

    return results


def parse_query(query: str) -> list[ParsedCondition]:
    """Parse a natural-language query into a list of filter conditions.

    Supports:
    - ``column above/below/equal/... value``
    - ``column between X and Y``
    - ``grade A/B/C/D/F/A+``
    - Multiple conditions combined with AND logic.

    Parameters
    ----------
    query : str
        The user's search query.

    Returns
    -------
    list[ParsedCondition]
        Parsed conditions (may be empty if nothing could be understood).
    """
    normalised = _normalise(query)
    conditions: list[ParsedCondition] = []
    consumed: list[str] = []

    # 1 — between
    for col, lo, hi, raw in _extract_between(normalised):
        conditions.append(
            ParsedCondition(column=col, operator="between",
                            value=lo, value_high=hi, raw_text=raw)
        )
        consumed.append(raw)

    # 2 — grade
    for col, grade_val, raw in _extract_grade_conditions(normalised):
        conditions.append(
            ParsedCondition(column=col, operator="==",
                            value=grade_val, raw_text=raw)
        )
        consumed.append(raw)

    # 3 — numeric comparisons
    for col, op, val, raw in _extract_numeric_conditions(normalised, consumed):
        conditions.append(
            ParsedCondition(column=col, operator=op,
                            value=val, raw_text=raw)
        )

    return conditions


# ── Filtering ────────────────────────────────────────────────────────────────

def apply_conditions(
    df: pd.DataFrame,
    conditions: list[ParsedCondition],
) -> pd.DataFrame:
    """Apply parsed conditions to *df* using AND logic.

    Parameters
    ----------
    df : pd.DataFrame
    conditions : list[ParsedCondition]

    Returns
    -------
    pd.DataFrame
        Filtered subset of *df*.
    """
    if df.empty or not conditions:
        return df

    mask = pd.Series(True, index=df.index)

    for cond in conditions:
        if cond.column not in df.columns:
            continue

        series = df[cond.column]

        if cond.operator == "between" and cond.value_high is not None:
            mask &= series.between(float(cond.value), float(cond.value_high))
        elif cond.operator == "==" and isinstance(cond.value, str):
            # Grade comparison (case-insensitive)
            mask &= series.astype(str).str.upper() == cond.value.upper()
        elif cond.operator == ">":
            mask &= series > float(cond.value)
        elif cond.operator == "<":
            mask &= series < float(cond.value)
        elif cond.operator == ">=":
            mask &= series >= float(cond.value)
        elif cond.operator == "<=":
            mask &= series <= float(cond.value)
        elif cond.operator == "==":
            mask &= series == float(cond.value)

    return df[mask]


# ── Public high-level API ───────────────────────────────────────────────────

def smart_search(df: pd.DataFrame, query: str) -> SearchResult:
    """Parse *query* and filter *df*, returning a :class:`SearchResult`.

    Parameters
    ----------
    df : pd.DataFrame
        The student dataset (cleaned, with a ``grade`` column).
    query : str
        Natural-language search query.

    Returns
    -------
    SearchResult
    """
    query = query.strip()
    if not query:
        return SearchResult(
            success=False,
            error_message="Please enter a search query.",
        )

    conditions = parse_query(query)

    if not conditions:
        return SearchResult(
            success=False,
            error_message=(
                "Sorry, I couldn't understand that query. "
                "Try something like: \"students with marks above 80\" "
                "or \"attendance below 75\"."
            ),
        )

    desc_parts = [c.describe() for c in conditions]
    description = " AND ".join(desc_parts)

    matches = apply_conditions(df, conditions)

    return SearchResult(
        success=True,
        conditions=conditions,
        description=description,
        match_count=len(matches),
        matches=matches,
    )


# ── Example queries (for UI display) ────────────────────────────────────────

EXAMPLE_QUERIES: list[str] = [
    "Show students with marks above 80",
    "Show students with marks below 40",
    "Show students with attendance below 75",
    "Show students who study more than 5 hours",
    "Show students with grade A",
    "Show students with marks between 60 and 80",
    "Show students with attendance above 85 and marks below 60",
    "Show students with grade A+",
    "Show students with internal marks above 40",
    "Show students with attendance at least 90",
]
