"""
app.py — LearnTrack: Student Performance Analytics Dashboard.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit

import pandas as pd
import streamlit as st

# ── Ensure project root is on sys.path so ``src.*`` imports work ────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_validate, DataLoadError, SchemaValidationError
from src.preprocessing import clean_dataset, generate_quality_report
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
    full_correlation_report,
    shapiro_wilk_test,
    GRADE_LABELS,
)
from src.smart_search import smart_search, EXAMPLE_QUERIES
from src.visualizations import (
    attendance_distribution,
    study_hours_distribution,
    internal_vs_final_marks,
    grade_distribution_chart,
    correlation_heatmap,
    student_comparison_chart,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LearnTrack – Student Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* KPI card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d23 0%, #262a33 100%);
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #9ca3af !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
    }

    /* Section headings */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #fafafa;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #7B2D8E;
    }

    /* Tabs styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
    }

    /* Sidebar branding */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #161b22 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading (cached) ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    """Load and validate the CSV, returning the raw DataFrame."""
    return load_and_validate()


@st.cache_data(show_spinner="Cleaning data…")
def get_cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    return clean_dataset(raw)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    # Header
    st.markdown("# 📊 LearnTrack")
    st.markdown(
        "*Interactive analytics dashboard for student academic performance — "
        "powered by attendance, study-hours, and examination data.*"
    )
    st.markdown("---")

    # Load data
    try:
        raw_df = load_data()
    except (DataLoadError, SchemaValidationError) as exc:
        st.error(f"**Data Error:** {exc}")
        st.stop()

    cleaned_df = get_cleaned(raw_df)

    # Assign grades
    cleaned_df = cleaned_df.copy()
    cleaned_df["grade"] = assign_grades(cleaned_df)

    # ── Sidebar filters ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎛️ Filters")
        st.caption("Adjust filters to explore subsets of the data.")

        # Attendance range
        att_min = float(cleaned_df["attendance_percentage"].min())
        att_max = float(cleaned_df["attendance_percentage"].max())
        att_range = st.slider(
            "Attendance (%)",
            min_value=att_min,
            max_value=att_max,
            value=(att_min, att_max),
            step=0.5,
        )

        # Study hours range
        sh_min = float(cleaned_df["study_hours_per_day"].min())
        sh_max = float(cleaned_df["study_hours_per_day"].max())
        sh_range = st.slider(
            "Study Hours / Day",
            min_value=sh_min,
            max_value=sh_max,
            value=(sh_min, sh_max),
            step=0.1,
        )

        # Final marks range
        fm_min = float(cleaned_df["final_marks"].min())
        fm_max = float(cleaned_df["final_marks"].max())
        fm_range = st.slider(
            "Final Marks",
            min_value=fm_min,
            max_value=fm_max,
            value=(fm_min, fm_max),
            step=1.0,
        )

        # Grade filter
        available_grades = sorted(
            cleaned_df["grade"].dropna().unique().tolist(),
            key=lambda g: GRADE_LABELS.index(g) if g in GRADE_LABELS else 99,
        )
        selected_grades = st.multiselect(
            "Grades",
            options=available_grades,
            default=available_grades,
        )

        st.markdown("---")
        st.caption("💡 Filters update all metrics, charts, and tables below.")

    # Apply filters
    filtered = cleaned_df[
        (cleaned_df["attendance_percentage"] >= att_range[0])
        & (cleaned_df["attendance_percentage"] <= att_range[1])
        & (cleaned_df["study_hours_per_day"] >= sh_range[0])
        & (cleaned_df["study_hours_per_day"] <= sh_range[1])
        & (cleaned_df["final_marks"] >= fm_range[0])
        & (cleaned_df["final_marks"] <= fm_range[1])
        & (cleaned_df["grade"].isin(selected_grades))
    ]

    # ── KPI Cards ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📈 Key Performance Indicators</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Students", total_students(filtered))
    with k2:
        avg_att = column_mean(filtered, "attendance_percentage")
        st.metric("Avg Attendance", f"{avg_att:.1f}%" if avg_att is not None else "N/A")
    with k3:
        avg_sh = column_mean(filtered, "study_hours_per_day")
        st.metric("Avg Study Hrs / Day", f"{avg_sh:.1f}" if avg_sh is not None else "N/A")
    with k4:
        avg_fm = column_mean(filtered, "final_marks")
        st.metric("Avg Final Marks", f"{avg_fm:.1f}" if avg_fm is not None else "N/A")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_overview, tab_distributions, tab_correlations, tab_compare, tab_quality, tab_search = st.tabs(
        ["📋 Overview", "📊 Distributions", "🔗 Correlations", "🆚 Compare", "🩺 Data Quality", "🔍 Smart Search"]
    )

    # ── TAB 1 — Overview ─────────────────────────────────────────────────
    with tab_overview:
        st.markdown('<p class="section-header">📋 Student Data Table</p>', unsafe_allow_html=True)

        display_cols = [
            "student_id",
            "student_name",
            "attendance_percentage",
            "study_hours_per_day",
            "internal_marks",
            "final_marks",
            "grade",
        ]
        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        st.markdown('<p class="section-header">📊 Descriptive Statistics</p>', unsafe_allow_html=True)
        desc = descriptive_summary(filtered)
        if not desc.empty:
            # Make column names more readable
            desc["Column"] = desc["Column"].replace(
                {
                    "attendance_percentage": "Attendance %",
                    "study_hours_per_day": "Study Hrs/Day",
                    "internal_marks": "Internal Marks",
                    "final_marks": "Final Marks",
                }
            )
            st.dataframe(desc, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for the current filters.")

        # Download filtered data
        st.markdown("---")
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Filtered Dataset (CSV)",
            data=csv_bytes,
            file_name="learntrack_filtered.csv",
            mime="text/csv",
        )

    # ── TAB 2 — Distributions ───────────────────────────────────────────
    with tab_distributions:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p class="section-header">📊 Attendance Distribution</p>', unsafe_allow_html=True)
            st.pyplot(attendance_distribution(filtered))
        with col_b:
            st.markdown('<p class="section-header">📊 Study Hours Distribution</p>', unsafe_allow_html=True)
            st.pyplot(study_hours_distribution(filtered))

        st.markdown('<p class="section-header">📊 Internal vs Final Marks</p>', unsafe_allow_html=True)
        st.pyplot(internal_vs_final_marks(filtered))

        st.markdown('<p class="section-header">📊 Grade Distribution</p>', unsafe_allow_html=True)
        grade_df = grade_distribution(filtered)
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            st.pyplot(grade_distribution_chart(grade_df))
        with col_g2:
            st.markdown("**Grading Thresholds**")
            st.markdown(
                """
                | Grade | Marks Range |
                |-------|-------------|
                | A+    | 90 – 100    |
                | A     | 75 – 89     |
                | B     | 60 – 74     |
                | C     | 50 – 59     |
                | D     | 40 – 49     |
                | F     | 0 – 39      |
                """
            )
            if not grade_df.empty:
                st.dataframe(grade_df, use_container_width=True, hide_index=True)

    # ── TAB 3 — Correlations ────────────────────────────────────────────
    with tab_correlations:
        st.markdown('<p class="section-header">🔗 Correlation Heatmap</p>', unsafe_allow_html=True)
        corr_mat = correlation_matrix(filtered)
        if not corr_mat.empty:
            st.pyplot(correlation_heatmap(corr_mat))
        else:
            st.info("Not enough data to compute correlations.")

        st.markdown('<p class="section-header">🔗 Pairwise Correlations (with p-values)</p>', unsafe_allow_html=True)
        corr_report = full_correlation_report(filtered)
        if not corr_report.empty:
            st.dataframe(corr_report, use_container_width=True, hide_index=True)
            st.caption(
                "ℹ️ Pearson *r* measures linear association. "
                "A statistically significant correlation does **not** imply causation."
            )
        else:
            st.info("Not enough data for correlation analysis.")

        # Normality tests
        st.markdown('<p class="section-header">📐 Normality Tests (Shapiro–Wilk)</p>', unsafe_allow_html=True)
        st.caption(
            "Shapiro–Wilk tests whether each variable's distribution is "
            "approximately normal — an assumption behind Pearson *r*."
        )
        norm_rows = []
        for col in ["attendance_percentage", "study_hours_per_day", "internal_marks", "final_marks"]:
            res = shapiro_wilk_test(filtered, col)
            norm_rows.append(
                {
                    "Variable": col,
                    "W Statistic": res["statistic"],
                    "p-value": res["p_value"],
                    "Normal (α=0.05)?": "Yes" if res["p_value"] is not None and res["p_value"] > 0.05 else "No",
                }
            )
        st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, hide_index=True)

    # ── TAB 4 — Compare ─────────────────────────────────────────────────
    with tab_compare:
        st.markdown('<p class="section-header">🆚 Student Performance Comparison</p>', unsafe_allow_html=True)
        st.caption("Select two or more students to compare their metrics side by side.")

        if "student_id" in filtered.columns and not filtered.empty:
            options = filtered.apply(
                lambda r: f"{r['student_id']} – {r['student_name']}", axis=1
            ).tolist()
            selected = st.multiselect(
                "Select students to compare",
                options=options,
                default=options[:3] if len(options) >= 3 else options,
            )
            selected_ids = [s.split(" – ")[0] for s in selected]

            if len(selected_ids) >= 2:
                st.pyplot(student_comparison_chart(filtered, selected_ids))

                # Side-by-side table
                compare_df = filtered[filtered["student_id"].isin(selected_ids)][
                    ["student_name", "attendance_percentage", "study_hours_per_day", "internal_marks", "final_marks", "grade"]
                ].reset_index(drop=True)
                st.dataframe(compare_df, use_container_width=True, hide_index=True)
            elif len(selected_ids) == 1:
                st.warning("Please select at least **two** students to compare.")
            else:
                st.info("Use the selector above to pick students.")
        else:
            st.info("No students available with the current filters.")

    # ── TAB 5 — Data Quality ────────────────────────────────────────────
    with tab_quality:
        st.markdown('<p class="section-header">🩺 Data Quality Summary</p>', unsafe_allow_html=True)
        report = generate_quality_report(raw_df, cleaned_df)

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Total Rows (Raw)", report.total_rows)
        with q2:
            st.metric("Missing Values", report.total_missing)
        with q3:
            st.metric("Duplicate Rows", report.duplicate_count)
        with q4:
            st.metric("Out-of-Range Values", report.total_out_of_range)

        st.markdown("---")

        if report.missing_values:
            st.markdown("**Missing Values by Column:**")
            st.json(report.missing_values)
        else:
            st.success("✅ No missing values detected.")

        if report.out_of_range:
            st.markdown("**Out-of-Range Values by Column:**")
            st.json(report.out_of_range)
        else:
            st.success("✅ All numeric values are within expected ranges.")

        if report.duplicate_count > 0:
            st.warning(f"⚠️ {report.duplicate_count} duplicate row(s) found and removed during cleaning.")
        else:
            st.success("✅ No duplicate rows detected.")

        st.markdown("---")
        st.markdown(
            f"**Rows after cleaning:** {report.rows_after_cleaning} "
            f"(out of {report.total_rows} raw rows)"
        )

    # ── TAB 6 — Smart Search ─────────────────────────────────────────────
    with tab_search:
        st.markdown('<p class="section-header">🔍 Smart Search</p>', unsafe_allow_html=True)
        st.markdown(
            "Search student records using natural language. "
            "Type a query describing the students you're looking for."
        )

        search_query = st.text_input(
            "Enter your search query",
            placeholder="e.g. students with marks above 80",
            key="smart_search_input",
        )

        def _apply_example_query(query_text: str) -> None:
            st.session_state["smart_search_input"] = query_text

        # Suggested clickable example queries
        st.markdown("**💡 Quick examples (click to search):**")
        ex_cols = st.columns(2)
        for i, eq in enumerate(EXAMPLE_QUERIES[:6]):
            with ex_cols[i % 2]:
                st.button(
                    eq,
                    key=f"ex_btn_{i}",
                    on_click=_apply_example_query,
                    args=(eq,),
                    use_container_width=True,
                )

        with st.expander("📖 More examples & supported syntax", expanded=False):
            for i, eq in enumerate(EXAMPLE_QUERIES[6:], start=6):
                st.button(
                    eq,
                    key=f"ex_btn_{i}",
                    on_click=_apply_example_query,
                    args=(eq,),
                    use_container_width=True,
                )
            st.markdown(
                "**Supported query patterns:**\n"
                "- **Comparison:** `marks above 80`, `attendance below 75`, `study hours more than 5`\n"
                "- **Exact / Grade:** `grade A`, `grade A+`, `grade is B`\n"
                "- **Range:** `marks between 60 and 80`\n"
                "- **Compound:** `attendance above 85 and marks below 60`\n"
                "- **Aliases:** `marks` (final marks), `internal marks`, `attendance`, `study hours`"
            )

        if search_query:
            result = smart_search(filtered, search_query)

            if result.success:
                st.markdown(f"**Interpreted condition:** {result.description}")
                st.markdown(f"**Matching students:** {result.match_count}")

                if result.match_count > 0 and result.matches is not None:
                    display_search_cols = [
                        c for c in [
                            "student_id", "student_name",
                            "attendance_percentage", "study_hours_per_day",
                            "internal_marks", "final_marks", "grade",
                        ]
                        if c in result.matches.columns
                    ]
                    st.dataframe(
                        result.matches[display_search_cols].reset_index(drop=True),
                        use_container_width=True,
                        height=min(400, 40 + 35 * result.match_count),
                    )
                else:
                    st.info(
                        "🔎 No students match your query with the current filters. "
                        "Try adjusting the sidebar filters or broadening your search."
                    )
            else:
                st.warning(f"⚠️ {result.error_message}")

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "LearnTrack • Student Performance Analytics Dashboard • "
        "Built with Streamlit, Pandas, Matplotlib, Seaborn & SciPy • "
        "Dataset is **synthetic** — not real student data."
    )


if __name__ == "__main__":
    main()
