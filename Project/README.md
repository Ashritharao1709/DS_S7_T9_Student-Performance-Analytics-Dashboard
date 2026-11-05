# 📊 LearnTrack — Student Performance Analytics Dashboard

An interactive, web-based analytics dashboard that helps educators understand student academic performance through attendance, study hours, and examination data.

> **Note:** This is a **data-analytics** project. It does not include predictive models, AI features, authentication, or database backends.

---

## 🎯 Objectives

- Analyse student performance patterns using descriptive statistics.
- Explore relationships between attendance, study habits, and exam results through correlation analysis.
- Present insights via an interactive, filterable Streamlit dashboard.
- Maintain clean, tested, and well-documented code suitable for an academic submission.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **KPI Cards** | Student count, average attendance, study hours, and final marks at a glance. |
| **Sidebar Filters** | Filter by attendance range, study hours, final marks, and letter grade. |
| **Data Table** | Searchable, sortable table of all (filtered) student records. |
| **Distributions** | Histograms with KDE for attendance and study hours. |
| **Marks Comparison** | Scatter plot of internal vs. final marks with trend line. |
| **Grade Distribution** | Bar chart with clearly documented grading thresholds. |
| **Correlation Heatmap** | Pearson correlation matrix with colour-coded heatmap. |
| **Pairwise Correlations** | r-values with SciPy p-values for every numeric pair. |
| **Normality Tests** | Shapiro–Wilk test per variable (validates Pearson assumptions). |
| **Student Comparison** | Select students and compare their metrics side by side. |
| **Data Quality** | Summary of missing values, duplicates, and out-of-range records. |
| **Smart Search** | Natural-language query filter using Python's built-in `re` module (comparisons, ranges, grades, compound conditions). |
| **CSV Download** | Download the filtered dataset with one click. |

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language |
| **Streamlit** | Interactive web dashboard |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **SciPy** | Statistical tests (Pearson r with p-values, Shapiro–Wilk) |
| **Matplotlib** | Chart rendering |
| **Seaborn** | Statistical visualizations |
| **pytest** | Automated testing |

## 📁 Project Structure

```
Project/
├── app.py                      # Streamlit dashboard entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── data/
│   └── student_performance.csv # Synthetic dataset (50 records)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CSV loading & schema validation
│   ├── preprocessing.py        # Cleaning, missing-value & range handling
│   ├── analytics.py            # Descriptive stats, grading, correlation
│   ├── smart_search.py         # Regex-based natural language query parser
│   └── visualizations.py       # Matplotlib/Seaborn chart builders
└── tests/
    ├── __init__.py
    ├── test_data_loader.py     # Data-loading & validation tests
    ├── test_preprocessing.py   # Preprocessing & quality-report tests
    ├── test_analytics.py       # Statistics, grading & correlation tests
    └── test_smart_search.py    # Smart Search parser & filtering tests
```

## 🚀 Installation

1. **Clone or copy** the project directory.

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   .\.venv\Scripts\python.exe pip install -r requirements.txt
   ```

## ▶️ Launching the Dashboard

From the project root directory, run:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

## 🔍 Smart Search Feature

The **Smart Search** tab provides lightweight, plain-English querying without any external AI APIs or machine learning:
- **Numerical comparisons**: e.g., `marks above 80`, `attendance below 75%`, `study hours more than 5`
- **Grade matching**: e.g., `grade A`, `grade A+`, `grade is B`
- **Range queries**: e.g., `marks between 60 and 80`
- **Compound conditions**: e.g., `attendance above 85 and marks below 60`
- **Interactive query chips**: Clickable suggested examples to test with a single click.

## 🧪 Running Automated Tests

```bash
python -m pytest tests/ -v
```

All **89 tests** should pass. Tests cover:

- Dataset loading and schema validation
- Missing-value and duplicate handling
- Out-of-range value detection and cleaning
- Descriptive statistics (mean, median, std, min, max)
- Grade assignment and distribution
- Pearson correlation with p-values
- Shapiro–Wilk normality tests
- Smart Search query parsing, operators, ranges, grade matching, compound conditions, and edge cases
- Empty datasets and boundary conditions

## 📊 Dataset Schema

The synthetic dataset (`data/student_performance.csv`) contains **50 records** with the following columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `student_id` | String | S001–S050 | Unique student identifier |
| `student_name` | String | — | Student name |
| `attendance_percentage` | Float | 0–100 | Class attendance (%) |
| `study_hours_per_day` | Float | 0–24 | Average daily study hours |
| `internal_marks` | Float | 0–50 | Internal assessment marks |
| `final_marks` | Float | 0–100 | Final examination marks |

### Grading Scale

| Grade | Final Marks Range |
|-------|-------------------|
| A+    | 90 – 100          |
| A     | 75 – 89           |
| B     | 60 – 74           |
| C     | 50 – 59           |
| D     | 40 – 49           |
| F     | 0 – 39            |

> ⚠️ **Disclaimer:** This dataset is entirely **synthetic** and was generated for development and demonstration purposes. It does not represent real student data.

## 📄 License

This project was created for academic purposes.
