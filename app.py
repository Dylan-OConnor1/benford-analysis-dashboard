import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Benford's Law Analysis", page_icon="📊", layout="wide")

st.title("📊 Benford's Law Analysis Dashboard")
st.markdown(
    "Upload a CSV or Excel workbook, filter the records, and evaluate whether "
    "the selected numeric column follows Benford's Law."
)

st.sidebar.header("1. Upload data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
)

@st.cache_data(show_spinner=False)
def load_file(file_bytes, filename, sheet_name=None):
    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(buffer)
    return pd.read_excel(buffer, sheet_name=sheet_name)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file in the sidebar to begin.")
    st.markdown(
        "**Supported formats:** CSV, XLSX, XLS  \n"
        "**Analysis:** first significant digit and first two significant digits."
    )
    st.stop()

try:
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.sidebar.selectbox("Excel worksheet", xls.sheet_names)
        df = load_file(uploaded_file.getvalue(), uploaded_file.name, sheet)
    else:
        df = load_file(uploaded_file.getvalue(), uploaded_file.name)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded file contains no rows.")
    st.stop()

st.sidebar.header("2. Filter records")
filtered = df.copy()
filter_count = 0

# Filter criterion 1: categorical/text selection
categorical_cols = [
    c for c in df.columns
    if not pd.api.types.is_numeric_dtype(df[c])
    and df[c].nunique(dropna=True) <= 100
]
if categorical_cols:
    cat_col = st.sidebar.selectbox("Category filter (optional)", ["None"] + categorical_cols)
    if cat_col != "None":
        values = sorted(df[cat_col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(
            f"Select {cat_col} values", values, default=values
        )
        filtered = filtered[filtered[cat_col].astype(str).isin(selected)]
        filter_count += 1

# Filter criterion 2: numeric range
numeric_cols_all = df.select_dtypes(include="number").columns.tolist()
if numeric_cols_all:
    range_col = st.sidebar.selectbox("Numeric range filter (optional)", ["None"] + numeric_cols_all)
    if range_col != "None":
        numeric_values = pd.to_numeric(df[range_col], errors="coerce").dropna()
        if not numeric_values.empty:
            low, high = float(numeric_values.min()), float(numeric_values.max())
            if low < high:
                chosen_range = st.sidebar.slider(
                    f"Range for {range_col}", min_value=low, max_value=high,
                    value=(low, high)
                )
                filtered = filtered[
                    pd.to_numeric(filtered[range_col], errors="coerce").between(*chosen_range)
                ]
            else:
                st.sidebar.caption(f"{range_col} has only one distinct numeric value.")
                filtered = filtered[pd.to_numeric(filtered[range_col], errors="coerce") == low]
            filter_count += 1

# Filter criterion 3: date range (when a date-like column is available)
date_candidates = []
for c in df.columns:
    if c in numeric_cols_all:
        continue
    parsed = pd.to_datetime(df[c], errors="coerce")
    if parsed.notna().mean() >= 0.7 and parsed.notna().sum() > 1:
        date_candidates.append(c)

if date_candidates:
    date_col = st.sidebar.selectbox("Date filter (optional)", ["None"] + date_candidates)
    if date_col != "None":
        parsed_all = pd.to_datetime(df[date_col], errors="coerce")
        min_date, max_date = parsed_all.min().date(), parsed_all.max().date()
        date_range = st.sidebar.date_input(
            f"Date range for {date_col}", value=(min_date, max_date),
            min_value=min_date, max_value=max_date
        )
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
            parsed_filtered = pd.to_datetime(filtered[date_col], errors="coerce")
            filtered = filtered[
                parsed_filtered.dt.date.between(start_date, end_date)
            ]
        filter_count += 1

st.sidebar.caption("Use the three independent filter types above where applicable.")
st.subheader("Dataset overview")
m1, m2, m3 = st.columns(3)
m1.metric("Uploaded rows", f"{len(df):,}")
m2.metric("Filtered rows", f"{len(filtered):,}")
m3.metric("Columns", f"{len(df.columns):,}")
st.dataframe(filtered.head(1000), use_container_width=True)
if len(filtered) > 1000:
    st.caption("Preview limited to the first 1,000 filtered rows.")

if filtered.empty:
    st.warning("No records remain after filtering. Adjust the filters.")
    st.stop()

numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error("No numeric columns were found. Benford analysis requires numeric data.")
    st.stop()

st.sidebar.header("3. Choose analysis")
analysis_col = st.sidebar.selectbox("Numeric column to analyze", numeric_cols)

def leading_digits(values, digits):
    """Return first significant digit(s) from finite, nonzero numeric values."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = np.abs(arr[np.isfinite(arr) & (arr != 0)])
    if arr.size == 0:
        return np.array([], dtype=int)
    # Scaling by the base-10 exponent avoids string-formatting and rounding issues.
    mantissa = arr / (10.0 ** np.floor(np.log10(arr)))
    leading = np.floor(mantissa * (10 ** (digits - 1))).astype(int)
    lower, upper = 10 ** (digits - 1), 10 ** digits
    leading = leading[(leading >= lower) & (leading < upper)]
    return leading

def benford_expected(digits):
    start, end = (1, 9) if digits == 1 else (10, 99)
    values = np.arange(start, end + 1)
    expected = np.log10(1 + 1 / values)
    return values, expected

def ks_one_sample(observed_counts, expected_probabilities):
    """
    One-sample two-sided KS distance between empirical and Benford CDFs.
    Critical value uses the conventional alpha=.05 asymptotic approximation
    D_critical = 1.36/sqrt(n). The statistic itself is calculated with NumPy.
    """
    n = int(np.sum(observed_counts))
    if n == 0:
        return np.nan, np.nan, False
    observed_cdf = np.cumsum(observed_counts / n)
    expected_cdf = np.cumsum(expected_probabilities)
    statistic = float(np.max(np.abs(observed_cdf - expected_cdf)))
    critical = 1.36 / np.sqrt(n)
    return statistic, float(critical), bool(statistic > critical)

def analyze(values, digits):
    labels, expected = benford_expected(digits)
    found = leading_digits(values, digits)
    counts = np.array([(found == label).sum() for label in labels], dtype=int)
    n = int(counts.sum())
    actual = counts / n if n else np.zeros_like(expected, dtype=float)
    ks, critical, different = ks_one_sample(counts, expected)
    result = pd.DataFrame({
        "Digit": labels,
        "Actual Count": counts,
        "Actual Percentage": actual * 100,
        "Benford Expected Percentage": expected * 100,
        "Difference (percentage points)": (actual - expected) * 100,
    })
    return result, n, ks, critical, different

st.header("Benford analysis")
st.write(
    f"Selected column: **{analysis_col}**. Values are converted to absolute values; "
    "zeros, missing values, and non-finite values are excluded. Leading digits "
    "are significant digits, not digits after the decimal point."
)

one, n1, ks1, crit1, diff1 = analyze(filtered[analysis_col], 1)
two, n2, ks2, crit2, diff2 = analyze(filtered[analysis_col], 2)

def plot_distribution(result, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=result["Digit"].astype(str), y=result["Actual Percentage"],
        name="Observed", marker_color="#3B82F6"
    ))
    fig.add_trace(go.Scatter(
        x=result["Digit"].astype(str), y=result["Benford Expected Percentage"],
        name="Benford expected", mode="lines+markers",
        line=dict(color="#EF4444", width=2)
    ))
    fig.update_layout(
        title=title, xaxis_title="Leading digit(s)", yaxis_title="Percentage (%)",
        barmode="group", template="plotly_white", legend_title=""
    )
    return fig

def show_test(label, n, statistic, critical, different):
    st.subheader(label)
    a, b, c, d = st.columns(4)
    a.metric("Valid observations", f"{n:,}")
    b.metric("KS statistic (D)", f"{statistic:.4f}" if np.isfinite(statistic) else "N/A")
    c.metric("5% critical value", f"{critical:.4f}" if np.isfinite(critical) else "N/A")
    d.metric("Decision", "Different" if different else ("Not different" if n else "No data"))
    if n:
        if different:
            st.warning(
                "The KS statistic exceeds the approximate 5% critical value. "
                "The observed cumulative distribution differs statistically from Benford's distribution "
                "at the 5% level using this KS critical-value approximation."
            )
        else:
            st.success(
                "The KS statistic does not exceed the approximate 5% critical value. "
                "This analysis does not find a statistically significant difference at the 5% level."
            )
        st.caption(
            "KS test: D = max |F_observed(x) − F_Benford(x)|; "
            "decision rule at α = 0.05: D > 1.36/√n. "
            "This is an asymptotic approximation and assumes independent observations."
        )
    else:
        st.info("No valid nonzero numeric observations are available for this test.")

tab1, tab2 = st.tabs(["First digit", "First two digits"])
with tab1:
    show_test("First significant digit (1–9)", n1, ks1, crit1, diff1)
    st.plotly_chart(plot_distribution(one, "Observed vs. Benford — first digit"), use_container_width=True)
    st.dataframe(one, use_container_width=True, hide_index=True)
with tab2:
    show_test("First two significant digits (10–99)", n2, ks2, crit2, diff2)
    st.plotly_chart(plot_distribution(two, "Observed vs. Benford — first two digits"), use_container_width=True)
    st.dataframe(two, use_container_width=True, hide_index=True)

st.header("Summary statistics")
series = pd.to_numeric(filtered[analysis_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
if not series.empty:
    summary = series.describe().to_frame(name="Value").reset_index().rename(columns={"index": "Statistic"})
    st.dataframe(summary, use_container_width=True, hide_index=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mean", f"{series.mean():,.4g}")
    s2.metric("Median", f"{series.median():,.4g}")
    s3.metric("Standard deviation", f"{series.std():,.4g}")
    s4.metric("Zero / missing excluded", f"{len(filtered) - n1:,}")
else:
    summary = pd.DataFrame(columns=["Statistic", "Value"])
    st.info("No finite numeric values available for summary statistics.")

# Build a portable Excel workbook and CSV report for sharing.
report = pd.DataFrame([
    {"Analysis": "First digit", "Valid observations": n1,
     "KS statistic": ks1, "5% critical value": crit1,
     "Statistically different at 5%": diff1 if n1 else "Not testable"},
    {"Analysis": "First two digits", "Valid observations": n2,
     "KS statistic": ks2, "5% critical value": crit2,
     "Statistically different at 5%": diff2 if n2 else "Not testable"},
])
st.header("Download and share results")
st.caption("The report includes the KS decisions, digit-level observed/expected distributions, and descriptive statistics.")
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data (CSV)", csv_bytes, "filtered_data.csv", "text/csv")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    report.to_excel(writer, sheet_name="KS Results", index=False)
    one.to_excel(writer, sheet_name="First Digit", index=False)
    two.to_excel(writer, sheet_name="First Two Digits", index=False)
    summary.to_excel(writer, sheet_name="Summary Statistics", index=False)
    filtered.to_excel(writer, sheet_name="Filtered Data", index=False)
st.download_button(
    "Download full analysis (Excel)",
    excel_buffer.getvalue(),
    "benford_analysis_report.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

text_report = (
    "BENFORD'S LAW ANALYSIS REPORT\n"
    f"Source file: {uploaded_file.name}\n"
    f"Analyzed column: {analysis_col}\n"
    f"Rows uploaded: {len(df)}\nRows after filters: {len(filtered)}\n\n"
    "KS TEST RESULTS (alpha = 0.05; critical value approximation = 1.36/sqrt(n))\n"
    f"First digit: n={n1}, D={ks1}, critical={crit1}, different={diff1 if n1 else 'not testable'}\n"
    f"First two digits: n={n2}, D={ks2}, critical={crit2}, different={diff2 if n2 else 'not testable'}\n\n"
    "Interpretation: A result marked 'different' indicates the KS statistic exceeded the "
    "approximate 5% critical value. Benford analysis is a screening procedure, not proof "
    "of fraud or error. Data suitability and independence assumptions should be considered.\n"
)
st.download_button("Download shareable analysis summary (TXT)", text_report, "benford_summary.txt", "text/plain")

with st.expander("Methodology and limitations"):
    st.markdown(
        """
        - **Benford probability:** P(d) = log10(1 + 1/d), for first digit d=1…9
          and first-two-digit value d=10…99.
        - **KS statistic:** computed using NumPy as the maximum absolute difference
          between observed and expected cumulative distributions.
        - **Decision:** compares D with the conventional asymptotic 5% critical
          approximation, 1.36/√n. This is a critical-value decision, not an exact p-value.
        - The classical one-sample KS test assumes a continuous null distribution;
          Benford digit distributions are discrete. Treat the result as an approximate
          diagnostic, especially for small samples or dependent/rounded data.
        - Benford's Law is not appropriate for every dataset (for example, assigned
          identifiers, check numbers, constrained ranges, or values with fixed minimums).
          A deviation is a flag for further review, not evidence of wrongdoing.
        """
    )

st.divider()
st.caption("Built with Streamlit, Pandas, NumPy, and Plotly.")
