"""
shared_eda.py
=============

Plain, reusable EDA functions — no Streamlit page setup, no session_state,
no widgets. Functions take a dataframe (and sometimes a target column) and
return either data (DataFrames, dicts) or Plotly `Figure` objects a caller
can render however it likes (e.g. `st.plotly_chart(fig)` on a page, or
just inspected directly by fusion_core.py / tests).

WHY THIS FILE EXISTS:
Same reason as shared_preprocessing.py and shared_features.py before it:
app12.py is a full Streamlit PAGE, and importing it directly would re-run
its unconditional `st.set_page_config()` at import time — which crashes
because fusion.py already calls its own. This file holds copies of the
side-effect-free chart-building logic from app12.py's `tab_eda()`, with
nothing else in it. Safe to import from anywhere.

app12.py's own tab_eda() is left completely untouched — these are copies
of the underlying computation, not moves.

WHAT'S DIFFERENT FROM app12.py's SITUATION:
Exactly like preprocessing, app12.py's EDA is 100% manual — every chart
requires the user to pick a column (or several) and a chart type from a
dropdown first. There's no "here's what's probably worth looking at"
logic. So, same two-tier structure as shared_preprocessing.py:

  1. Extracted MECHANICS — the actual px.* calls app12.py already makes
     once a column/type is chosen (histogram+box, correlation heatmap,
     custom chart builder, pair plot, ...). Copies of app12.py's logic.

  2. NEW auto-decide logic (written here for the first time):
       - chart type picked automatically by column dtype
       - columns ranked by relevance to a target column (or, with no
         target, a mild variety heuristic)
       - near-constant and pure-ID-like columns are skipped entirely
       - the ranked, chartable columns are split into a "shown" (top-N)
         set and a "more" set for a later "show more" UI control

RULE-BASED, DETERMINISTIC, NO AI/LLM INVOLVED ANYWHERE — matching the
existing project principle that EDA stays charts-only and fully
explainable.

NOTHING in this file talks to st.session_state or renders any UI.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px


TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# COLUMN TYPE HELPERS
# (numeric_cols/categorical_cols mirror app12.py's helpers of the same
#  purpose; datetime/boolean detection is new — app12.py never separated
#  these out, since its EDA never needed to auto-pick a chart type.)
# ---------------------------------------------------------------------------

def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()


def boolean_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["bool"]).columns.tolist()


# ---------------------------------------------------------------------------
# 1. OVERVIEW
# (mechanic copied from app12.py tab_eda "eda_t1 Overview")
# ---------------------------------------------------------------------------

def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Column / dtype / unique / missing / missing % — same table
    app12.py shows in its Overview sub-tab."""
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "unique": df.nunique().values,
        "missing": df.isnull().sum().values,
        "missing_pct": (df.isnull().mean() * 100).round(1).values,
    })


def describe_all(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors app12.py's `df.describe(include="all").T`."""
    return df.describe(include="all").T


# ---------------------------------------------------------------------------
# 2. DISTRIBUTIONS
# (mechanic copied from app12.py tab_eda "eda_t2 Distributions")
# ---------------------------------------------------------------------------

def build_distribution_chart(df: pd.DataFrame, column: str):
    """Histogram with marginal box plot. Mirrors app12.py's
    px.histogram(df, x=column, marginal="box", ...) call exactly."""
    return px.histogram(
        df, x=column, marginal="box", template=TEMPLATE,
        title=f"Distribution — {column}", color_discrete_sequence=["#1a1a2e"],
    )


def build_box_chart(df: pd.DataFrame, column: str):
    """Mirrors app12.py's px.box(df, y=column, ...) call."""
    return px.box(
        df, y=column, template=TEMPLATE, title=f"Box Plot — {column}",
        color_discrete_sequence=["#374151"],
    )


def build_violin_by_group(df: pd.DataFrame, column: str, group_col: str):
    """Mirrors app12.py's grouped violin plot (shown when the user picks
    a categorical 'Group by' column alongside a numeric distribution)."""
    return px.violin(
        df, y=column, x=group_col, box=True, template=TEMPLATE,
        title=f"{column} by {group_col}",
    )


# ---------------------------------------------------------------------------
# NEW: chart types app12.py's Distributions tab never needed, because it
# only ever plotted a column the user had already confirmed was numeric.
# Auto-mode needs a sensible default for categorical/datetime/boolean
# columns too, since it picks the column (and therefore the dtype) itself.
# ---------------------------------------------------------------------------

def build_bar_counts_chart(df: pd.DataFrame, column: str, top_n: int = 20):
    """Value-counts bar chart for a categorical/boolean column, capped at
    top_n categories so a high-cardinality column doesn't render an
    unreadable chart."""
    counts = df[column].value_counts(dropna=True).head(top_n).reset_index()
    counts.columns = [column, "count"]
    return px.bar(
        counts, x=column, y="count", template=TEMPLATE,
        title=f"Value Counts — {column}", color_discrete_sequence=["#1a1a2e"],
    )


def build_time_series_chart(df: pd.DataFrame, column: str, value_col: Optional[str] = None):
    """Daily-binned line chart for a datetime column: either the mean of
    `value_col` per day, or a record count per day if no value column is
    given. New — app12.py has no datetime-specific chart."""
    work = df[[column]].copy()
    if value_col is not None:
        work[value_col] = df[value_col]
        grouped = work.groupby(pd.Grouper(key=column, freq="D"))[value_col].mean().dropna().reset_index()
        return px.line(
            grouped, x=column, y=value_col, template=TEMPLATE,
            title=f"{value_col} over time ({column})",
        )
    counts = work.groupby(pd.Grouper(key=column, freq="D")).size().reset_index(name="count")
    return px.line(
        counts, x=column, y="count", template=TEMPLATE,
        title=f"Record count over time ({column})",
    )


# ---------------------------------------------------------------------------
# 3. CORRELATIONS
# (mechanic copied from app12.py tab_eda "eda_t3 Correlations")
# ---------------------------------------------------------------------------

def correlation_matrix(
    df: pd.DataFrame, columns: Optional[list[str]] = None, method: str = "pearson"
) -> pd.DataFrame:
    if columns is None:
        columns = numeric_columns(df)
    return df[columns].corr(method=method)


def build_correlation_heatmap(
    df: pd.DataFrame, columns: Optional[list[str]] = None, method: str = "pearson"
):
    """Mirrors app12.py's px.imshow correlation heatmap."""
    corr = correlation_matrix(df, columns, method)
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title=f"Correlation Matrix ({method})", template=TEMPLATE,
    )
    fig.update_layout(height=500)
    return fig


def top_correlated_pairs(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "pearson",
    top_n: int = 20,
) -> pd.DataFrame:
    """Mirrors app12.py's top-correlated-pairs table: upper-triangle only
    (no self-pairs / duplicate mirror pairs), sorted by |correlation|.

    NOTE: app12.py's original code calls plain `.stack()` after masking
    the lower triangle/diagonal to NaN with `.where(...)`, relying on
    `.stack()` to drop those NaNs. Pandas 3.0 changed `.stack()`'s
    behavior so it no longer drops pre-existing NaN values (and passing
    `dropna=True` explicitly now raises a ValueError, since that
    parameter only ever applied to NaNs *introduced by* the reshape, not
    NaNs already present in the data). The same call on app12.py today
    would leak self-pairs and NaN rows into the result on this pandas
    version. Dropping the NaNs explicitly afterward makes this copy
    correct regardless of pandas version — a fix worth carrying back into
    app12.py separately, but out of scope for this file since app12.py
    itself is left untouched."""
    corr = correlation_matrix(df, columns, method)
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna().reset_index()
    pairs.columns = ["feature_a", "feature_b", "correlation"]
    pairs = pairs.iloc[pairs["correlation"].abs().argsort()[::-1].values]
    return pairs.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. CUSTOM CHART BUILDER
# (mechanic copied from app12.py tab_eda "eda_t4 Custom Chart" — the
#  chart_type -> px call mapping, including the melt-based Multi-Y
#  variants.)
# ---------------------------------------------------------------------------

def build_custom_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
    pivot_value_col: Optional[str] = None,
):
    """
    General single-series chart builder. chart_type: "scatter" | "line" |
    "bar" | "histogram" | "box" | "violin" | "pie" | "heatmap_pivot".
    Mirrors app12.py's Custom Chart Builder mapping exactly (same px
    calls, same barmode="group" for bar, same box=True for violin).
    """
    if chart_type == "scatter":
        return px.scatter(df, x=x, y=y, color=color, size=size, template=TEMPLATE)
    elif chart_type == "line":
        return px.line(df, x=x, y=y, color=color, template=TEMPLATE)
    elif chart_type == "bar":
        return px.bar(df, x=x, y=y, color=color, barmode="group", template=TEMPLATE)
    elif chart_type == "histogram":
        return px.histogram(df, x=x, color=color, template=TEMPLATE)
    elif chart_type == "box":
        return px.box(df, x=x, y=y, color=color, template=TEMPLATE)
    elif chart_type == "violin":
        return px.violin(df, x=x, y=y, color=color, box=True, template=TEMPLATE)
    elif chart_type == "pie":
        return px.pie(df, names=x, values=y, template=TEMPLATE)
    elif chart_type == "heatmap_pivot":
        if pivot_value_col is None:
            raise ValueError("heatmap_pivot requires pivot_value_col")
        pivot = df.pivot_table(index=y, columns=x, values=pivot_value_col, aggfunc="mean")
        return px.imshow(pivot, template=TEMPLATE)
    else:
        raise ValueError(f"Unknown chart_type: {chart_type!r}")


def build_multi_series_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y_columns: list[str],
    color_col: Optional[str] = None,
):
    """
    Multi-Y variants: "multi_line" | "multi_bar" | "multi_scatter".
    Mirrors app12.py's melt-then-plot approach exactly (melt on the
    y_columns into a long "series"/"value" pair, then one px call).
    """
    plot_df = df[[x] + y_columns].copy().dropna()
    melted = plot_df.melt(id_vars=x, value_vars=y_columns, var_name="series", value_name="value")
    color = "series" if color_col is None else color_col

    if chart_type == "multi_line":
        return px.line(
            melted, x=x, y="value", color=color, template=TEMPLATE,
            title=f"Multi-Line: {', '.join(y_columns)} vs {x}",
        )
    elif chart_type == "multi_bar":
        return px.bar(
            melted, x=x, y="value", color=color, barmode="group", template=TEMPLATE,
            title=f"Multi-Bar: {', '.join(y_columns)} vs {x}",
        )
    elif chart_type == "multi_scatter":
        return px.scatter(
            melted, x=x, y="value", color=color, template=TEMPLATE,
            title=f"Multi-Scatter: {', '.join(y_columns)} vs {x}",
        )
    else:
        raise ValueError(f"Unknown multi chart_type: {chart_type!r}")


# ---------------------------------------------------------------------------
# 5. PAIR PLOT
# (mechanic copied from app12.py tab_eda "eda_t5 Pair Plot")
# ---------------------------------------------------------------------------

def build_pair_plot(df: pd.DataFrame, columns: list[str], color_col: Optional[str] = None):
    """Mirrors app12.py's px.scatter_matrix pair plot, including hiding
    the diagonal and upper half."""
    fig = px.scatter_matrix(
        df, dimensions=columns, color=color_col, template=TEMPLATE, title="Pair Plot",
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(height=700)
    return fig


# ---------------------------------------------------------------------------
# NEW: AUTO-DECIDE LOGIC — none of this exists in app12.py today.
# ---------------------------------------------------------------------------

def classify_column_utility(
    series: pd.Series,
    name: Optional[str] = None,
    near_constant_threshold: float = 0.98,
    id_uniqueness_threshold: float = 0.98,
) -> tuple[bool, Optional[str]]:
    """
    Decides whether a column is worth auto-charting on its own. Returns
    (useful, reason) — reason is None when useful is True.

    Skips two kinds of uninformative columns:
      - near-constant: one value accounts for >= near_constant_threshold
        of non-null rows — a chart would show almost nothing.
      - pure-ID-like: nearly every value is unique (>= id_uniqueness_
        threshold) AND (the dtype is non-numeric and non-datetime, e.g. a
        UUID/string, OR the column name hints at being an identifier:
        "id", "uuid", "key", ...). A genuinely continuous numeric
        measurement that happens to be mostly unique (e.g. precise sensor
        readings) is NOT skipped unless its name also hints at being an
        identifier — uniqueness alone isn't enough for numeric columns.
        Datetime columns are explicitly exempted from the "non-numeric ->
        ID candidate" rule: a timestamp column is *expected* to be
        near-fully-unique while still being one of the most useful
        columns to chart (time series), so it must not be treated as an
        identifier just for being non-numeric and highly unique.
    """
    s = series.dropna()
    if s.empty:
        return False, "entirely missing"

    top_frac = s.value_counts(normalize=True).iloc[0]
    if top_frac >= near_constant_threshold:
        return False, f"near-constant — {top_frac * 100:.1f}% of values are the same"

    uniqueness = s.nunique() / len(s)
    name_hints_id = (
        name is not None
        and any(h in name.lower() for h in ("id", "uuid", "guid", "_pk", "primarykey", "key"))
    )
    is_nonnumeric_non_datetime = (
        not pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_datetime64_any_dtype(series)
    )

    if uniqueness >= id_uniqueness_threshold and (name_hints_id or is_nonnumeric_non_datetime):
        return False, f"looks like an identifier — {uniqueness * 100:.1f}% unique values"

    return True, None


def auto_chart_type_for_column(series: pd.Series) -> str:
    """
    Rule-based chart-type pick by dtype:
      boolean      -> "bar_counts"
      datetime     -> "line_over_time"
      numeric      -> "histogram_box"
      categorical  -> "bar_counts" (<=20 categories) else "bar_counts_top_n"
    """
    if pd.api.types.is_bool_dtype(series):
        return "bar_counts"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "line_over_time"
    if pd.api.types.is_numeric_dtype(series):
        return "histogram_box"
    n_unique = series.nunique(dropna=True)
    return "bar_counts" if n_unique <= 20 else "bar_counts_top_n"


def _correlation_ratio(categories: pd.Series, values: pd.Series, min_avg_group_size: float = 3.0) -> float:
    """Correlation ratio (eta): fraction of numeric `values`' variance
    explained by grouping on `categories`. A dependency-free stand-in for
    an ANOVA F-test / eta-squared, used to rank a categorical column's
    relevance against a numeric target (or vice versa).

    Guard: when `categories` has very high cardinality relative to the
    sample size (e.g. a near-unique ID column), each group ends up with
    ~1 member, so its mean trivially equals that single value and
    ss_between collapses onto ss_total — producing a perfect but
    meaningless eta of ~1.0 for columns that are really just identifiers.
    Requiring a minimum average group size before trusting the ratio
    prevents high-cardinality columns from appearing maximally
    "relevant" by construction."""
    paired = pd.DataFrame({"cat": categories.values, "val": values.values}).dropna()
    if paired.empty or paired["cat"].nunique() < 2:
        return 0.0
    avg_group_size = len(paired) / paired["cat"].nunique()
    if avg_group_size < min_avg_group_size:
        return 0.0
    grand_mean = paired["val"].mean()
    ss_between = sum(
        len(g) * (g["val"].mean() - grand_mean) ** 2
        for _, g in paired.groupby("cat")
    )
    ss_total = ((paired["val"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(np.sqrt(max(ss_between / ss_total, 0.0)))


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V: dependency-free chi-squared-based association measure
    between two categorical series, used to rank a categorical column's
    relevance against a categorical target."""
    paired = pd.DataFrame({"a": a.values, "b": b.values}).dropna()
    if paired.empty:
        return 0.0
    ct = pd.crosstab(paired["a"], paired["b"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    n = ct.values.sum()
    row_sums = ct.sum(axis=1).values
    col_sums = ct.sum(axis=0).values
    expected = np.outer(row_sums, col_sums) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (ct.values - expected) ** 2 / expected, 0.0))
    r, k = ct.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def rank_columns_by_target_relevance(
    df: pd.DataFrame,
    target: Optional[str] = None,
    candidate_columns: Optional[list[str]] = None,
) -> list[dict]:
    """
    Ranks candidate_columns by relevance to `target`, descending. Returns
    a list of {"column":, "score": (0-1ish), "method":}.

      - No target: ordered by a mild "variety heuristic" — columns with
        moderate uniqueness (neither near-constant nor near-fully-unique)
        score higher, since those tend to make the most informative
        charts on their own. Only used as an ordering tiebreak when
        there's nothing to correlate against.
      - Numeric target:
          numeric candidate     -> |Pearson correlation| with target
          categorical candidate -> correlation ratio (eta) with target
      - Categorical target:
          numeric candidate     -> correlation ratio (eta), roles reversed
          categorical candidate -> Cramér's V with target
    """
    if candidate_columns is None:
        candidate_columns = [c for c in df.columns if c != target]
    else:
        candidate_columns = [c for c in candidate_columns if c != target]

    if target is None or target not in df.columns:
        # NOTE: this heuristic is a tiebreak ordering among columns that
        # have ALREADY passed classify_column_utility (i.e. columns worth
        # charting at all, per select_top_charts()'s normal flow). It
        # deliberately does NOT re-detect near-constant or ID-like
        # columns — that's classify_column_utility's job, and re-deriving
        # it here from a single distinct-value-ratio misfires on
        # legitimately high-cardinality NUMERIC data: a continuous price
        # column is supposed to have almost all-unique values, and
        # penalizing that the same way as a categorical ID column would
        # wrongly bury the most chartable numeric columns.
        scores = []
        for c in candidate_columns:
            s = df[c].dropna()
            if s.empty:
                scores.append({"column": c, "score": 0.0, "method": "no target — no data"})
                continue
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                # Numeric: reward genuine spread (coefficient of
                # variation). A well-spread histogram is more informative
                # than a tightly clustered one, regardless of how many
                # distinct values it has.
                mean_abs = abs(s.mean())
                cv = float(s.std() / mean_abs) if mean_abs > 1e-9 else float(s.std())
                score = min(1.0, cv)
                method = "coefficient of variation (no target set)"
            else:
                # Categorical/boolean: reward a moderate number of
                # categories relative to sample size — enough to be more
                # than a coin flip, few enough to read cleanly on a bar
                # chart.
                uniqueness = s.nunique() / len(s)
                score = max(0.0, 1 - abs(uniqueness - 0.05) / 0.5)
                method = "category-count heuristic (no target set)"
            scores.append({"column": c, "score": round(float(score), 4), "method": method})
        return sorted(scores, key=lambda d: d["score"], reverse=True)

    target_is_numeric = pd.api.types.is_numeric_dtype(df[target])
    scores = []
    for c in candidate_columns:
        col_is_numeric = pd.api.types.is_numeric_dtype(df[c])
        try:
            if target_is_numeric and col_is_numeric:
                score = abs(df[[c, target]].dropna().corr().iloc[0, 1])
                method = "|Pearson correlation| with numeric target"
            elif target_is_numeric and not col_is_numeric:
                score = _correlation_ratio(df[c], df[target])
                method = "correlation ratio (eta) with numeric target"
            elif not target_is_numeric and col_is_numeric:
                score = _correlation_ratio(df[target], df[c])
                method = "correlation ratio (eta) with categorical target"
            else:
                score = _cramers_v(df[c], df[target])
                method = "Cramér's V with categorical target"
            if pd.isna(score):
                score = 0.0
        except Exception:
            score, method = 0.0, "could not compute — defaulted to 0"
        scores.append({"column": c, "score": round(float(score), 4), "method": method})

    return sorted(scores, key=lambda d: d["score"], reverse=True)


def select_top_charts(
    df: pd.DataFrame,
    target: Optional[str] = None,
    top_n: int = 6,
    near_constant_threshold: float = 0.98,
    id_uniqueness_threshold: float = 0.98,
) -> dict:
    """
    Full auto-EDA column/chart selection pipeline (no dataframe mutation,
    no figures rendered here — see auto_eda() below for that):
      1. Classify every column's utility, skipping near-constant / ID-like.
      2. Rank the remaining useful columns by relevance to `target`.
      3. Pick a chart type per column by dtype.
      4. Split into `shown` (top_n) and `more` (the rest).

    Returns {"shown": [...], "more": [...], "skipped": [...]} where each
    chart entry is
      {"column", "chart_type", "relevance_score", "relevance_method", "description"}
    and each skipped entry is {"column", "reason"}.
    """
    skipped = []
    useful_cols = []
    for c in df.columns:
        if c == target:
            continue
        useful, reason = classify_column_utility(
            df[c], c, near_constant_threshold, id_uniqueness_threshold
        )
        if useful:
            useful_cols.append(c)
        else:
            skipped.append({"column": c, "reason": reason})

    ranking = rank_columns_by_target_relevance(df, target, useful_cols)
    rank_map = {r["column"]: r for r in ranking}

    charts = []
    for c in useful_cols:
        chart_type = auto_chart_type_for_column(df[c])
        r = rank_map.get(c, {"score": 0.0, "method": "n/a"})
        target_note = (
            f" — ranked by {r['method']} (score {r['score']:.2f})"
            if target else " — ordered by variety heuristic (no target set)"
        )
        charts.append({
            "column": c,
            "chart_type": chart_type,
            "relevance_score": r["score"],
            "relevance_method": r["method"],
            "description": (
                f"Showing '{c}' as a {chart_type.replace('_', ' ')} chart{target_note}."
            ),
        })

    charts.sort(key=lambda d: d["relevance_score"], reverse=True)

    return {
        "shown": charts[:top_n],
        "more": charts[top_n:],
        "skipped": skipped,
    }


def _render_chart_for_entry(df: pd.DataFrame, entry: dict):
    """Internal: dispatches a select_top_charts() entry to the right
    chart-building function above."""
    col = entry["column"]
    ctype = entry["chart_type"]
    if ctype == "histogram_box":
        return build_distribution_chart(df, col)
    elif ctype in ("bar_counts", "bar_counts_top_n"):
        return build_bar_counts_chart(df, col)
    elif ctype == "line_over_time":
        return build_time_series_chart(df, col)
    return None


def auto_eda(df: pd.DataFrame, target: Optional[str] = None, top_n: int = 6) -> dict:
    """
    The "automatic by default" EDA pass — the single entry point fusion
    (or any other automatic caller) needs. Chains:
      1. dataset_overview()
      2. select_top_charts() (skip near-constant/ID, rank by relevance,
         pick chart type by dtype, split shown/more)
      3. renders a Plotly figure for every shown/more entry
      4. a correlation heatmap + top pairs table, if there are >= 2
         numeric columns

    Returns:
      {
        "overview": DataFrame,
        "skipped": [{"column","reason"}, ...],
        "shown":  [{"column","chart_type","relevance_score",
                    "relevance_method","description","figure"}, ...],
        "more":   [... same shape ...],
        "correlation": {"heatmap": fig, "top_pairs": DataFrame} | None,
      }
    `more` is populated but not meant to be rendered until the user asks
    to "show more" — the figures are ready either way so the UI doesn't
    need a second round-trip through this module.
    """
    selection = select_top_charts(df, target=target, top_n=top_n)

    shown = [{**e, "figure": _render_chart_for_entry(df, e)} for e in selection["shown"]]
    more = [{**e, "figure": _render_chart_for_entry(df, e)} for e in selection["more"]]

    result = {
        "overview": dataset_overview(df),
        "skipped": selection["skipped"],
        "shown": shown,
        "more": more,
        "correlation": None,
    }

    num_cols = numeric_columns(df)
    if len(num_cols) >= 2:
        result["correlation"] = {
            "heatmap": build_correlation_heatmap(df, num_cols),
            "top_pairs": top_correlated_pairs(df, num_cols),
        }

    return result
