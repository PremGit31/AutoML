"""
shared_preprocessing.py
========================

Plain, reusable preprocessing functions — no Streamlit page setup, no
session_state, no widgets. Just "dataframe in, cleaned dataframe out"
functions, mirroring shared_features.py's pattern.

WHY THIS FILE EXISTS:
app12.py is a full Streamlit PAGE. Importing it directly (e.g.
`from app12 import fill_missing`) would re-run everything in the module
top to bottom, including its unconditional `st.set_page_config(...)` call
near the top of the file. fusion.py already calls its own
st.set_page_config(), and Streamlit errors if it's called twice in one
page run — so importing app12.py directly would crash fusion.py.

This file solves that the same way shared_features.py did for
nlp_app2.py / image_app.py: it holds copies of the genuinely reusable,
side-effect-free computation from app12.py's `tab_preprocessing()`, with
nothing else in it. Safe to import from anywhere.

WHAT'S DIFFERENT FROM shared_features.py's SITUATION:
app12.py's preprocessing is 100% manual today — every step in
tab_preprocessing() requires the user to pick a strategy from a dropdown
and click Apply. There is no "automatic" mode to extract. So this file
has TWO kinds of functions for each preprocessing area:

  1. The extracted MECHANIC — the actual computation app12.py already
     performs once a strategy is chosen (e.g. `fill_missing(df, cols,
     "median")`). These are copies of app12.py's logic, not moves —
     app12.py's tab_preprocessing() is left completely untouched.

  2. A NEW auto-decide function — brand-new rule-based logic (written
     here for the first time) that picks a strategy automatically so
     fusion's "automatic by default" mode has something to call. Every
     auto-decide function returns plain-English description strings
     alongside its decisions, ready to drive an inline "change this" UI
     control later — nothing here renders one.

RULE-BASED DEFAULTS (deterministic, no AI/LLM involved anywhere):
  - Missing numeric values  -> fill with column median
  - Missing categorical values -> fill with column most-frequent (mode)
  - Column missing > ~50-60% -> drop the column instead of imputing
  - Outliers -> flagged via IQR, never auto-deleted — EXCEPT values that
    are physically impossible for the column (e.g. a negative "price" or
    "age"), which get capped at a sane floor since those are data-entry
    errors, not legitimate extreme values.
  - Scaling -> chosen (or skipped) based on which algorithm will consume
    the data downstream (tree-based algorithms don't need it; distance/
    gradient-based algorithms do).

NOTHING in this file talks to st.session_state or renders any UI. If
you're tempted to add a widget here, it belongs in the page instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)


# ---------------------------------------------------------------------------
# 1. MISSING VALUES
# (mechanic copied from app12.py tab_preprocessing "1. Handle Missing
#  Values" -> Global Strategy block, lines ~ "Global Strategy — apply to
#  column(s) as a whole". The per-row/per-PK override UI in that same
#  expander is interactive row-browsing, not a preprocessing rule, and is
#  not extracted here.)
# ---------------------------------------------------------------------------

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Column / missing count / missing % for every column that has any
    nulls. Same numbers app12.py shows in its Missing Values expander."""
    missing = df.isnull().sum()
    cols = missing[missing > 0]
    return pd.DataFrame({
        "column": cols.index,
        "missing_count": cols.values,
        "missing_pct": (df[cols.index].isnull().mean() * 100).round(2).values,
    })


def fill_missing(
    df: pd.DataFrame,
    columns: list[str],
    strategy: str,
    custom_value: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Manual-mode mechanic. strategy: "drop_rows" | "mean" | "median" |
    "mode" | "custom" | "drop_columns". Mirrors app12.py's Global Strategy
    block exactly (same fillna calls, same numeric-dtype guard for
    mean/median).
    Returns (new_df, description).
    """
    work = df.copy()

    if strategy == "drop_rows":
        before_rows = len(work)
        work = work.dropna(subset=columns)
        return work, f"Dropped {before_rows - len(work)} row(s) with nulls in {columns}"

    elif strategy == "mean":
        for c in columns:
            if pd.api.types.is_numeric_dtype(work[c]):
                work[c] = work[c].fillna(work[c].mean())
        return work, f"Filled missing values in {columns} with column mean"

    elif strategy == "median":
        for c in columns:
            if pd.api.types.is_numeric_dtype(work[c]):
                work[c] = work[c].fillna(work[c].median())
        return work, f"Filled missing values in {columns} with column median"

    elif strategy == "mode":
        for c in columns:
            m = work[c].mode()
            if len(m) > 0:
                work[c] = work[c].fillna(m[0])
        return work, f"Filled missing values in {columns} with column mode (most frequent)"

    elif strategy == "custom":
        for c in columns:
            work[c] = work[c].fillna(custom_value)
        return work, f"Filled missing values in {columns} with custom value {custom_value!r}"

    elif strategy == "drop_columns":
        work = work.drop(columns=columns)
        return work, f"Dropped columns (too much missing data or by choice): {columns}"

    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy!r}")


def auto_decide_missing_value_strategy(
    df: pd.DataFrame, drop_threshold: float = 0.55
) -> list[dict]:
    """
    NEW rule-based logic (not present in app12.py). For every column that
    has missing values, decides:
      - missing % > drop_threshold  -> drop the column
      - else numeric                -> fill with median
      - else categorical            -> fill with mode (most frequent)

    drop_threshold defaults to 0.55, the middle of the requested 50-60%
    band. Returns a list of decision dicts, each carrying a plain-English
    `description` for an inline "change this" control. Makes no changes
    to `df` itself.
    """
    missing_frac = df.isnull().mean()
    decisions = []
    for col in missing_frac[missing_frac > 0].index:
        pct = missing_frac[col] * 100
        if missing_frac[col] > drop_threshold:
            decisions.append({
                "column": col,
                "action": "drop_column",
                "strategy": None,
                "description": (
                    f"Drop column '{col}' — {pct:.1f}% missing, above the "
                    f"{drop_threshold * 100:.0f}% threshold where imputation "
                    f"stops being reliable."
                ),
            })
        elif pd.api.types.is_numeric_dtype(df[col]):
            decisions.append({
                "column": col,
                "action": "fill",
                "strategy": "median",
                "description": (
                    f"Fill '{col}' ({pct:.1f}% missing) with the column median — "
                    f"the standard robust default for numeric data."
                ),
            })
        else:
            decisions.append({
                "column": col,
                "action": "fill",
                "strategy": "mode",
                "description": (
                    f"Fill '{col}' ({pct:.1f}% missing) with the most frequent "
                    f"value — the standard default for categorical data."
                ),
            })
    return decisions


def apply_auto_missing_value_strategy(
    df: pd.DataFrame,
    decisions: Optional[list[dict]] = None,
    drop_threshold: float = 0.55,
) -> tuple[pd.DataFrame, list[str]]:
    """Runs auto_decide_missing_value_strategy (unless decisions are
    supplied, e.g. after a user edited them in the 'change this' UI) and
    applies every decision in one pass. Returns (new_df, log_strings)."""
    if decisions is None:
        decisions = auto_decide_missing_value_strategy(df, drop_threshold)

    work = df.copy()
    log = []

    drop_cols = [d["column"] for d in decisions if d["action"] == "drop_column"]
    if drop_cols:
        work = work.drop(columns=[c for c in drop_cols if c in work.columns])

    for d in decisions:
        if d["action"] != "fill" or d["column"] not in work.columns:
            log.append(d["description"])
            continue
        c = d["column"]
        if d["strategy"] == "median":
            work[c] = work[c].fillna(work[c].median())
        else:  # mode
            m = work[c].mode()
            if len(m) > 0:
                work[c] = work[c].fillna(m[0])
        log.append(d["description"])

    return work, log


# ---------------------------------------------------------------------------
# 2. FIX DATA TYPES
# (mechanic copied from app12.py tab_preprocessing "2. Fix Data Types")
# ---------------------------------------------------------------------------

_DTYPE_TARGETS = {
    "numeric_float", "numeric_int", "string", "datetime", "category", "boolean",
}


def convert_dtypes(
    df: pd.DataFrame, columns: list[str], target_type: str
) -> tuple[pd.DataFrame, list[str]]:
    """
    target_type: "numeric_float" | "numeric_int" | "string" | "datetime" |
    "category" | "boolean". Mirrors app12.py's dtype-conversion block
    (same pd.to_numeric / astype calls).
    Returns (new_df, list of per-column description strings).
    """
    if target_type not in _DTYPE_TARGETS:
        raise ValueError(f"Unknown target_type: {target_type!r}")

    work = df.copy()
    old_types = {c: str(work[c].dtype) for c in columns}
    for c in columns:
        if target_type == "numeric_float":
            work[c] = pd.to_numeric(work[c], errors="coerce").astype(float)
        elif target_type == "numeric_int":
            work[c] = pd.to_numeric(work[c], errors="coerce").astype("Int64")
        elif target_type == "string":
            work[c] = work[c].astype(str)
        elif target_type == "datetime":
            work[c] = pd.to_datetime(work[c], errors="coerce")
        elif target_type == "category":
            work[c] = work[c].astype("category")
        elif target_type == "boolean":
            work[c] = work[c].astype(bool)

    desc = [f"Converted '{c}' {old_types[c]} -> {target_type}" for c in columns]
    return work, desc


# ---------------------------------------------------------------------------
# 3. REMOVE DUPLICATES
# (mechanic copied from app12.py tab_preprocessing "3. Remove Duplicates")
# ---------------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    keep: str | bool = "first",
) -> tuple[pd.DataFrame, str]:
    """keep: "first" | "last" | False (drop every occurrence, matching
    app12.py's "none (drop all)" option)."""
    before = len(df)
    work = df.drop_duplicates(subset=subset if subset else None, keep=keep)
    removed = before - len(work)
    scope = f"subset {subset}" if subset else "full row match"
    return work, f"Removed {removed} duplicate row(s) ({scope})"


# ---------------------------------------------------------------------------
# 4. OUTLIERS
# (mechanic copied from app12.py tab_preprocessing "4. Handle Outliers" —
#  same three methods, same two actions.)
# ---------------------------------------------------------------------------

def detect_outliers(
    df: pd.DataFrame, columns: list[str], method: str = "iqr"
) -> tuple[pd.Series, dict[str, tuple[float, float]]]:
    """
    method: "zscore" (|z| > 3) | "iqr" (1.5x IQR) | "percentile" (1%-99%).
    Returns (boolean mask — True where ANY selected column is out of
    bounds, {column: (lower_bound, upper_bound)}).
    """
    mask = pd.Series(False, index=df.index)
    bounds: dict[str, tuple[float, float]] = {}
    for c in columns:
        cd = df[c].dropna()
        if method == "zscore":
            m_, s_ = cd.mean(), cd.std()
            lo, hi = m_ - 3 * s_, m_ + 3 * s_
        elif method == "iqr":
            q1, q3 = cd.quantile(0.25), cd.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        elif method == "percentile":
            lo, hi = cd.quantile(0.01), cd.quantile(0.99)
        else:
            raise ValueError(f"Unknown outlier method: {method!r}")
        bounds[c] = (float(lo), float(hi))
        mask = mask | (df[c] < lo) | (df[c] > hi)
    return mask, bounds


def apply_outlier_action(
    df: pd.DataFrame, columns: list[str], method: str = "iqr", action: str = "cap"
) -> tuple[pd.DataFrame, str]:
    """action: "remove" (drop outlier rows) | "cap" (clip to bounds).
    Manual-mode mechanic, mirrors app12.py exactly."""
    work = df.copy()
    mask, bounds = detect_outliers(work, columns, method)
    if action == "cap":
        for c in columns:
            lo, hi = bounds[c]
            work[c] = work[c].clip(lo, hi)
        return work, f"Capped outliers in {columns} to {method} boundaries"
    elif action == "remove":
        before = len(work)
        work = work[~mask]
        return work, f"Removed {before - len(work)} outlier row(s) in {columns} via {method}"
    else:
        raise ValueError(f"Unknown outlier action: {action!r}")


# Column-name fragments that imply a quantity which cannot legitimately be
# negative. Used only to catch clear data-entry errors (a negative price,
# age, count, ...) — NOT a semantic understanding of the dataset, just a
# narrow, explainable heuristic to avoid blindly capping every column.
_NONNEGATIVE_NAME_HINTS = (
    "price", "cost", "amount", "age", "count", "qty", "quantity",
    "weight", "height", "salary", "income", "revenue", "duration",
    "distance", "population", "size", "volume", "fee", "score", "rating",
    "units", "stock", "inventory", "length", "width", "area",
)


def auto_decide_outlier_strategy(
    df: pd.DataFrame,
    numeric_columns: Optional[list[str]] = None,
    nonnegative_name_hints: Optional[tuple[str, ...]] = None,
) -> list[dict]:
    """
    NEW rule-based logic (not present in app12.py), per the project rule:
    "outliers flagged via IQR, never auto-deleted, except clearly-
    impossible values like negative price."

    For every numeric column:
      - If the column name suggests a quantity that can't be negative
        (price, age, count, ...) AND it actually contains negative
        values, that's treated as an impossible value, not a statistical
        outlier -> flagged for auto-capping at a floor of 0.
      - Otherwise, IQR is used purely to FLAG outliers for human review —
        no row or value is touched.

    Returns a list of decisions; makes no changes to `df`.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    hints = nonnegative_name_hints or _NONNEGATIVE_NAME_HINTS

    decisions = []
    for c in numeric_columns:
        cd = df[c].dropna()
        if cd.empty:
            continue

        name_suggests_nonneg = any(h in c.lower() for h in hints)
        has_negative = bool((cd < 0).any())

        if name_suggests_nonneg and has_negative:
            n_impossible = int((df[c] < 0).sum())
            decisions.append({
                "column": c,
                "action": "cap_impossible",
                "floor": 0,
                "n_impossible": n_impossible,
                "description": (
                    f"'{c}' has {n_impossible} negative value(s), which looks like a "
                    f"data-entry error for a column named like a quantity that can't "
                    f"go below zero. Capping these to 0 rather than treating them as "
                    f"ordinary statistical outliers."
                ),
            })
            continue

        mask, bounds = detect_outliers(df, [c], method="iqr")
        n_outliers = int(mask.sum())
        if n_outliers > 0:
            lo, hi = bounds[c]
            decisions.append({
                "column": c,
                "action": "flag_only",
                "n_outliers": n_outliers,
                "bounds": (lo, hi),
                "description": (
                    f"'{c}' has {n_outliers} value(s) outside the IQR range "
                    f"[{lo:.2f}, {hi:.2f}]. Flagged for review — nothing removed "
                    f"or changed automatically."
                ),
            })
    return decisions


def apply_auto_outlier_strategy(
    df: pd.DataFrame,
    decisions: Optional[list[dict]] = None,
    numeric_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """
    Applies ONLY the 'cap_impossible' decisions. 'flag_only' decisions are
    deliberately NOT applied — per the never-auto-delete-outliers rule,
    they're returned separately as `flagged` so the caller can surface
    them to the user (e.g. in the fusion mismatch-style report) without
    silently mutating the data.
    Returns (new_df, applied_log_strings, flagged_decisions).
    """
    if decisions is None:
        decisions = auto_decide_outlier_strategy(df, numeric_columns)

    work = df.copy()
    applied_log = []
    flagged = []
    for d in decisions:
        if d["action"] == "cap_impossible":
            c = d["column"]
            work[c] = work[c].clip(lower=d["floor"])
            applied_log.append(d["description"])
        else:
            flagged.append(d)
    return work, applied_log, flagged


# ---------------------------------------------------------------------------
# 5. ENCODE CATEGORICAL
# (mechanic copied from app12.py tab_preprocessing "5. Encode Categorical
#  Variables")
# ---------------------------------------------------------------------------

def encode_categorical(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "label",
    new_names: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, str, dict]:
    """
    method: "label" | "onehot". Mirrors app12.py's encoding block exactly,
    including capturing the label->int mapping for label encoding.
    Returns (new_df, description, mappings) where mappings is {} for
    one-hot (no single mapping to report).
    """
    work = df.copy()
    mappings: dict = {}

    if method == "label":
        for c in columns:
            out_col = (new_names or {}).get(c, f"{c}_enc")
            le = LabelEncoder()
            le.fit(work[c].astype(str))
            work[out_col] = le.transform(work[c].astype(str))
            mappings[c] = {
                "output_col": out_col,
                "mapping": {cls: idx for idx, cls in enumerate(le.classes_)},
            }
        return work, f"Label-encoded {columns}", mappings

    elif method == "onehot":
        dummies = pd.get_dummies(work[columns], prefix=columns, drop_first=True)
        work = pd.concat([work, dummies], axis=1)
        return work, f"One-hot encoded {columns} ({dummies.shape[1]} new column(s))", mappings

    else:
        raise ValueError(f"Unknown encoding method: {method!r}")


# ---------------------------------------------------------------------------
# 6. SCALE / NORMALIZE
# (mechanic copied from app12.py tab_preprocessing "6. Scale / Normalize")
# ---------------------------------------------------------------------------

_SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scale_columns(
    df: pd.DataFrame, columns: list[str], method: str = "standard"
) -> tuple[pd.DataFrame, str, object]:
    """method: "standard" | "minmax" | "robust". Mirrors app12.py's
    scaling block. Returns (new_df, description, fitted_scaler_object) —
    the fitted scaler is returned so it can be reused at inference time,
    the same way app12.py keeps its scaler in session_state."""
    if method not in _SCALER_MAP:
        raise ValueError(f"Unknown scaling method: {method!r}")
    work = df.copy()
    scaler = _SCALER_MAP[method]()
    work[columns] = scaler.fit_transform(work[columns])
    return work, f"Scaled {columns} using {method} scaler", scaler


# Algorithms that are invariant to monotonic feature scaling (tree splits
# don't change), vs. algorithms that are scale-sensitive. Used only by the
# NEW auto-decide function below — app12.py has no such notion today.
_TREE_BASED_ALGORITHMS = {
    "random_forest", "decision_tree", "gradient_boosting",
    "xgboost", "lightgbm", "catboost", "extra_trees",
}
_SCALE_SENSITIVE_ALGORITHMS = {
    "knn", "svm", "svc", "kmeans", "dbscan", "logistic_regression",
    "linear_regression", "ridge", "lasso", "polynomial_regression",
    "pca", "sentence_transformer_downstream",
}


def auto_decide_scaling_strategy(
    df: pd.DataFrame,
    downstream_algorithm: str,
    numeric_columns: Optional[list[str]] = None,
) -> dict:
    """
    NEW rule-based logic (not present in app12.py), per the project rule
    "normalization applied based on algorithm type":
      - Tree-based algorithms (Random Forest, Decision Tree, Gradient
        Boosting, ...) -> skip scaling entirely; split decisions are
        unaffected by monotonic feature scale.
      - Distance/gradient-based algorithms (KNN, SVM, K-Means, logistic/
        linear regression, PCA, ...) -> scale, using Standard Scaler by
        default, or Robust Scaler if the data has heavy IQR outliers
        (median/IQR-based scaling is less thrown off by extreme values).
    Returns a single decision dict (not a list) with a `description`
    field ready for the "change this" UI. Makes no changes to `df`.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    algo = downstream_algorithm.lower().replace(" ", "_").replace("-", "_")

    if algo in _TREE_BASED_ALGORITHMS:
        return {
            "should_scale": False,
            "method": None,
            "columns": [],
            "description": (
                f"Skipping scaling — '{downstream_algorithm}' is tree-based, so "
                f"split decisions are unaffected by feature scale."
            ),
        }

    if not numeric_columns:
        return {
            "should_scale": False,
            "method": None,
            "columns": [],
            "description": "No numeric columns available to scale.",
        }

    outlier_decisions = auto_decide_outlier_strategy(df, numeric_columns)
    heavy_outlier_cols = [
        d["column"] for d in outlier_decisions
        if d["action"] == "flag_only" and d.get("n_outliers", 0) > max(3, 0.05 * len(df))
    ]
    use_robust = len(heavy_outlier_cols) > 0

    method = "robust" if use_robust else "standard"
    reason = (
        f"{len(heavy_outlier_cols)} column(s) ({heavy_outlier_cols}) have notable IQR "
        f"outliers, so Robust Scaler (median/IQR-based) is used to reduce their influence"
        if use_robust else
        "Standard Scaler (z-score) is the default for scale-sensitive algorithms"
    )
    return {
        "should_scale": True,
        "method": method,
        "columns": numeric_columns,
        "description": (
            f"Scaling {len(numeric_columns)} numeric column(s) with {method} scaler "
            f"for '{downstream_algorithm}' — {reason}."
        ),
    }


def apply_auto_scaling_strategy(
    df: pd.DataFrame,
    downstream_algorithm: str,
    numeric_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, str, Optional[object]]:
    """Runs auto_decide_scaling_strategy and applies it. Returns
    (new_df, description, fitted_scaler_or_None)."""
    decision = auto_decide_scaling_strategy(df, downstream_algorithm, numeric_columns)
    if not decision["should_scale"]:
        return df.copy(), decision["description"], None
    work, _, scaler = scale_columns(df, decision["columns"], decision["method"])
    return work, decision["description"], scaler


# ---------------------------------------------------------------------------
# 7. DROP COLUMNS
# (mechanic copied from app12.py tab_preprocessing "7. Drop Columns")
# ---------------------------------------------------------------------------

def drop_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, str]:
    work = df.drop(columns=columns)
    return work, f"Dropped columns: {columns}"


# ---------------------------------------------------------------------------
# ORCHESTRATOR — full automatic pass, no manual input required
# ---------------------------------------------------------------------------

def auto_preprocess(
    df: pd.DataFrame,
    downstream_algorithm: str = "linear_regression",
    drop_threshold: float = 0.55,
) -> tuple[pd.DataFrame, list[str]]:
    """
    The "automatic by default" preprocessing pass for fusion runs — chains
    every auto-decide step above with zero manual input, in this order:
      1. Missing values (median / mode fill, or drop column if too sparse)
      2. Outliers (IQR-flag only; auto-cap only clearly-impossible values)
      3. Scaling (chosen by downstream_algorithm)

    Duplicate removal, dtype conversion, and encoding are deliberately
    left out of the automatic pass — those need column-level intent
    (e.g. "is this really a duplicate?", "should this be a category or a
    string?") that shouldn't be guessed blindly. They stay available as
    manual functions above, or as later, more targeted fusion steps.

    Returns (new_df, log) where log is a flat list of plain-English
    description strings — one per decision made, in order applied — ready
    to render as an inline, editable "here's what I did" list.
    """
    log: list[str] = []
    work = df.copy()

    work, missing_log = apply_auto_missing_value_strategy(work, drop_threshold=drop_threshold)
    log.extend(missing_log)

    numeric_cols_now = work.select_dtypes(include=[np.number]).columns.tolist()
    work, outlier_log, flagged = apply_auto_outlier_strategy(work, numeric_columns=numeric_cols_now)
    log.extend(outlier_log)
    for f in flagged:
        log.append(f["description"])

    numeric_cols_now = work.select_dtypes(include=[np.number]).columns.tolist()
    work, scale_desc, _ = apply_auto_scaling_strategy(work, downstream_algorithm, numeric_cols_now)
    log.append(scale_desc)

    return work, log
