"""
fusion_core.py
==============

The engine behind the Cross-Modal Fusion feature.

WHAT THIS FILE DOES (in plain terms):
1. Defines how the app "remembers" fusion data as the user clicks around
   (the session state structure).
2. Takes the user's CSV + ZIP of images and matches each row to the right
   image file, cleanly reporting anything that didn't match.
3. Takes the already-matched data and turns it into ONE combined table,
   where every column is labeled with where it came from (structured data,
   text, or image) so later steps (SHAP/LIME, experiment tracking) know
   which source drove which result.

WHAT THIS FILE DOES NOT DO:
- It does not do the actual text-to-numbers or image-to-numbers conversion.
  That already exists in nlp_app2.py and image_app.py. This file just calls
  those pipelines through two small "adapter" functions you plug in
  (see PLUGGING IN YOUR PIPELINES near the bottom).
- It does not train models, do SHAP/LIME, or preprocessing (median fill,
  outlier flagging, etc). Those are the next steps in the build order.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. SESSION STATE STRUCTURE
# ---------------------------------------------------------------------------
#
# In Streamlit, "session state" is just a dictionary that survives between
# clicks (normally every click reruns the whole script and forgets
# everything, so this dict is the exception).
#
# We keep one FusionSession object per user session. It holds:
#   - the raw inputs (csv + list of image files)
#   - the matching results (matched rows / unmatched rows)
#   - every combined table ever built, keyed by a version number, so the
#     user can go back to an earlier fused table without re-uploading
#     anything (mirrors the version-counter pattern already used in app12.py)
#
# NOTE: this is written as a plain dataclass so it's testable outside of
# Streamlit. In the actual app, you'd store one of these in
# st.session_state["fusion_session"].

@dataclass
class FusionSession:
    # --- raw inputs, set once on upload ---
    raw_csv: Optional[pd.DataFrame] = None
    image_files: dict[str, bytes] = field(default_factory=dict)  # filename -> file bytes

    # --- which columns the user told us to use ---
    id_col: Optional[str] = None            # column that uniquely identifies a row
    text_col: Optional[str] = None          # optional column with free text
    image_filename_col: Optional[str] = None  # optional column naming an image file

    # --- matching results (filled in by match_records) ---
    matched_df: Optional[pd.DataFrame] = None
    unmatched_df: Optional[pd.DataFrame] = None

    # --- every combined table built so far ---
    # key = version number (1, 2, 3...), value = the DataFrame at that version.
    # NOTE: this single dict holds every version regardless of which stage
    # produced it - the original fused table from build_combined_table(),
    # AND every preprocessed table from run_preprocessing() below. Keeping
    # them in one pool (rather than a separate dict per stage) is what lets
    # the UI offer a single "which version?" dropdown per stage, and lets a
    # user preprocess an already-preprocessed version if they want to.
    combined_versions: dict[int, pd.DataFrame] = field(default_factory=dict)
    current_version: int = 0

    # --- human-readable label per version, e.g. "Fused table",
    # "Preprocessed" - shown next to the version number in the UI's
    # dropdown instead of a bare "v2". Falls back to "Fused table" in
    # get_all_available_dfs() if a version has no explicit label. ---
    version_labels: dict[int, str] = field(default_factory=dict)

    # --- which version a derived version was built FROM, if any (e.g. a
    # preprocessed version points back at its source fused version). Not
    # present for original fused versions - lets the UI show lineage like
    # "v2 (from v1)" and lets run_preprocessing() re-run on top of an
    # already-preprocessed version if the user wants to chain steps. ---
    version_sources: dict[int, int] = field(default_factory=dict)

    # --- preprocessing decision log (plain-English strings from
    # shared_preprocessing.auto_preprocess()), keyed by the NEW version
    # number it produced - so the UI can show exactly what changed between
    # a fused version and its preprocessed counterpart. ---
    preprocessing_logs: dict[int, list[str]] = field(default_factory=dict)

    # --- latest EDA result per version, keyed by version number. EDA
    # doesn't produce a new table (nothing to version), so unlike
    # preprocessing this is "latest run wins" per version rather than
    # append-only. Each entry is {"target": ..., "result": <auto_eda()
    # return dict>} so the UI can show which target the ranking used. ---
    eda_results: dict[int, dict] = field(default_factory=dict)

    # --- every training run ever kicked off, in order, across every
    # version/target/task_type combo the user has tried. Append-only (not
    # keyed/overwritten) so multiple runs can sit side-by-side - this is
    # exactly the data the future "experiment tracking" roadmap item reads
    # from, so it's built this way from day one rather than "latest wins". ---
    training_runs: list[dict] = field(default_factory=list)

    def next_version(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None,
        source_version: Optional[int] = None,
    ) -> int:
        """
        Store a new combined table and return its version number.
        `label` is a short human-readable description (e.g. "Fused table",
        "Preprocessed") shown in the version picker; `source_version`
        records which version this one was derived from, if any (omitted
        for an original fused table with no preprocessing history yet).
        """
        self.current_version += 1
        self.combined_versions[self.current_version] = df
        if label:
            self.version_labels[self.current_version] = label
        if source_version is not None:
            self.version_sources[self.current_version] = source_version
        return self.current_version

    def get_all_available_dfs(self) -> dict[str, pd.DataFrame]:
        """
        Matches the existing app12.py pattern: one flat pool of every
        dataframe the user could choose from in a dropdown, regardless of
        which stage created it - now labeled so a preprocessed version
        reads as "v2 — Preprocessed" rather than an indistinguishable
        "Fused table v2".
        """
        pool = {}
        if self.raw_csv is not None:
            pool["Original upload"] = self.raw_csv
        for version, df in self.combined_versions.items():
            label = self.version_labels.get(version, "Fused table")
            pool[f"v{version} — {label}"] = df
        return pool


# ---------------------------------------------------------------------------
# 2. MATCHING: line up CSV rows with the right image files
# ---------------------------------------------------------------------------

def load_images_from_zip(zip_bytes: bytes) -> dict[str, bytes]:
    """
    Unpacks an uploaded ZIP of images into a simple dict:
        {"cat1.jpg": <raw file bytes>, "cat2.jpg": <raw file bytes>, ...}
    Uses just the filename (not folder paths inside the zip), since that's
    what the user's CSV will reference.
    """
    images = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            if name.endswith("/"):  # skip folder entries
                continue
            filename = name.split("/")[-1]
            images[filename] = z.read(name)
    return images


def match_records(
    csv_df: pd.DataFrame,
    image_files: dict[str, bytes],
    image_filename_col: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the uploaded CSV into two tables:
      - matched_df: rows that are ready to use (either they don't need an
        image at all, or their named image was found in the ZIP)
      - unmatched_df: rows that named an image file that wasn't found

    If image_filename_col is None (user didn't provide an image column),
    everything is considered matched automatically.
    """
    if image_filename_col is None or image_filename_col not in csv_df.columns:
        return csv_df.copy(), csv_df.iloc[0:0].copy()  # empty unmatched table

    found_mask = csv_df[image_filename_col].isin(image_files.keys())
    matched_df = csv_df[found_mask].copy()
    unmatched_df = csv_df[~found_mask].copy()
    return matched_df, unmatched_df


def unmatched_report(unmatched_df: pd.DataFrame) -> bytes:
    """
    Turns the unmatched rows into a downloadable CSV (as raw bytes), so the
    user can inspect exactly which rows were skipped and why, before
    deciding whether to Continue or Stop.
    """
    return unmatched_df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# 3. COMBINING: build one row per record with labeled columns
# ---------------------------------------------------------------------------
#
# PipelineFn is the "shape" of function this expects for turning raw text or
# raw image bytes into numeric columns. You already have real versions of
# these inside nlp_app2.py and image_app.py - you just need a thin wrapper
# around each that matches this shape.
#
#   TextPipelineFn:  takes a pandas Series of text -> returns a DataFrame of
#                     numeric columns, same row order/length as the input.
#   ImagePipelineFn: takes a list of raw image bytes -> returns a DataFrame
#                     of numeric columns, same row order/length as the input.

TextPipelineFn = Callable[[pd.Series], pd.DataFrame]
ImagePipelineFn = Callable[[list[bytes]], pd.DataFrame]


def build_combined_table(
    matched_df: pd.DataFrame,
    image_files: dict[str, bytes],
    id_col: str,
    text_col: Optional[str] = None,
    image_filename_col: Optional[str] = None,
    text_pipeline_fn: Optional[TextPipelineFn] = None,
    image_pipeline_fn: Optional[ImagePipelineFn] = None,
) -> pd.DataFrame:
    """
    Builds the single fused table: one row per record, with every column
    prefixed to show where it came from:
        struct__<column>   -> came straight from the uploaded CSV
        text__<feature>     -> produced by the text pipeline
        img__<feature>       -> produced by the image pipeline

    These prefixes are what let the SHAP/LIME step later say things like
    "this prediction was driven mostly by img__edge_density and
    struct__price" instead of just an unlabeled list of numbers.
    """
    # Start with the structured columns, always included.
    struct_part = matched_df.add_prefix("struct__")
    combined = struct_part.copy()
    combined[id_col] = matched_df[id_col].values  # keep a clean, unprefixed ID column

    # Add text-derived columns, if the user included a text column.
    if text_col is not None and text_pipeline_fn is not None:
        text_features = text_pipeline_fn(matched_df[text_col])
        text_features = text_features.reset_index(drop=True).add_prefix("text__")
        combined = pd.concat([combined.reset_index(drop=True), text_features], axis=1)

    # Add image-derived columns, if the user included an image column.
    if image_filename_col is not None and image_pipeline_fn is not None:
        ordered_bytes = [image_files[fn] for fn in matched_df[image_filename_col]]
        image_features = image_pipeline_fn(ordered_bytes)
        image_features = image_features.reset_index(drop=True).add_prefix("img__")
        combined = pd.concat([combined.reset_index(drop=True), image_features], axis=1)

    return combined


# ---------------------------------------------------------------------------
# PLUGGING IN YOUR PIPELINES
# ---------------------------------------------------------------------------
# The actual text/image logic lives in shared_features.py (a small, new,
# UI-free module - see that file's docstring for why it's separate from
# nlp_app2.py / image_app.py rather than importing those pages directly).
#
# Default for automatic (non-manual) fusion runs: Sentence Transformer
# embeddings for text - semantic, GPU-aware, matches the quality bar of the
# rest of the "auto" pipeline. TF-IDF/Count remain available as a manual
# override (faster, no GPU needed) via the same vectorize_text() function.

from shared_features import vectorize_text, extract_image_features_batch
from shared_preprocessing import auto_preprocess
from shared_eda import auto_eda, classify_column_utility
from shared_training import (
    auto_train_regression,
    auto_train_classification,
    auto_train_clustering,
)


def default_text_pipeline_fn(text_series: pd.Series) -> pd.DataFrame:
    """Auto-mode default: Sentence Transformer embeddings."""
    return vectorize_text(text_series, method="sentence_transformer")


def manual_text_pipeline_fn(
    text_series: pd.Series, method: str, **kwargs
) -> pd.DataFrame:
    """Manual-mode: user picks the method (and its settings) explicitly."""
    return vectorize_text(text_series, method=method, **kwargs)


def default_image_pipeline_fn(image_bytes_list: list[bytes]) -> pd.DataFrame:
    return extract_image_features_batch(image_bytes_list)


# Usage in the fusion page (auto mode):
#   combined = build_combined_table(
#       matched_df, session.image_files, id_col="record_id",
#       text_col="review_text", image_filename_col="image_filename",
#       text_pipeline_fn=default_text_pipeline_fn,
#       image_pipeline_fn=default_image_pipeline_fn,
#   )
#   session.next_version(combined, label="Fused table")
#
# Usage in manual mode (user picked TF-IDF with custom settings in the UI):
#   text_fn = lambda s: manual_text_pipeline_fn(s, method="tfidf", max_features=2000)
#   combined = build_combined_table(..., text_pipeline_fn=text_fn, ...)


# ---------------------------------------------------------------------------
# 4. PREPROCESSING / EDA / TRAINING — the "Run" button for each stage
# ---------------------------------------------------------------------------
#
# Same role as build_combined_table() above, one level further down the
# pipeline: each function here takes a FusionSession + which version to act
# on, calls straight into the matching shared_*.py module's auto_*()
# entry point (no new logic of its own - the decision-making lives in
# those modules), and records the result back onto the session so the UI
# has somewhere to read it from afterward.
#
# All three are only ever called on an explicit user action (the page's
# "Run" button for that stage) - nothing here runs automatically as a side
# effect of an earlier stage completing. The version to act on, and (for
# EDA/training) the target column and task type, are always passed in
# explicitly by the caller; this file makes no assumption about which one
# the user "probably" wants.

def run_preprocessing(
    session: FusionSession,
    source_version: int,
    downstream_algorithm: str = "linear_regression",
    drop_threshold: float = 0.55,
) -> int:
    """
    Runs shared_preprocessing.auto_preprocess() on
    session.combined_versions[source_version] and stores the result as a
    NEW version - it does not overwrite the source version, so both the
    raw fused table and its preprocessed counterpart stay independently
    selectable (e.g. for comparing a model trained on each).

    `downstream_algorithm` feeds auto_preprocess()'s scaling decision
    (tree-based -> skip scaling, else Standard/Robust). Since preprocessing
    runs before the user has necessarily chosen a task type or algorithm
    for training, this defaults to "linear_regression" - a scale-sensitive
    algorithm, so the safe assumption is "scale unless told otherwise".
    Advanced callers can pass a different hint (e.g. "random_forest") if
    they already know which algorithm they'll train with.

    Returns the new version number.
    """
    if source_version not in session.combined_versions:
        raise ValueError(f"No such version: {source_version!r}")

    source_df = session.combined_versions[source_version]
    processed_df, log = auto_preprocess(
        source_df, downstream_algorithm=downstream_algorithm, drop_threshold=drop_threshold
    )

    new_version = session.next_version(
        processed_df, label="Preprocessed", source_version=source_version
    )
    session.preprocessing_logs[new_version] = log
    return new_version


def run_eda(
    session: FusionSession,
    version: int,
    target: Optional[str] = None,
    top_n: int = 6,
) -> dict:
    """
    Runs shared_eda.auto_eda() on session.combined_versions[version] and
    caches the result on the session (overwriting any previous EDA run for
    that same version - EDA doesn't produce a new table, so there's
    nothing to keep multiple versions of the way preprocessing does).

    `target` is optional and entirely the user's choice (per-run, not
    stored as a session-wide default) - EDA works fine with no target
    (falls back to shared_eda's variety heuristic), and a user may want to
    check the predictability of several different candidate target
    columns before committing to one for training.

    Returns the auto_eda() result dict directly (overview / shown / more /
    skipped / correlation), unmodified - the caller renders it.
    """
    if version not in session.combined_versions:
        raise ValueError(f"No such version: {version!r}")

    df = session.combined_versions[version]
    result = auto_eda(df, target=target, top_n=top_n)
    session.eda_results[version] = {"target": target, "result": result}
    return result


def _default_feature_columns(
    df: pd.DataFrame, exclude: set[Optional[str]], numeric_cols: Optional[list[str]] = None
) -> list[str]:
    """
    Default feature set for run_training() when the caller doesn't supply
    an explicit feature_columns list: every numeric column except
    `exclude` (the target and/or the session's known id_col), further
    filtered through shared_eda.classify_column_utility() so a near-
    constant flag or an ID-like column doesn't silently end up as a model
    input just because it happened to be numeric.

    This matters concretely for fusion tables: build_combined_table()
    deliberately keeps a clean, unprefixed copy of the id column
    alongside its struct__<id_col> counterpart (for row tracking), which
    means the SAME identifier appears as two numeric columns. Without
    this filtering, both would be fed into the model as "features" -
    meaningless for prediction, and liable to trigger false
    multicollinearity red flags purely from the duplication.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in numeric_cols if c not in exclude]
    selected = []
    for c in candidates:
        useful, _reason = classify_column_utility(df[c], c)
        if useful:
            selected.append(c)
    return selected


def run_training(
    session: FusionSession,
    version: int,
    target: Optional[str],
    task_type: str,
    feature_columns: Optional[list[str]] = None,
) -> dict:
    """
    Runs the matching shared_training.py auto_train_*() entry point on
    session.combined_versions[version], and appends a compact record of
    the run to session.training_runs (append-only, so multiple runs across
    different versions/targets/task types can sit side-by-side for later
    comparison — see FusionSession.training_runs above).

    task_type: "regression" | "classification" | "clustering". `target`
    is required for regression/classification and ignored for clustering
    (clustering has no target). Both are always the user's explicit choice
    for this run - never inferred or reused from a previous call.

    feature_columns defaults to every numeric column except the target and
    the session's id column, further filtered by shared_eda's near-
    constant/ID-like detector (see _default_feature_columns() above) -
    mirrors app12.py's default feature-selection pattern, but ID-aware so
    fusion's duplicated id-column columns don't end up as model inputs.
    Pass an explicit list to override.

    Returns the underlying auto_train_*() result dict directly (model,
    metrics, log, ...), unmodified - the caller renders it.
    """
    if version not in session.combined_versions:
        raise ValueError(f"No such version: {version!r}")
    if task_type not in ("regression", "classification", "clustering"):
        raise ValueError(f"Unknown task_type: {task_type!r}")
    if task_type != "clustering" and not target:
        raise ValueError(f"task_type={task_type!r} requires a target column.")

    df = session.combined_versions[version]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if task_type == "clustering":
        exclude = {session.id_col} if session.id_col else set()
        feature_cols = feature_columns or _default_feature_columns(df, exclude, numeric_cols)
        result = auto_train_clustering(df, feature_cols)
    else:
        exclude = {target, session.id_col} if session.id_col else {target}
        feature_cols = feature_columns or _default_feature_columns(df, exclude, numeric_cols)
        if task_type == "regression":
            result = auto_train_regression(df, feature_cols, target)
        else:  # classification
            result = auto_train_classification(df, feature_cols, target)

    run_record = {
        "version": version,
        "task_type": task_type,
        "target": target if task_type != "clustering" else None,
        "features": feature_cols,
        "model_choice": result.get("model_choice") or result.get("algorithm"),
        "log": result["log"],
        "timestamp": time.time(),
    }
    if task_type == "regression":
        run_record["metric_name"], run_record["metric_value"] = "r2", result["r2"]
    elif task_type == "classification":
        run_record["metric_name"], run_record["metric_value"] = "accuracy", result["acc"]
    else:
        run_record["metric_name"] = "silhouette"
        run_record["metric_value"] = result.get("silhouette")

    session.training_runs.append(run_record)
    return result
