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
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional

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
    # key = version number (1, 2, 3...), value = the DataFrame at that version
    combined_versions: dict[int, pd.DataFrame] = field(default_factory=dict)
    current_version: int = 0

    def next_version(self, df: pd.DataFrame) -> int:
        """Store a new combined table and return its version number."""
        self.current_version += 1
        self.combined_versions[self.current_version] = df
        return self.current_version

    def get_all_available_dfs(self) -> dict[str, pd.DataFrame]:
        """
        Matches the existing app12.py pattern: one flat pool of every
        dataframe the user could choose from in a dropdown, regardless of
        which tab created it.
        """
        pool = {}
        if self.raw_csv is not None:
            pool["Original upload"] = self.raw_csv
        for version, df in self.combined_versions.items():
            pool[f"Fused table v{version}"] = df
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
# these inside nlp_app2.py and image_app.py — you just need a thin wrapper
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
# UI-free module — see that file's docstring for why it's separate from
# nlp_app2.py / image_app.py rather than importing those pages directly).
#
# Default for automatic (non-manual) fusion runs: Sentence Transformer
# embeddings for text — semantic, GPU-aware, matches the quality bar of the
# rest of the "auto" pipeline. TF-IDF/Count remain available as a manual
# override (faster, no GPU needed) via the same vectorize_text() function.

from shared_features import vectorize_text, extract_image_features_batch


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
#   session.next_version(combined)
#
# Usage in manual mode (user picked TF-IDF with custom settings in the UI):
#   text_fn = lambda s: manual_text_pipeline_fn(s, method="tfidf", max_features=2000)
#   combined = build_combined_table(..., text_pipeline_fn=text_fn, ...)
