"""
fusion.py — Cross-Modal Fusion Page
====================================
Streamlit page wiring for the fusion feature. All decision-making logic
lives in fusion_core.py / shared_preprocessing.py / shared_eda.py /
shared_training.py / shared_features.py — this file is UI only, plus a
thin layer of session-local bookkeeping (below) that those shared modules
don't expose on their own.

WHY THIS FILE ALSO DOES A LITTLE EXTRA BOOKKEEPING:
Three things needed for a good end-to-end experience aren't exposed by
the shared/core modules as given, so this file adds them locally rather
than editing those (mostly page-agnostic, reusable) modules:

  1. Predicting on brand-new data requires REPLAYING the exact same
     pipeline used to build a version (which text vectorizer was fit,
     which columns got scaled and with what fitted scaler, which columns
     got median/mode-filled and with what value). fusion_core.py's
     build_combined_table()/run_preprocessing() don't persist these
     fitted artifacts anywhere retrievable after the fact — only the
     resulting DataFrame. So this file keeps its own parallel
     "recipe" per version (st.session_state.fusion_recipes), and reruns
     the same shared_preprocessing.py step functions itself (instead of
     going through fusion_core.run_preprocessing, which discards the
     fitted scaler) so it can capture what Predict needs.
  2. shared_features.vectorize_text() fits a brand-new TF-IDF/Count
     vectorizer on every call — fine for one-off use, but wrong for
     Predict (a second fit on new, likely much smaller, text would
     produce a different vocabulary than training). So for the manual
     TF-IDF/Count path only, this file fits the vectorizer once itself
     (identical settings to vectorize_text) and stores the fitted object
     for reuse. The Sentence Transformer path is untouched — it's
     already stateless-safe.
  3. fusion_core.run_training()'s session.training_runs log only keeps a
     compact record (metrics, feature list) — not the fitted model
     object. This file keeps its own parallel list, index-aligned with
     session.training_runs, so Predict can reach the actual model.

Everything else — matching, table-building, preprocessing decisions, EDA
column ranking, model choice/training — is unchanged, calling straight
into the shared modules as before.

CHANGELOG (this revision):
  - ID column is now optional (defaults to an internally-generated row
    number) instead of forcing the user to pick a real column as a
    pseudo-ID — most structured datasets (e.g. Iris) have no natural ID.
  - Multiple CSVs can be uploaded and merged (join type + per-pair keys),
    mirroring app12.py's Data Manager merge tab.
  - "Build Fused Table" is a manual click only when an image ZIP is in
    play (feature extraction is slow enough to justify gating). With no
    images, the table now builds itself reactively, matching what this
    docstring's UI caption already promised for row-matching.
  - The Train Model tab now shows every auto-suggested parameter (model
    choice, train/test split, CV, K-Means k via the elbow/silhouette
    curve, DBSCAN eps/min_samples) BEFORE running, adjustable in place —
    this was always the platform's intended core UX pattern (see the
    project's design-decisions notes) but the Train Model tab hadn't
    wired it up yet; it previously only exposed the blind auto_train_*()
    path.
  - Classification now shows the target's Label-Encoding map + a preview
    of the encoded column before training (One-Hot doesn't apply to a
    single target column for standard scikit-learn classifiers — that's
    explained inline rather than offered as a nonfunctional option).
  - Classification's model menu gained Multinomial Naïve Bayes and Linear
    SVC (mirrored from nlp_app2.py's Text Classification tab), with the
    same non-negative-input guard nlp_app2.py uses for Naïve Bayes.
  - A "Feature source" filter (All / Structured only / Text only / Image
    only) lets training isolate a single modality — the fusion equivalent
    of nlp_app2.py training purely on vectorized text.
  - EDA now excludes text__ embedding columns from charts/correlation
    (still used automatically at training time) and gained a separate
    "📝 Text Analysis" section (length distributions, n-grams, word cloud,
    KWIC search) mirroring nlp_app2.py's Text EDA tab, in a new UI-free
    shared_text_eda.py module.
"""

import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import fusion_core as fc
import shared_training as strain
import shared_text_eda as stext
from shared_preprocessing import (
    auto_decide_missing_value_strategy,
    apply_auto_missing_value_strategy,
    auto_decide_outlier_strategy,
    apply_auto_outlier_strategy,
    apply_auto_scaling_strategy,
)
from shared_eda import numeric_columns, categorical_columns


# ============================================================
# Page Config & Styling (matches the rest of the app)
# ============================================================
try:
    st.set_page_config(
        page_title="Cross-Modal Fusion",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #1a1a2e; }
.stApp { background-color: #f8f7f4; color: #1a1a2e; }

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e8e4de;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }

.dashboard-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem; font-weight: 700;
    color: #1a1a2e; letter-spacing: -0.5px; margin-bottom: 2px;
}
.dashboard-subtitle { font-size: 0.92rem; color: #6b7280; margin-bottom: 20px; }

.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #ffffff; padding: 5px;
    border-radius: 12px; border: 1px solid #e8e4de;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6b7280; border-radius: 8px;
    font-size: 13px; font-weight: 500; padding: 8px 16px; transition: all 0.15s ease;
}
.stTabs [aria-selected="true"] { background: #1a1a2e !important; color: #ffffff !important; }

h1, h2, h3 { font-family: 'Playfair Display', serif; color: #1a1a2e; }
h1 { font-size: 1.7rem; font-weight: 700; }
h2 { font-size: 1.2rem; color: #374151; }
h3 { font-size: 1rem; color: #4b5563; }

.stButton > button {
    background: #1a1a2e; color: #ffffff; border: none; border-radius: 8px;
    font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: 14px;
    padding: 9px 22px; transition: all 0.2s ease;
    box-shadow: 0 2px 6px rgba(26,26,46,0.15);
}
.stButton > button:hover {
    background: #2d2d4e; box-shadow: 0 4px 12px rgba(26,26,46,0.25); transform: translateY(-1px);
}

.active-banner {
    background: linear-gradient(135deg, #ffffff, #f0eee9);
    border: 1px solid #e8e4de; border-left: 4px solid #1a1a2e;
    border-radius: 8px; padding: 12px 18px;
    font-family: 'DM Mono', monospace; font-size: 12px;
    color: #374151; margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.stat-card {
    background: #ffffff; border: 1px solid #e8e4de; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    font-family: 'DM Sans', sans-serif; font-size: 14px; color: #374151;
}
.stat-after  { border-left: 4px solid #10b981; }
.stat-warn   { border-left: 4px solid #f59e0b; }
.stat-info   { border-left: 4px solid #3b82f6; }

[data-testid="stExpander"] {
    background: #ffffff; border: 1px solid #e8e4de;
    border-radius: 10px; margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

.log-entry {
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: #374151; background: #f8f7f4;
    border: 1px solid #e8e4de; border-radius: 6px;
    padding: 5px 10px; margin: 3px 0;
}
.status-badge {
    display:inline-block; border-radius:6px; padding:3px 10px;
    font-size:12px; font-weight:600; margin-top:6px; margin-bottom:2px;
}
.status-badge-ok   { background:#d1fae5; color:#065f46; border:1px solid #10b981; }
.status-badge-fail { background:#fee2e2; color:#991b1b; border:1px solid #ef4444; }
.status-detail { font-size:11px; color:#6b7280; margin-left:6px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
def init_session_state():
    if "fusion_session" not in st.session_state:
        st.session_state.fusion_session = fc.FusionSession()
    defaults = {
        "fusion_gate_passed": False,
        # {version: recipe_dict} — see module docstring, point 1.
        "fusion_recipes": {},
        # index-aligned with session.training_runs — see point 3.
        "fusion_train_results": [],
        # signature of the last reactively-auto-built table (no-image
        # case) — see tab_upload_fuse()'s "5. Build" section.
        "fusion_auto_build_sig": None,
        # {version: auto_text_eda() result} — text-EDA cache, mirrors how
        # session.eda_results caches structured EDA per version.
        "fusion_text_eda_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()
session: fc.FusionSession = st.session_state.fusion_session


# ============================================================
# Small shared UI helpers
# ============================================================
def download_csv_button(df, filename="result.csv", label="⬇ Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def stateful_apply_button(label, status_key, key=None, **kwargs):
    clicked = st.button(label, key=key or f"btn_{status_key}", **kwargs)
    status = st.session_state.get(status_key)
    if status:
        cls = "status-badge-ok" if status["ok"] else "status-badge-fail"
        icon = "✅" if status["ok"] else "❌"
        st.markdown(
            f"<span class='status-badge {cls}'>{icon} {status['label']}</span>"
            f"<span class='status-detail'>{status['msg']}</span>",
            unsafe_allow_html=True,
        )
    return clicked


def set_apply_status(status_key, ok, label, msg=""):
    st.session_state[status_key] = {"ok": bool(ok), "label": label, "msg": msg, "ts": time.time()}


def pt():
    return "plotly_white"


# ============================================================
# Text-vectorization helpers — see module docstring, point 2.
# ============================================================
def _fit_manual_text_vectorizer(text_series, method, max_features, ngram_range):
    docs = text_series.fillna("").astype(str).tolist()
    if method == "tfidf":
        vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
    else:
        vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    matrix = vec.fit_transform(docs)
    df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())
    return df, vec


def _transform_manual_text_vectorizer(vec, text_series):
    docs = text_series.fillna("").astype(str).tolist()
    matrix = vec.transform(docs)
    return pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())


def _make_manual_text_fn(vec):
    def fn(s):
        return _transform_manual_text_vectorizer(vec, s)
    return fn


# ============================================================
# Recipe bookkeeping — see module docstring, points 1 & 2.
# ============================================================
def _get_recipe_chain(version):
    """Walk version_sources back to the root 'fused' recipe. Returns an
    ordered list [(version, recipe), ...] root-first, or None if any
    version in the chain has no recipe on file (built outside this
    session / session reset)."""
    chain = []
    v = version
    seen = set()
    while True:
        if v in seen:
            return None
        seen.add(v)
        recipe = st.session_state.fusion_recipes.get(v)
        if recipe is None:
            return None
        chain.append((v, recipe))
        if recipe["kind"] == "fused":
            break
        v = recipe["source_version"]
    chain.reverse()
    return chain


def _chain_has_scaling(version):
    chain = _get_recipe_chain(version)
    if chain is None:
        return None
    return any(r.get("scaler") is not None for _, r in chain if r["kind"] == "preprocessed")


def _replay_preprocessing(chain, work):
    """Applies every 'preprocessed' step in the chain, in order, to a
    freshly-built raw combined table — same fill values / dropped
    columns / capped floors / fitted scaler used at training time."""
    work = work.copy()
    for _, recipe in chain[1:]:
        if recipe["kind"] != "preprocessed":
            continue
        for col, val in recipe["fill_values"].items():
            if col in work.columns and val is not None:
                work[col] = work[col].fillna(val)
        drop_now = [c for c in recipe["dropped_columns"] if c in work.columns]
        if drop_now:
            work = work.drop(columns=drop_now)
        for col, floor in recipe["capped_columns"].items():
            if col in work.columns:
                work[col] = work[col].clip(lower=floor)
        if recipe.get("scaler") is not None and recipe.get("scaled_columns"):
            cols_present = [c for c in recipe["scaled_columns"] if c in work.columns]
            if cols_present:
                work[cols_present] = recipe["scaler"].transform(work[cols_present])
    return work


def _rebuild_features_for_prediction(version, new_csv_df, new_image_files):
    """Rebuilds struct__/text__/img__ feature columns for brand-new raw
    data, using the exact pipeline (+ fitted artifacts) that built
    `version`, then replays every preprocessing step on top."""
    chain = _get_recipe_chain(version)
    if chain is None:
        raise ValueError(
            "No pipeline recipe found for this version — it must have been built "
            "in this same session for Predict to replay its feature pipeline."
        )
    _, root_recipe = chain[0]
    id_col = root_recipe["id_col"]
    id_is_synthetic = root_recipe.get("id_is_synthetic", False)
    text_col = root_recipe["text_col"]
    image_col = root_recipe["image_col"]

    new_csv_df = new_csv_df.copy()
    if id_is_synthetic:
        # The training data had no real ID column either — row_id was
        # generated internally (see tab_upload_fuse()), so new data isn't
        # expected to carry it. Generate a fresh one instead of demanding
        # a column that was never meant to exist in the source data.
        new_csv_df[id_col] = range(len(new_csv_df))
    elif id_col is not None and id_col not in new_csv_df.columns:
        raise ValueError(f"New data is missing the ID column '{id_col}'.")

    for needed, name in [(text_col, "text"), (image_col, "image filename")]:
        if needed is not None and needed not in new_csv_df.columns:
            raise ValueError(f"New data is missing the {name} column '{needed}'.")

    combined = new_csv_df.add_prefix("struct__")
    combined[id_col] = new_csv_df[id_col].values

    if text_col is not None:
        if root_recipe["text_method"] == "sentence_transformer":
            text_features = fc.default_text_pipeline_fn(new_csv_df[text_col])
        else:
            vec = root_recipe.get("text_vectorizer")
            if vec is None:
                raise ValueError("No fitted text vectorizer was stored for this version.")
            text_features = _transform_manual_text_vectorizer(vec, new_csv_df[text_col])
        text_features = text_features.reset_index(drop=True).add_prefix("text__")
        combined = pd.concat([combined.reset_index(drop=True), text_features], axis=1)

    if image_col is not None:
        missing_imgs = [fn for fn in new_csv_df[image_col] if fn not in new_image_files]
        if missing_imgs:
            raise ValueError(
                f"{len(missing_imgs)} row(s) reference an image not found in the uploaded ZIP."
            )
        ordered_bytes = [new_image_files[fn] for fn in new_csv_df[image_col]]
        image_features = fc.default_image_pipeline_fn(ordered_bytes)
        image_features = image_features.reset_index(drop=True).add_prefix("img__")
        combined = pd.concat([combined.reset_index(drop=True), image_features], axis=1)

    return _replay_preprocessing(chain, combined)


def _run_preprocessing_with_recipe(session, source_version, downstream_algorithm, drop_threshold=0.55):
    """Same three-step pipeline as shared_preprocessing.auto_preprocess()
    (and therefore the same result fusion_core.run_preprocessing() would
    give), reimplemented here step-by-step so the fitted scaler and the
    exact fill values used can be captured into a recipe for Predict."""
    source_df = session.combined_versions[source_version]

    missing_decisions = auto_decide_missing_value_strategy(source_df, drop_threshold)
    fill_values = {}
    for d in missing_decisions:
        if d["action"] == "fill":
            c = d["column"]
            if d["strategy"] == "median":
                fill_values[c] = float(source_df[c].median())
            else:
                m = source_df[c].mode()
                fill_values[c] = m.iloc[0] if len(m) else None
    dropped_columns = [d["column"] for d in missing_decisions if d["action"] == "drop_column"]

    work, log = apply_auto_missing_value_strategy(
        source_df, decisions=missing_decisions, drop_threshold=drop_threshold
    )

    numeric_cols_now = work.select_dtypes(include=[np.number]).columns.tolist()
    outlier_decisions = auto_decide_outlier_strategy(work, numeric_columns=numeric_cols_now)
    work, outlier_log, flagged = apply_auto_outlier_strategy(
        work, decisions=outlier_decisions, numeric_columns=numeric_cols_now
    )
    log.extend(outlier_log)
    for f in flagged:
        log.append(f["description"])
    capped_columns = {d["column"]: d["floor"] for d in outlier_decisions if d["action"] == "cap_impossible"}

    numeric_cols_now = work.select_dtypes(include=[np.number]).columns.tolist()
    work, scale_desc, scaler = apply_auto_scaling_strategy(work, downstream_algorithm, numeric_cols_now)
    log.append(scale_desc)
    scaled_columns = numeric_cols_now if scaler is not None else []

    new_version = session.next_version(work, label="Preprocessed", source_version=source_version)
    session.preprocessing_logs[new_version] = log

    st.session_state.fusion_recipes[new_version] = {
        "kind": "preprocessed",
        "source_version": source_version,
        "fill_values": fill_values,
        "dropped_columns": dropped_columns,
        "capped_columns": capped_columns,
        "scaler": scaler,
        "scaled_columns": scaled_columns,
    }
    return new_version


# ============================================================
# EDA chart-card helpers
# ============================================================
_AUTO_TYPE_MAP = {
    "histogram_box": "histogram",
    "bar_counts": "bar_counts",
    "bar_counts_top_n": "bar_counts",
    "line_over_time": "line",
}
_CHART_TYPES = ["histogram", "bar_counts", "scatter", "line", "bar", "box", "violin", "pie"]


def _auto_relationship_specs(df, target=None, max_specs=4):
    """New relationship-chart picks (scatter / grouped violin / pie) that
    shared_eda.select_top_charts() doesn't produce on its own — that
    function only ever picks a chart type for ONE column at a time."""
    num_c = numeric_columns(df)
    cat_c = categorical_columns(df)
    specs = []

    if target and target in df.columns:
        if target in num_c:
            others = [c for c in num_c if c != target]
            if others:
                corrs = df[others + [target]].corr()[target].drop(target, errors="ignore").abs()
                corrs = corrs.dropna().sort_values(ascending=False)
                for c in corrs.head(2).index:
                    specs.append({
                        "chart_type": "scatter", "x": c, "y": target, "color": None,
                        "description": f"'{c}' vs target '{target}' — top correlated numeric feature.",
                    })
            low_card = [c for c in cat_c if 2 <= df[c].nunique(dropna=True) <= 8]
            for c in low_card[:1]:
                specs.append({
                    "chart_type": "violin", "x": c, "y": target, "color": c,
                    "description": f"'{target}' distribution grouped by '{c}'.",
                })
        else:
            for c in num_c[:2]:
                specs.append({
                    "chart_type": "violin", "x": target, "y": c, "color": target,
                    "description": f"'{c}' distribution grouped by target '{target}'.",
                })
    else:
        if len(num_c) >= 2:
            corr = df[num_c].corr().abs()
            corr_vals = corr.values.copy()
            np.fill_diagonal(corr_vals, 0.0)
            if corr_vals.size and np.nanmax(corr_vals) > 0:
                i, j = np.unravel_index(np.nanargmax(corr_vals), corr_vals.shape)
                specs.append({
                    "chart_type": "scatter", "x": num_c[i], "y": num_c[j], "color": None,
                    "description": f"Most correlated numeric pair: '{num_c[i]}' vs '{num_c[j]}'.",
                })

    for c in cat_c:
        n_unique = df[c].nunique(dropna=True)
        if 2 <= n_unique <= 6:
            specs.append({
                "chart_type": "pie", "x": c, "y": None, "color": None,
                "description": f"'{c}' composition ({n_unique} categories).",
            })

    return specs[:max_specs]


def _render_chart_card(df, spec, key_prefix):
    """Renders one auto-picked chart with editable X / Y / color / type
    controls beneath it, defaulting to the auto pick — matches app12.py's
    Custom Chart Builder controls, just pre-filled instead of blank."""
    all_cols = df.columns.tolist()
    num_c = df.select_dtypes(include=[np.number]).columns.tolist()

    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        ctype = st.selectbox(
            "Chart type", _CHART_TYPES,
            index=_CHART_TYPES.index(spec["chart_type"]) if spec["chart_type"] in _CHART_TYPES else 0,
            key=f"{key_prefix}_type",
        )
    x_default = spec.get("x") or (all_cols[0] if all_cols else None)
    with c2:
        x = st.selectbox("X", all_cols, index=all_cols.index(x_default) if x_default in all_cols else 0,
                          key=f"{key_prefix}_x")

    y = None
    with c3:
        if ctype in ("scatter", "line", "bar", "box", "violin"):
            y_options = num_c if num_c else all_cols
            y_default = spec.get("y") or (y_options[0] if y_options else None)
            y = st.selectbox("Y", y_options,
                              index=y_options.index(y_default) if y_default in y_options else 0,
                              key=f"{key_prefix}_y")
        elif ctype == "pie":
            y_choice = st.selectbox("Values", ["(count)"] + num_c, key=f"{key_prefix}_y_pie")
            y = None if y_choice == "(count)" else y_choice

    with c4:
        color_options = ["None"] + all_cols
        color_default = spec.get("color") or "None"
        color = st.selectbox(
            "Color", color_options,
            index=color_options.index(color_default) if color_default in color_options else 0,
            key=f"{key_prefix}_color",
        )
        color = None if color == "None" else color

    try:
        fig = None
        if ctype == "histogram":
            fig = px.histogram(df, x=x, color=color, marginal="box", template=pt(),
                                title=f"Distribution — {x}")
        elif ctype == "bar_counts":
            counts = df[x].value_counts(dropna=True).head(30).reset_index()
            counts.columns = [x, "count"]
            fig = px.bar(counts, x=x, y="count", template=pt(), title=f"Value Counts — {x}")
        elif ctype == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, template=pt(), title=f"{y} vs {x}")
        elif ctype == "line":
            fig = px.line(df, x=x, y=y, color=color, template=pt(), title=f"{y} over {x}")
        elif ctype == "bar":
            fig = px.bar(df, x=x, y=y, color=color, barmode="group", template=pt(), title=f"{y} by {x}")
        elif ctype == "box":
            fig = px.box(df, x=x, y=y, color=color, template=pt(), title=f"{y} by {x}")
        elif ctype == "violin":
            fig = px.violin(df, x=x, y=y, color=color, box=True, template=pt(), title=f"{y} by {x}")
        elif ctype == "pie":
            if y:
                fig = px.pie(df, names=x, values=y, template=pt(), title=f"{x} composition")
            else:
                counts = df[x].value_counts(dropna=True).reset_index()
                counts.columns = [x, "count"]
                fig = px.pie(counts, names=x, values="count", template=pt(), title=f"{x} composition")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Couldn't render this chart with the current settings: {e}")


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧬 Cross-Modal Fusion")
        st.markdown("---")

        pool = session.get_all_available_dfs()
        if not pool:
            st.info("No fused table yet — start in the Upload & Fuse tab.")
        else:
            st.markdown("**Versions in this session**")
            for label, df in pool.items():
                st.caption(f"{label} — {df.shape[0]:,} × {df.shape[1]}")

        if session.training_runs:
            st.markdown("---")
            st.markdown("**Training Runs**")
            for i, r in enumerate(session.training_runs, 1):
                metric = f"{r['metric_name']}={r['metric_value']:.4f}" if r.get("metric_value") is not None else ""
                st.markdown(
                    f"<div class='log-entry'>#{i} · v{r['version']} · {r['task_type']} · "
                    f"{r['model_choice']} · {metric}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("""
**Navigation**
1. 📤 Upload & Fuse
2. 🔧 Preprocessing
3. 📊 EDA
4. 🧪 Train Model
        """)

        st.markdown("---")
        if st.button("🗑 Reset Fusion Session", key="btn_reset_fusion"):
            st.session_state.fusion_session = fc.FusionSession()
            st.session_state.fusion_recipes = {}
            st.session_state.fusion_train_results = []
            st.session_state.fusion_gate_passed = False
            st.session_state.fusion_auto_build_sig = None
            st.session_state.fusion_text_eda_cache = {}
            st.session_state.pop("fusion_merged_csv", None)
            st.rerun()


# ============================================================
# TAB 1 — Upload & Fuse
# ============================================================
def tab_upload_fuse():
    st.header("📤 Upload & Fuse")
    st.markdown(
        "Upload one or more CSVs of structured records, optionally with a text column and/or a "
        "column naming an image file in an accompanying ZIP. Matching and row-checks run "
        "automatically the moment your columns are picked. If there's no image column, the fused "
        "table itself builds automatically too — no extra click needed. Images are the one case "
        "that stays a manual step, since extracting image features can take a while."
    )

    # ── 1. Upload ──
    st.subheader("1. Upload Data")
    c1, c2 = st.columns(2)
    with c1:
        csv_files = st.file_uploader(
            "CSV file(s) — upload more than one to merge them", type=["csv"],
            accept_multiple_files=True, key="fusion_csv_upload",
        )
    with c2:
        zip_file = st.file_uploader("Image ZIP (optional)", type=["zip"], key="fusion_zip_upload")

    if not csv_files:
        st.info("Upload at least one CSV to begin.")
        return

    csv_dict = {f.name: pd.read_csv(f) for f in csv_files}

    if len(csv_dict) == 1:
        raw_csv = next(iter(csv_dict.values()))
    else:
        st.subheader("1b. Multiple Files Uploaded")
        st.caption(
            "More than one CSV was uploaded — merge them into a single table (like a database "
            "join), or just pick one of them to proceed with as-is."
        )
        names = list(csv_dict.keys())
        use_mode = st.radio(
            "How to proceed", ["🔗 Merge selected files", "📄 Use a single file"],
            key="fusion_multi_csv_mode", horizontal=True,
        )
        if use_mode.startswith("📄"):
            chosen = st.selectbox("File to use", names, key="fusion_single_csv_choice")
            raw_csv = csv_dict[chosen]
        else:
            files_to_merge = st.multiselect(
                "Files to merge (in order)", names, default=names, key="fusion_merge_files"
            )
            if len(files_to_merge) < 2:
                st.warning("Select at least 2 files to merge.")
                return
            join_type = st.selectbox("Join type", ["inner", "outer", "left", "right"],
                                      key="fusion_merge_join")
            merge_keys_list = []
            for i in range(len(files_to_merge) - 1):
                left_df = csv_dict[files_to_merge[i]]
                right_df = csv_dict[files_to_merge[i + 1]]
                common = list(set(left_df.columns) & set(right_df.columns))
                if not common:
                    st.warning(f"No common columns between **{files_to_merge[i]}** and **{files_to_merge[i+1]}**.")
                    merge_keys_list.append([])
                else:
                    keys = st.multiselect(
                        f"Join key(s): `{files_to_merge[i]}` ➜ `{files_to_merge[i+1]}`",
                        common, default=[common[0]], key=f"fusion_merge_key_{i}",
                    )
                    merge_keys_list.append(keys)

            if st.button("🔗 Merge Files", key="btn_fusion_merge"):
                try:
                    result = csv_dict[files_to_merge[0]].copy()
                    ok = True
                    for i in range(1, len(files_to_merge)):
                        keys = merge_keys_list[i - 1]
                        if not keys:
                            st.error(f"No join key(s) selected for step {i}.")
                            ok = False
                            break
                        result = pd.merge(result, csv_dict[files_to_merge[i]], on=keys, how=join_type)
                    if ok:
                        st.session_state.fusion_merged_csv = result
                        st.success(f"✅ Merged → {result.shape[0]:,} rows × {result.shape[1]} cols")
                except Exception as e:
                    st.error(f"Merge failed: {e}")

            if "fusion_merged_csv" not in st.session_state:
                st.info("Click **Merge Files** to continue.")
                return
            raw_csv = st.session_state.fusion_merged_csv
            with st.expander("📋 Preview merged result", expanded=False):
                st.dataframe(raw_csv.head(50), use_container_width=True)

    image_files = fc.load_images_from_zip(zip_file.read()) if zip_file else {}
    session.image_files = image_files

    with st.expander("📋 Preview data", expanded=False):
        st.dataframe(raw_csv.head(50), use_container_width=True)
    if image_files:
        st.caption(f"📦 {len(image_files):,} image(s) found in ZIP.")

    # ── 2. Column roles ──
    st.subheader("2. Column Roles")
    all_cols = raw_csv.columns.tolist()
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        id_choice = st.selectbox(
            "ID column", ["Auto — use row number (no ID column in this data)"] + all_cols,
            key="fusion_id_col",
            help="Most datasets (like Iris) have no natural ID column — leave this on Auto "
                 "unless your data has a real unique identifier you want carried through to "
                 "predictions.",
        )
    with cc2:
        text_col = st.selectbox("Text column (optional)", ["None"] + all_cols, key="fusion_text_col")
        text_col = None if text_col == "None" else text_col
    with cc3:
        image_col = st.selectbox(
            "Image filename column (optional)", ["None"] + all_cols,
            key="fusion_image_col", disabled=not image_files,
            help="Disabled until you upload an image ZIP." if not image_files else None,
        )
        image_col = None if image_col == "None" else image_col

    id_is_synthetic = id_choice.startswith("Auto")
    if id_is_synthetic:
        id_col = "row_id"
        raw_csv = raw_csv.copy()
        if id_col not in raw_csv.columns:
            raw_csv[id_col] = range(len(raw_csv))
    else:
        id_col = id_choice

    session.raw_csv = raw_csv
    session.id_col, session.text_col, session.image_filename_col = id_col, text_col, image_col

    # ── 3. Row matching (automatic, reactive) ──
    st.subheader("3. Row Matching")
    matched_df, unmatched_df = fc.match_records(raw_csv, image_files, image_col)
    session.matched_df, session.unmatched_df = matched_df, unmatched_df

    m1, m2 = st.columns(2)
    m1.metric("Matched rows", f"{len(matched_df):,}")
    m2.metric("Unmatched rows", f"{len(unmatched_df):,}")

    gate_needed = image_col is not None and len(unmatched_df) > 0
    if gate_needed:
        st.warning(f"{len(unmatched_df):,} row(s) named an image that wasn't found in the ZIP.")
        with st.expander("📋 View unmatched rows", expanded=not st.session_state.fusion_gate_passed):
            st.dataframe(unmatched_df.head(50), use_container_width=True)
        st.download_button(
            "⬇ Download skipped-rows report", fc.unmatched_report(unmatched_df),
            file_name="unmatched_rows.csv", mime="text/csv", key="dl_unmatched",
        )
        if not st.session_state.fusion_gate_passed:
            gc1, gc2 = st.columns(2)
            with gc1:
                if st.button("✅ Continue with matched rows only", key="btn_gate_continue"):
                    st.session_state.fusion_gate_passed = True
                    st.rerun()
            with gc2:
                if st.button("🛑 Stop — I'll fix the CSV and re-upload", key="btn_gate_stop"):
                    return
            return
        st.caption("Continuing with matched rows only.")
    else:
        st.session_state.fusion_gate_passed = True
        if image_col is not None:
            st.success("Every row matched cleanly — no images missing.")

    if len(matched_df) == 0:
        st.error("No matched rows to build from.")
        return

    # ── 4. Text vectorization (only if a text column is set) ──
    text_pipeline_fn = None
    text_method_for_recipe = None
    text_vectorizer_for_recipe = None
    text_settings_for_recipe = {}

    if text_col is not None:
        st.subheader("4. Text Vectorization Method")
        method_choice = st.radio(
            "Method",
            ["Auto (Sentence Transformer — semantic, GPU-aware)", "Manual (TF-IDF / Count)"],
            key="fusion_text_method",
        )
        if method_choice.startswith("Auto"):
            text_pipeline_fn = fc.default_text_pipeline_fn
            text_method_for_recipe = "sentence_transformer"
        else:
            tm1, tm2, tm3 = st.columns(3)
            with tm1:
                vec_type = st.selectbox("Vectorizer", ["tfidf", "count"], key="fusion_text_vec_type")
            with tm2:
                max_feat = st.number_input("Max features", 100, 20000, 2000, step=100, key="fusion_text_max_feat")
            with tm3:
                ngram_max = st.select_slider("N-gram max", options=[1, 2, 3], value=1, key="fusion_text_ngram")
            fitted_vec = None
            try:
                _, fitted_vec = _fit_manual_text_vectorizer(
                    matched_df[text_col], vec_type, max_feat, (1, ngram_max)
                )
            except Exception as e:
                st.error(f"Couldn't fit vectorizer: {e}")
            text_pipeline_fn = _make_manual_text_fn(fitted_vec)
            text_method_for_recipe = vec_type
            text_vectorizer_for_recipe = fitted_vec
            text_settings_for_recipe = {"max_features": max_feat, "ngram_range": (1, ngram_max)}
        st.divider()

    image_pipeline_fn = fc.default_image_pipeline_fn if image_col is not None else None
    uses_images = image_col is not None and len(image_files) > 0

    def _make_fused_recipe():
        return {
            "kind": "fused",
            "id_col": id_col,
            "id_is_synthetic": id_is_synthetic,
            "text_col": text_col,
            "image_col": image_col,
            "text_method": text_method_for_recipe,
            "text_vectorizer": text_vectorizer_for_recipe,
            "text_settings": text_settings_for_recipe,
        }

    # ── 5. Build ──
    # Images stay a manual, explicit "Build" click — feature extraction is
    # slow enough to be worth gating, and the mismatch gate above already
    # requires a click anyway. Structured-only or structured+text data has
    # neither cost, so it builds itself the moment the inputs above are set
    # (matching the "no extra click needed" promise above) — re-building
    # only when something the table actually depends on changes.
    if uses_images:
        st.subheader("5. Build Fused Table")
        st.caption("Image feature extraction takes a moment, so this step stays a manual click.")
        if stateful_apply_button("🚀 Build Fused Table", "status_build_fusion"):
            try:
                with st.spinner("Building fused table…"):
                    combined = fc.build_combined_table(
                        matched_df, image_files, id_col=id_col,
                        text_col=text_col, image_filename_col=image_col,
                        text_pipeline_fn=text_pipeline_fn, image_pipeline_fn=image_pipeline_fn,
                    )
                    new_version = session.next_version(combined, label="Fused table")
                    st.session_state.fusion_recipes[new_version] = _make_fused_recipe()
                set_apply_status(
                    "status_build_fusion", True, "Built",
                    f"v{new_version} — {combined.shape[0]:,} rows × {combined.shape[1]} cols",
                )
            except Exception as e:
                set_apply_status("status_build_fusion", False, "Failed", str(e))
            st.rerun()

        pool = session.get_all_available_dfs()
        fused_labels = [l for l in pool if l != "Original upload"]
        if fused_labels:
            st.markdown("---")
            preview_label = st.selectbox("Preview a built version", fused_labels, index=len(fused_labels) - 1,
                                          key="fusion_build_preview")
            preview_df = pool[preview_label]
            st.dataframe(preview_df.head(50), use_container_width=True)
            download_csv_button(preview_df, "fused_table.csv")
    else:
        st.subheader("5. Fused Table")
        sig = (
            int(pd.util.hash_pandas_object(matched_df, index=True).sum()),
            id_choice, text_col, image_col,
            text_method_for_recipe, tuple(sorted(text_settings_for_recipe.items())),
        )
        if st.session_state.get("fusion_auto_build_sig") != sig:
            try:
                combined = fc.build_combined_table(
                    matched_df, image_files, id_col=id_col,
                    text_col=text_col, image_filename_col=image_col,
                    text_pipeline_fn=text_pipeline_fn, image_pipeline_fn=None,
                )
                new_version = session.next_version(combined, label="Fused table")
                st.session_state.fusion_recipes[new_version] = _make_fused_recipe()
                st.session_state.fusion_auto_build_sig = sig
                st.session_state.fusion_auto_build_version = new_version
            except Exception as e:
                st.error(f"Couldn't build the table: {e}")
                return

        v = st.session_state.get("fusion_auto_build_version")
        if v and v in session.combined_versions:
            built_df = session.combined_versions[v]
            st.success(f"✅ Table ready — v{v} ({built_df.shape[0]:,} rows × {built_df.shape[1]} cols)")
            with st.expander("📋 Preview built table", expanded=False):
                st.dataframe(built_df.head(50), use_container_width=True)
            download_csv_button(built_df, "fused_table.csv")


# ============================================================
# TAB 2 — Preprocessing
# ============================================================
_ALGO_HINT_OPTIONS = {
    "Linear / Logistic Regression, Ridge, Lasso, SVM, KNN, PCA (scale-sensitive)": "linear_regression",
    "Random Forest / Decision Tree / Gradient Boosting (tree-based — skip scaling)": "random_forest",
}


def tab_preprocessing():
    st.header("🔧 Preprocessing")

    if not session.combined_versions:
        st.warning("Build a fused table in the Upload & Fuse tab first.")
        return

    labels = list(session.combined_versions.keys())
    default_label = labels[-1]
    source_version = st.selectbox(
        "📁 Version to preprocess", labels,
        index=labels.index(default_label),
        format_func=lambda v: f"v{v} — {session.version_labels.get(v, 'Fused table')} "
                               f"({session.combined_versions[v].shape[0]:,} × {session.combined_versions[v].shape[1]})",
        key="fusion_pp_source",
    )
    source_df = session.combined_versions[source_version]
    st.markdown(f"**{source_df.shape[0]:,} rows × {source_df.shape[1]} columns**")
    with st.expander("📋 View selected version", expanded=False):
        st.dataframe(source_df.head(100), use_container_width=True)

    algo_label = st.selectbox(
        "Downstream algorithm (guides the scaling decision)",
        list(_ALGO_HINT_OPTIONS.keys()), key="fusion_pp_algo_hint",
    )
    algo_hint = _ALGO_HINT_OPTIONS[algo_label]

    if stateful_apply_button("🚀 Run Automatic Preprocessing", "status_run_pp"):
        try:
            with st.spinner("Preprocessing…"):
                new_version = _run_preprocessing_with_recipe(session, source_version, algo_hint)
            set_apply_status(
                "status_run_pp", True, "Done",
                f"v{new_version} — {len(session.preprocessing_logs.get(new_version, []))} step(s)",
            )
        except Exception as e:
            set_apply_status("status_run_pp", False, "Failed", str(e))
        st.rerun()

    logs = {v: session.preprocessing_logs.get(v, []) for v in session.combined_versions
            if v in session.preprocessing_logs and session.version_sources.get(v) == source_version}
    if logs:
        st.markdown("---")
        st.subheader("Preprocessed results from this version")
        for v, steps in logs.items():
            with st.expander(f"v{v} — {len(steps)} step(s)", expanded=False):
                for i, step in enumerate(steps, 1):
                    st.markdown(f"<div class='log-entry'>✓ Step {i}: {step}</div>", unsafe_allow_html=True)
                pdf = session.combined_versions[v]
                st.dataframe(pdf.head(50), use_container_width=True)
                download_csv_button(pdf, f"preprocessed_v{v}.csv")


# ============================================================
# TAB 3 — EDA
# ============================================================
def tab_eda():
    st.header("📊 EDA")

    if not session.combined_versions:
        st.warning("Build a fused table in the Upload & Fuse tab first.")
        return

    labels = list(session.combined_versions.keys())
    version = st.selectbox(
        "📁 Version to explore", labels, index=len(labels) - 1,
        format_func=lambda v: f"v{v} — {session.version_labels.get(v, 'Fused table')}",
        key="fusion_eda_version",
    )
    full_df = session.combined_versions[version]

    # text__ embedding columns are dozens-to-hundreds of individually
    # meaningless numeric columns — great for training, useless (and
    # slow) for one-column-at-a-time charts. Excluded here only; still
    # used automatically at training time. See "📝 Text Analysis" below
    # for text-specific insights instead.
    text_embed_cols = [c for c in full_df.columns if c.startswith("text__")]
    df = full_df.drop(columns=text_embed_cols) if text_embed_cols else full_df
    if text_embed_cols:
        st.caption(
            f"ℹ️ {len(text_embed_cols):,} text embedding column(s) excluded from the charts below — "
            "still used automatically during model training. See **📝 Text Analysis** further down "
            "for text-specific insights."
        )

    all_cols = df.columns.tolist()
    target_choice = st.selectbox("Target column (optional — sharpens chart relevance)",
                                  ["None"] + all_cols, key="fusion_eda_target")
    target = None if target_choice == "None" else target_choice
    top_n = st.slider("How many columns to auto-chart", 3, 12, 6, key="fusion_eda_topn")

    if stateful_apply_button("📊 Run EDA", "status_run_eda"):
        try:
            with st.spinner("Analyzing…"):
                result = fc.run_eda(session, version, target=target, top_n=top_n, exclude_prefixes=["text__"])
            set_apply_status("status_run_eda", True, "Done", f"v{version}")
        except Exception as e:
            set_apply_status("status_run_eda", False, "Failed", str(e))
        st.rerun()

    cached = session.eda_results.get(version)
    if not cached:
        st.info("Click **Run EDA** to analyze this version.")
    else:
        result = cached["result"]

        st.subheader("Overview")
        st.dataframe(result["overview"], use_container_width=True)

        if result["skipped"]:
            with st.expander(f"⏭ {len(result['skipped'])} column(s) skipped (near-constant / ID-like)", expanded=False):
                for s in result["skipped"]:
                    st.caption(f"**{s['column']}** — {s['reason']}")

        st.markdown("---")
        st.subheader("Charts")
        st.caption(
            "Shown automatically based on relevance to your target (or a variety heuristic "
            "if no target is set). Change the axes, color, or chart type on any card below."
        )

        chart_specs = []
        for entry in result["shown"]:
            ctype = _AUTO_TYPE_MAP.get(entry["chart_type"], "histogram")
            spec = {"chart_type": ctype, "x": entry["column"], "y": None, "color": None,
                    "description": entry["description"]}
            if ctype == "line":
                numeric_target = target if (target and target in df.select_dtypes(include=[np.number]).columns) else None
                if numeric_target:
                    spec["y"] = numeric_target
                else:
                    spec["chart_type"] = "bar_counts"
            chart_specs.append(spec)

        chart_specs.extend(_auto_relationship_specs(df, target))

        for i, spec in enumerate(chart_specs):
            st.caption(spec["description"])
            _render_chart_card(df, spec, key_prefix=f"eda_v{version}_chart{i}")
            st.markdown("")

        if result.get("more"):
            with st.expander(f"➕ {len(result['more'])} more column(s) worth a look", expanded=False):
                for j, entry in enumerate(result["more"]):
                    st.caption(entry["description"])
                    ctype = _AUTO_TYPE_MAP.get(entry["chart_type"], "histogram")
                    spec = {"chart_type": ctype, "x": entry["column"], "y": None, "color": None,
                            "description": entry["description"]}
                    _render_chart_card(df, spec, key_prefix=f"eda_v{version}_more{j}")

        if result.get("correlation"):
            st.markdown("---")
            st.subheader("Correlations")
            st.plotly_chart(result["correlation"]["heatmap"], use_container_width=True)
            st.dataframe(result["correlation"]["top_pairs"], use_container_width=True)

    # ── Text Analysis (separate from the numeric-EDA flow above) ──
    text_col_name = session.text_col
    raw_text_col = f"struct__{text_col_name}" if text_col_name else None
    if raw_text_col and raw_text_col in full_df.columns:
        st.markdown("---")
        st.subheader("📝 Text Analysis")
        st.caption(f"Text column: **{text_col_name}**. Mirrors the NLP Dashboard's Text EDA tab.")
        if stateful_apply_button("📝 Analyze Text", "status_run_text_eda"):
            with st.spinner("Analyzing text…"):
                text_result = stext.auto_text_eda(full_df[raw_text_col])
            st.session_state.setdefault("fusion_text_eda_cache", {})[version] = text_result

        cached_text = st.session_state.get("fusion_text_eda_cache", {}).get(version)
        if not cached_text:
            st.info("Click **Analyze Text** to see length, n-gram, and word-cloud charts.")
        else:
            tc1, tc2 = st.columns(2)
            with tc1:
                st.plotly_chart(cached_text["word_count_fig"], use_container_width=True)
            with tc2:
                st.plotly_chart(cached_text["char_count_fig"], use_container_width=True)

            st.markdown("**N-gram Frequency**")
            ngc1, ngc2 = st.columns([1, 3])
            with ngc1:
                ng_n = st.selectbox("N-gram type", [1, 2, 3],
                                     format_func=lambda n: {1: "Unigrams", 2: "Bigrams", 3: "Trigrams"}[n],
                                     key=f"text_eda_ngn_{version}")
                top_k = st.slider("Top K", 10, 50, 20, key=f"text_eda_topk_{version}")
            with ngc2:
                fig_ng = stext.build_ngram_chart(full_df[raw_text_col], ng_n, top_k)
                if fig_ng is not None:
                    st.plotly_chart(fig_ng, use_container_width=True)
                else:
                    st.info("Not enough tokens for this n-gram size.")

            st.markdown("**Word Cloud**")
            wcc1, wcc2 = st.columns([1, 3])
            with wcc1:
                max_words = st.slider("Max words", 50, 300, 150, key=f"text_eda_maxwords_{version}")
                bg = st.color_picker("Background", "#f8f7f4", key=f"text_eda_bg_{version}")
            with wcc2:
                wc_arr = stext.build_wordcloud_array(full_df[raw_text_col], max_words, bg)
                if wc_arr is not None:
                    st.image(wc_arr, use_container_width=True)
                else:
                    st.info("Not enough text, or the `wordcloud` package isn't installed.")

            st.markdown("**Key Word In Context (KWIC)**")
            kwic_q = st.text_input("Search term", key=f"text_eda_kwic_q_{version}")
            kwic_window = st.slider("Context window (words each side)", 3, 15, 7, key=f"text_eda_kwic_win_{version}")
            if kwic_q.strip():
                kwic_df = stext.kwic_search(full_df[raw_text_col], kwic_q, window=kwic_window)
                if kwic_df.empty:
                    st.info(f"No matches for '{kwic_q}'.")
                else:
                    st.caption(f"{len(kwic_df):,} occurrence(s) found (max 500 shown).")
                    st.dataframe(kwic_df, use_container_width=True)


# ============================================================
# TAB 4 — Train Model
# ============================================================
def _render_regression_result(res):
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{res['r2']:.4f}")
    m2.metric("RMSE", f"{res['rmse']:.4f}")
    m3.metric("MAE", f"{res['mae']:.4f}")
    if res.get("cv"):
        c1, c2 = st.columns(2)
        c1.metric("CV Mean R²", f"{res['cv']['cv_r2_mean']:.4f}")
        c2.metric("CV Mean RMSE", f"{res['cv']['cv_rmse_mean']:.4f}")
    fig = px.scatter(x=res["y_test"], y=res["y_pred"], template=pt(),
                      labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
    lo, hi = float(np.min(res["y_test"])), float(np.max(res["y_test"]))
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#ef4444", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    if res.get("coefficients"):
        cf = res["coefficients"]
        fig_c = px.bar(x=cf["features"], y=cf["coefficients"], template=pt(), title="Feature Coefficients")
        st.plotly_chart(fig_c, use_container_width=True)


def _render_classification_result(res):
    st.metric("Accuracy", f"{res['acc']:.4f}")
    if res.get("cv"):
        c1, c2 = st.columns(2)
        c1.metric("CV Mean Accuracy", f"{res['cv']['cv_accuracy_mean']:.4f}")
        c2.metric("CV Std", f"±{res['cv']['cv_accuracy_std']:.4f}")
    st.dataframe(pd.DataFrame(res["report"]).T.round(4), use_container_width=True)
    fig = px.imshow(res["cm"], text_auto=True, template=pt(), color_continuous_scale="Blues",
                     x=res["present_class_names"], y=res["present_class_names"], title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    if res.get("feature_importances") is not None:
        fig_i = px.bar(x=res["features"], y=res["feature_importances"], template=pt(),
                        title="Feature Importances")
        st.plotly_chart(fig_i, use_container_width=True)


def _render_clustering_result(res):
    if res.get("silhouette") is not None:
        st.metric("Silhouette Score", f"{res['silhouette']:.4f}")
    if res.get("n_clusters") is not None:
        c1, c2 = st.columns(2)
        c1.metric("Clusters found", res["n_clusters"])
        c2.metric("Noise points", res.get("n_noise", 0))
    counts = pd.Series(res["labels"].astype(str)).value_counts().sort_index().reset_index()
    counts.columns = ["cluster", "count"]
    fig = px.bar(counts, x="cluster", y="count", template=pt(), title="Documents per Cluster")
    st.plotly_chart(fig, use_container_width=True)


def _feature_source_options(df):
    """Which modality filters make sense for this version's columns — used
    to let a user isolate training to "Text only" (the fusion equivalent
    of nlp_app2.py's Text Classification tab training purely on
    vectorized text) or "Image only", instead of always mixing every
    modality's features together."""
    has_text = any(c.startswith("text__") for c in df.columns)
    has_img = any(c.startswith("img__") for c in df.columns)
    opts = ["All features", "Structured only"]
    if has_text:
        opts.append("Text only")
    if has_img:
        opts.append("Image only")
    return opts, has_text, has_img


def _filter_by_source(cols, source_choice, id_col):
    if source_choice == "Structured only":
        return [c for c in cols if c.startswith("struct__") or c == id_col]
    if source_choice == "Text only":
        return [c for c in cols if c.startswith("text__")]
    if source_choice == "Image only":
        return [c for c in cols if c.startswith("img__")]
    return cols


def tab_training():
    st.header("🧪 Train Model")

    if not session.combined_versions:
        st.warning("Build a fused table in the Upload & Fuse tab first.")
        return

    labels = list(session.combined_versions.keys())
    version = st.selectbox(
        "📁 Version to train on", labels, index=len(labels) - 1,
        format_func=lambda v: f"v{v} — {session.version_labels.get(v, 'Fused table')}",
        key="fusion_train_version",
    )
    df = session.combined_versions[version]

    task_type = st.radio("Task", ["regression", "classification", "clustering"], horizontal=True,
                          key="fusion_train_task")

    source_opts, has_text, has_img = _feature_source_options(df)
    if len(source_opts) > 2:
        source_choice = st.radio(
            "Feature source", source_opts, horizontal=True, key="fusion_train_source",
            help="Isolate training to one modality — e.g. 'Text only' trains purely on the text "
                 "embeddings, the way the NLP dashboard's Text Classification tab would. 'All "
                 "features' (default) mixes every modality together in one model.",
        )
    else:
        source_choice = "All features"

    # Embedding dimensions make nonsensical prediction targets — only
    # struct__/img__/id columns are offered as a target.
    non_text_cols = [c for c in df.columns if not c.startswith("text__")]

    target = None
    if task_type != "clustering":
        if task_type == "regression":
            target_candidates = [c for c in numeric_columns(df) if not c.startswith("text__")]
        else:
            target_candidates = non_text_cols
        if not target_candidates:
            st.warning("No suitable target columns available in this version.")
            return
        target = st.selectbox("Target column", target_candidates, key="fusion_train_target")

    numeric_c = numeric_columns(df)
    exclude = {session.id_col} if session.id_col else set()
    if target:
        exclude = exclude | {target}
    source_filtered = _filter_by_source(numeric_c, source_choice, session.id_col)
    default_feats = fc._default_feature_columns(df, exclude, source_filtered)
    override_features = st.multiselect(
        "Feature columns (defaults to every useful column in the selected source, minus target/ID)",
        [c for c in source_filtered if c != target], default=default_feats, key="fusion_train_features",
    )
    features = override_features or default_feats
    if not features:
        st.info("No feature columns available — pick a different feature source or version.")
        return

    st.markdown("---")

    # ============================================================
    # REGRESSION
    # ============================================================
    if task_type == "regression":
        model_df = df[features + [target]].dropna()
        n = len(model_df)
        if n < 5:
            st.error(f"Only {n} usable row(s) after dropping missing values — need more data.")
            return
        X = model_df[features].values
        y = model_df[target].values

        model_decision = strain.auto_decide_regression_model(df, features)
        split_decision = strain.auto_decide_split_and_cv(n)

        st.subheader("Training Configuration")
        st.caption("Auto-suggested from your data — adjust anything below, or just run it as-is.")
        rc1, rc2, rc3 = st.columns(3)
        model_options = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Polynomial Regression"]
        with rc1:
            model_choice = st.selectbox(
                "Model", model_options, index=model_options.index(model_decision["model_choice"]),
                key="fusion_reg_model",
            )
        with rc2:
            test_pct = st.slider("Test size (%)", 10, 40, int(round(split_decision["test_size"] * 100)),
                                  key="fusion_reg_test_pct")
        with rc3:
            use_cv = st.checkbox("Cross-validation", value=split_decision["use_cv"], key="fusion_reg_use_cv")
        cv_folds = (
            st.slider("CV folds", 3, 10, split_decision["cv_folds"], key="fusion_reg_cv_folds")
            if use_cv else None
        )
        alpha, poly_degree = 1.0, 2
        if model_choice == "Ridge Regression":
            alpha = st.slider("Alpha", 0.01, 100.0, 1.0, key="fusion_reg_ridge_alpha")
        elif model_choice == "Lasso Regression":
            alpha = st.slider("Alpha", 0.0001, 10.0, 0.1, key="fusion_reg_lasso_alpha")
        elif model_choice == "Polynomial Regression":
            poly_degree = st.slider("Degree", 2, 5, 2, key="fusion_reg_poly_deg")
        st.caption(f"💡 {model_decision['description']}")
        st.caption(f"💡 {split_decision['description']}")

        if stateful_apply_button("🚀 Run Training", "status_run_training"):
            try:
                with st.spinner("Training…"):
                    results = strain.train_regression_model(
                        X, y, model_choice, alpha=alpha, poly_degree=poly_degree, test_size=test_pct / 100,
                    )
                    log = [
                        f"Model: {model_choice}" + (
                            f" (alpha={alpha})" if model_choice in ("Ridge Regression", "Lasso Regression") else ""
                        ),
                        f"Test size: {test_pct}% (auto-suggested {int(round(split_decision['test_size']*100))}%)",
                    ]
                    if use_cv:
                        cv_model = strain.build_regression_model(model_choice, alpha, poly_degree)
                        cv_results = strain.cross_validate_regression(cv_model, X, y, cv_folds=cv_folds)
                        results["cv"] = cv_results
                        log.append(
                            f"Cross-validation ({cv_folds}-fold): mean R² {cv_results['cv_r2_mean']:.4f} "
                            f"(±{cv_results['cv_r2_std']:.4f})"
                        )
                    results["coefficients"] = strain.extract_regression_coefficients(
                        results["model"], model_choice, features
                    )
                    results["features"] = features
                    results["target"] = target
                    results["log"] = log
                    fc.record_training_run(session, version, "regression", target, features, results)
                    st.session_state.setdefault("fusion_train_results", []).append(results)
                set_apply_status("status_run_training", True, "Done", f"R²={results['r2']:.4f}")
            except Exception as e:
                set_apply_status("status_run_training", False, "Failed", str(e))
            st.rerun()

    # ============================================================
    # CLASSIFICATION
    # ============================================================
    elif task_type == "classification":
        model_df = df[features + [target]].dropna()
        n = len(model_df)
        if n < 5:
            st.error(f"Only {n} usable row(s) after dropping missing values — need more data.")
            return

        target_series = model_df[target]
        is_cat_target = not pd.api.types.is_numeric_dtype(target_series)

        st.subheader("Target Encoding")
        if is_cat_target:
            y_preview, class_names_preview, _ = strain.encode_classification_target(target_series)
            st.caption(
                f"**'{target}'** is categorical, so it's **Label-Encoded** (0…{len(class_names_preview)-1}) "
                "automatically before training — the correct encoding for a classification target, "
                "since scikit-learn classifiers expect one integer/label column, not a set of "
                "One-Hot columns. (One-Hot Encoding encodes categorical *feature* columns, not the "
                "thing being predicted — it doesn't apply to a single target column here.)"
            )
            mapping_df = pd.DataFrame({
                "original_label": class_names_preview,
                "encoded_value": range(len(class_names_preview)),
            })
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**Encoding map**")
                st.dataframe(mapping_df, use_container_width=True, hide_index=True)
            with mc2:
                st.markdown("**Preview (first 10 rows)**")
                preview_df = pd.DataFrame({
                    target: target_series.head(10).values,
                    f"{target}__encoded": pd.Series(y_preview, index=target_series.index).head(10).values,
                })
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
        else:
            st.caption(f"**'{target}'** is already numeric — used as-is, no encoding needed.")

        X = model_df[features].values
        nonneg = strain.is_nonnegative_matrix(X)

        model_options = [
            "Logistic Regression", "K-Nearest Neighbors", "SVM (RBF)", "Linear SVC",
            "Decision Tree", "Random Forest", "Multinomial Naïve Bayes",
        ]
        if not nonneg:
            model_options = [m for m in model_options if m != "Multinomial Naïve Bayes"]

        model_decision = strain.auto_decide_classification_model(n)
        default_model = model_decision["model_choice"]
        if default_model not in model_options:
            default_model = model_options[0]
        split_decision = strain.auto_decide_split_and_cv(n)

        st.subheader("Training Configuration")
        st.caption("Auto-suggested from your data — adjust anything below, or just run it as-is.")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            model_choice = st.selectbox("Model", model_options, index=model_options.index(default_model),
                                         key="fusion_cls_model")
        with rc2:
            test_pct = st.slider("Test size (%)", 10, 40, int(round(split_decision["test_size"] * 100)),
                                  key="fusion_cls_test_pct")
        with rc3:
            use_cv = st.checkbox("Cross-validation", value=split_decision["use_cv"], key="fusion_cls_use_cv")
        cv_folds = (
            st.slider("CV folds", 3, 10, split_decision["cv_folds"], key="fusion_cls_cv_folds")
            if use_cv else None
        )

        k_val, svm_c, dt_depth, rf_n, rf_d = 5, 1.0, 5, 100, 5
        if model_choice == "K-Nearest Neighbors":
            k_val = st.slider("Neighbors (k)", 1, 25, 5, key="fusion_cls_knn_k")
        elif model_choice == "SVM (RBF)":
            svm_c = st.slider("SVM C", 0.01, 100.0, 1.0, key="fusion_cls_svm_c")
        elif model_choice == "Decision Tree":
            dt_depth = st.slider("Max Depth", 1, 20, 5, key="fusion_cls_dt_depth")
        elif model_choice == "Random Forest":
            rf_n = st.slider("Trees", 10, 300, 100, key="fusion_cls_rf_n")
            rf_d = st.slider("Max Depth", 1, 20, 5, key="fusion_cls_rf_depth")
        if not nonneg:
            st.caption(
                "ℹ️ Multinomial Naïve Bayes is hidden — the active features contain negative values "
                "(e.g. Sentence Transformer embeddings or scaled columns), which it can't use."
            )
        st.caption(f"💡 {model_decision['description']}")
        st.caption(f"💡 {split_decision['description']}")

        if stateful_apply_button("🚀 Run Training", "status_run_training"):
            try:
                with st.spinner("Training…"):
                    y, class_names, label_encoder = strain.encode_classification_target(target_series)
                    results = strain.train_classification_model(
                        X, y, class_names, model_choice,
                        k=k_val, svm_c=svm_c, dt_depth=dt_depth, rf_n=rf_n, rf_d=rf_d,
                        test_size=test_pct / 100,
                    )
                    results["label_encoder"] = label_encoder
                    log = [
                        f"Model: {model_choice}",
                        f"Test size: {test_pct}% (auto-suggested {int(round(split_decision['test_size']*100))}%)",
                    ]
                    if is_cat_target:
                        log.append(f"Target '{target}' Label-Encoded ({len(class_names)} classes).")
                    if use_cv:
                        cv_model = strain.build_classification_model(model_choice, k_val, svm_c, dt_depth, rf_n, rf_d)
                        cv_results = strain.cross_validate_classification(cv_model, X, y, cv_folds=cv_folds)
                        results["cv"] = cv_results
                        log.append(
                            f"Cross-validation ({cv_folds}-fold): mean accuracy {cv_results['cv_accuracy_mean']:.4f} "
                            f"(±{cv_results['cv_accuracy_std']:.4f})"
                        )
                    results["feature_importances"] = strain.extract_classification_feature_importance(
                        results["model"], model_choice
                    )
                    results["features"] = features
                    results["target"] = target
                    results["log"] = log
                    fc.record_training_run(session, version, "classification", target, features, results)
                    st.session_state.setdefault("fusion_train_results", []).append(results)
                set_apply_status("status_run_training", True, "Done", f"acc={results['acc']:.4f}")
            except Exception as e:
                set_apply_status("status_run_training", False, "Failed", str(e))
            st.rerun()

    # ============================================================
    # CLUSTERING
    # ============================================================
    else:
        algo_choice = st.radio("Algorithm", ["K-Means", "DBSCAN"], horizontal=True, key="fusion_clust_algo")
        X_scaled, scaler, cleaned_df = strain.scale_for_clustering(df, features, missing_strategy="mean")

        if algo_choice == "K-Means":
            n_samples = X_scaled.shape[0]
            max_k = min(11, n_samples - 1)
            if max_k < 2:
                st.error(f"Not enough samples ({n_samples}) to cluster.")
                return
            st.subheader("Find k")
            with st.spinner("Computing elbow and silhouette curves…"):
                curve = strain.compute_kmeans_elbow_curve(X_scaled)
            best_idx = int(np.argmax(curve["silhouettes"]))
            best_k = curve["k_values"][best_idx]
            cc1, cc2 = st.columns(2)
            with cc1:
                fig_e = px.line(x=curve["k_values"], y=curve["inertias"], markers=True, template=pt(),
                                 title="Elbow Method", labels={"x": "k", "y": "Inertia"})
                st.plotly_chart(fig_e, use_container_width=True)
            with cc2:
                fig_s = px.line(x=curve["k_values"], y=curve["silhouettes"], markers=True, template=pt(),
                                 title="Silhouette Score", labels={"x": "k", "y": "Silhouette"})
                st.plotly_chart(fig_s, use_container_width=True)
            k_val = st.slider("Number of clusters (k)", min(curve["k_values"]), max(curve["k_values"]),
                               best_k, key="fusion_clust_k")
            st.caption(
                f"💡 Highest silhouette score is at k={best_k} ({curve['silhouettes'][best_idx]:.4f}) — "
                "pre-selected above, but pick any k on the slider (e.g. k=3 if you already know your "
                "data has 3 natural groups)."
            )
            if stateful_apply_button("🚀 Run K-Means", "status_run_training"):
                try:
                    with st.spinner(f"Clustering into {k_val} clusters…"):
                        results = strain.train_kmeans(X_scaled, k_val)
                        results["scaler"] = scaler
                        results["feature_cols"] = features
                        results["cleaned_df"] = cleaned_df
                        results["log"] = [f"Manually selected k={k_val} (elbow/silhouette suggested k={best_k})."]
                        fc.record_training_run(session, version, "clustering", None, features, results)
                        st.session_state.setdefault("fusion_train_results", []).append(results)
                    set_apply_status("status_run_training", True, "Done", f"silhouette={results['silhouette']:.4f}")
                except Exception as e:
                    set_apply_status("status_run_training", False, "Failed", str(e))
                st.rerun()
        else:
            suggestion = strain.auto_decide_dbscan_params(X_scaled)
            dc1, dc2 = st.columns(2)
            with dc1:
                eps = st.slider("Epsilon", 0.05, 5.0, float(suggestion["eps"]), step=0.05, key="fusion_dbscan_eps")
            with dc2:
                min_samples = st.slider("Min samples", 2, 20, int(suggestion["min_samples"]), key="fusion_dbscan_min")
            st.caption(f"💡 {suggestion['description']}")
            if stateful_apply_button("🚀 Run DBSCAN", "status_run_training"):
                try:
                    with st.spinner("Running DBSCAN…"):
                        results = strain.train_dbscan(X_scaled, eps, min_samples)
                        results["scaler"] = scaler
                        results["feature_cols"] = features
                        results["cleaned_df"] = cleaned_df
                        results["log"] = [f"eps={eps}, min_samples={min_samples} (auto-suggested, adjustable)."]
                        fc.record_training_run(session, version, "clustering", None, features, results)
                        st.session_state.setdefault("fusion_train_results", []).append(results)
                    set_apply_status("status_run_training", True, "Done", f"{results['n_clusters']} cluster(s)")
                except Exception as e:
                    set_apply_status("status_run_training", False, "Failed", str(e))
                st.rerun()

    train_results = st.session_state.get("fusion_train_results", [])
    if session.training_runs and train_results:
        last_run = session.training_runs[-1]
        last_result = train_results[-1]
        if last_run["version"] == version:
            st.markdown("---")
            st.subheader("📊 Latest Training Result")
            for line in last_run["log"]:
                st.markdown(f"<div class='log-entry'>✓ {line}</div>", unsafe_allow_html=True)
            if last_run["task_type"] == "regression":
                _render_regression_result(last_result)
            elif last_run["task_type"] == "classification":
                _render_classification_result(last_result)
            else:
                _render_clustering_result(last_result)

    # ──────────────────────────────────────────────────────
    # Predict on New Data
    # ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Predict on New Data")

    if not session.training_runs or not train_results:
        st.info("Train a model above first.")
        return

    run_labels = [
        f"Run {i+1}: v{r['version']} · {r['task_type']}"
        + (f" · target={r['target']}" if r["target"] else "")
        + f" · {r['model_choice']}"
        for i, r in enumerate(session.training_runs)
    ]
    run_idx = st.selectbox(
        "Which trained run to use?", list(range(len(run_labels))),
        format_func=lambda i: run_labels[i], index=len(run_labels) - 1, key="predict_run_select",
    )
    run_record = session.training_runs[run_idx]
    full_result = train_results[run_idx]
    p_version = run_record["version"]
    p_task = run_record["task_type"]
    feat_cols = full_result.get("features") or full_result.get("feature_cols", [])

    chain = _get_recipe_chain(p_version)
    if chain is None:
        st.warning(
            "This run's pipeline recipe isn't available (its version wasn't built in "
            "this session, or the session was reset) — Predict can't replay its feature "
            "pipeline. Re-build and re-train in this session to enable Predict."
        )
        return

    root_recipe = chain[0][1]
    has_scaling = _chain_has_scaling(p_version)
    all_struct_only = all(f.startswith("struct__") for f in feat_cols)
    uses_images = root_recipe["image_col"] is not None

    mode_options = ["📂 Upload file"]
    if all_struct_only and not has_scaling:
        mode_options.append("✏️ Manual entry")
    mode = st.radio("Input method", mode_options, key="predict_mode", horizontal=True)

    if mode == "✏️ Manual entry":
        manual_vals = {}
        chunks = [feat_cols[i:i + 3] for i in range(0, len(feat_cols), 3)]
        for chunk in chunks:
            row_cols = st.columns(len(chunk))
            for wcol, f in zip(row_cols, chunk):
                with wcol:
                    manual_vals[f] = st.number_input(f.replace("struct__", ""), value=0.0,
                                                       key=f"predict_manual_{f}")
        if st.button("🎯 Predict", key="btn_predict_manual"):
            try:
                X_new = np.array([[manual_vals[f] for f in feat_cols]])
                if p_task == "regression":
                    pred = strain.predict_regression(full_result["model"], X_new)[0]
                    st.success(f"**Predicted {run_record['target']}:** `{pred:.4f}`")
                elif p_task == "classification":
                    pred = strain.predict_classification(
                        full_result["model"], X_new, full_result.get("label_encoder")
                    )[0]
                    st.success(f"**Predicted {run_record['target']}:** `{pred}`")
                else:
                    train_X_scaled = (
                        full_result["scaler"].transform(full_result["cleaned_df"])
                        if full_result["algorithm"] == "DBSCAN" else None
                    )
                    cluster = strain.assign_clusters_to_new_data(
                        full_result["algorithm"], full_result["scaler"], X_new,
                        model=full_result.get("model"),
                        train_X_scaled=train_X_scaled, train_labels=full_result.get("labels"),
                    )[0]
                    st.success(f"**Assigned cluster:** `{cluster}`")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.caption(
            "Upload a CSV with the same columns as your original data"
            + (", plus a ZIP of any new images." if uses_images else ".")
        )
        new_csv_file = st.file_uploader("New CSV", type=["csv"], key="predict_new_csv")
        new_zip_file = None
        if uses_images:
            new_zip_file = st.file_uploader("New image ZIP", type=["zip"], key="predict_new_zip")

        if new_csv_file and st.button("🔮 Predict on Uploaded File", key="btn_predict_file"):
            try:
                new_csv_df = pd.read_csv(new_csv_file)
                new_image_files = fc.load_images_from_zip(new_zip_file.read()) if new_zip_file else {}
                work = _rebuild_features_for_prediction(p_version, new_csv_df, new_image_files)
                missing_feats = [f for f in feat_cols if f not in work.columns]
                if missing_feats:
                    st.error(f"Couldn't reconstruct feature(s): {', '.join(missing_feats)}")
                else:
                    X_new = work[feat_cols].fillna(0).values
                    out_df = new_csv_df.copy()
                    if p_task == "regression":
                        preds = strain.predict_regression(full_result["model"], X_new)
                        out_df[f"Predicted_{run_record['target']}"] = preds
                    elif p_task == "classification":
                        preds = strain.predict_classification(
                            full_result["model"], X_new, full_result.get("label_encoder")
                        )
                        out_df[f"Predicted_{run_record['target']}"] = preds
                        counts = pd.Series(preds).value_counts().reset_index()
                        counts.columns = ["class", "count"]
                        st.plotly_chart(
                            px.bar(counts, x="class", y="count", template=pt(),
                                   title="Predicted class distribution"),
                            use_container_width=True,
                        )
                    else:
                        train_X_scaled = (
                            full_result["scaler"].transform(full_result["cleaned_df"])
                            if full_result["algorithm"] == "DBSCAN" else None
                        )
                        preds = strain.assign_clusters_to_new_data(
                            full_result["algorithm"], full_result["scaler"], X_new,
                            model=full_result.get("model"),
                            train_X_scaled=train_X_scaled, train_labels=full_result.get("labels"),
                        )
                        out_df["Assigned_Cluster"] = preds.astype(str)
                    st.dataframe(out_df.head(50), use_container_width=True)
                    download_csv_button(out_df, "fusion_predictions.csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ============================================================
# Main
# ============================================================
def main():
    render_sidebar()

    st.markdown("<div class='dashboard-title'>Cross-Modal Fusion</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dashboard-subtitle'>"
        "Structured &nbsp;·&nbsp; Text &nbsp;·&nbsp; Image"
        "&nbsp;&nbsp;|&nbsp;&nbsp;"
        "Fuse &nbsp;·&nbsp; Preprocess &nbsp;·&nbsp; Explore &nbsp;·&nbsp; Model &nbsp;·&nbsp; Predict"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tabs = st.tabs(["📤 Upload & Fuse", "🔧 Preprocessing", "📊 EDA", "🧪 Train Model"])
    with tabs[0]: tab_upload_fuse()
    with tabs[1]: tab_preprocessing()
    with tabs[2]: tab_eda()
    with tabs[3]: tab_training()


if __name__ == "__main__":
    main()
