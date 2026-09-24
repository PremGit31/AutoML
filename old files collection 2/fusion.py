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
"""

import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import fusion_core as fc
import shared_training as strain
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
    text_col = root_recipe["text_col"]
    image_col = root_recipe["image_col"]

    for needed, name in [(id_col, "ID"), (text_col, "text"), (image_col, "image filename")]:
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
            st.rerun()


# ============================================================
# TAB 1 — Upload & Fuse
# ============================================================
def tab_upload_fuse():
    st.header("📤 Upload & Fuse")
    st.markdown(
        "Upload a CSV of structured records, optionally with a text column and/or a "
        "column naming an image file in an accompanying ZIP. Matching and row-checks "
        "run automatically the moment your columns are picked — no extra click needed "
        "if you have no image column at all."
    )

    # ── 1. Upload ──
    st.subheader("1. Upload Data")
    c1, c2 = st.columns(2)
    with c1:
        csv_file = st.file_uploader("CSV file", type=["csv"], key="fusion_csv_upload")
    with c2:
        zip_file = st.file_uploader("Image ZIP (optional)", type=["zip"], key="fusion_zip_upload")

    if csv_file is None:
        st.info("Upload a CSV to begin.")
        return

    raw_csv = pd.read_csv(csv_file)
    session.raw_csv = raw_csv
    image_files = fc.load_images_from_zip(zip_file.read()) if zip_file else {}
    session.image_files = image_files

    with st.expander("📋 Preview uploaded CSV", expanded=False):
        st.dataframe(raw_csv.head(50), use_container_width=True)
    if image_files:
        st.caption(f"📦 {len(image_files):,} image(s) found in ZIP.")

    # ── 2. Column roles ──
    st.subheader("2. Column Roles")
    all_cols = raw_csv.columns.tolist()
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        id_col = st.selectbox("ID column (required)", all_cols, key="fusion_id_col")
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

    # ── 5. Build ──
    st.subheader("5. Build Fused Table")
    if stateful_apply_button("🚀 Build Fused Table", "status_build_fusion"):
        try:
            with st.spinner("Building fused table…"):
                combined = fc.build_combined_table(
                    matched_df, image_files, id_col=id_col,
                    text_col=text_col, image_filename_col=image_col,
                    text_pipeline_fn=text_pipeline_fn, image_pipeline_fn=image_pipeline_fn,
                )
                new_version = session.next_version(combined, label="Fused table")
                st.session_state.fusion_recipes[new_version] = {
                    "kind": "fused",
                    "id_col": id_col,
                    "text_col": text_col,
                    "image_col": image_col,
                    "text_method": text_method_for_recipe,
                    "text_vectorizer": text_vectorizer_for_recipe,
                    "text_settings": text_settings_for_recipe,
                }
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
    df = session.combined_versions[version]

    all_cols = df.columns.tolist()
    target_choice = st.selectbox("Target column (optional — sharpens chart relevance)",
                                  ["None"] + all_cols, key="fusion_eda_target")
    target = None if target_choice == "None" else target_choice
    top_n = st.slider("How many columns to auto-chart", 3, 12, 6, key="fusion_eda_topn")

    if stateful_apply_button("📊 Run EDA", "status_run_eda"):
        try:
            with st.spinner("Analyzing…"):
                result = fc.run_eda(session, version, target=target, top_n=top_n)
            set_apply_status("status_run_eda", True, "Done", f"v{version}")
        except Exception as e:
            set_apply_status("status_run_eda", False, "Failed", str(e))
        st.rerun()

    cached = session.eda_results.get(version)
    if not cached:
        st.info("Click **Run EDA** to analyze this version.")
        return

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
    target = None
    if task_type != "clustering":
        num_c = numeric_columns(df) if task_type == "regression" else df.columns.tolist()
        target = st.selectbox("Target column", num_c, key="fusion_train_target")

    numeric_c = numeric_columns(df)
    default_feats = fc._default_feature_columns(
        df, {target, session.id_col} if target else ({session.id_col} if session.id_col else set()), numeric_c
    )
    override_features = st.multiselect(
        "Feature columns (defaults to every useful numeric column, minus target/ID)",
        [c for c in numeric_c if c != target], default=default_feats, key="fusion_train_features",
    )

    if stateful_apply_button("🚀 Run Training", "status_run_training"):
        try:
            with st.spinner(f"Training ({task_type})…"):
                result = fc.run_training(
                    session, version, target, task_type,
                    feature_columns=override_features or None,
                )
            st.session_state.setdefault("fusion_train_results", []).append(result)
            metric = (
                f"R²={result['r2']:.4f}" if task_type == "regression"
                else f"acc={result['acc']:.4f}" if task_type == "classification"
                else f"silhouette={result.get('silhouette')}"
            )
            set_apply_status("status_run_training", True, "Done", metric)
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
            if task_type == "regression":
                _render_regression_result(last_result)
            elif task_type == "classification":
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
