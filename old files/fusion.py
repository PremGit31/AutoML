"""
Cross-Modal Fusion Dashboard — fusion.py
==========================================
Combines structured data, text, and images into one fused table, then lets
you preprocess it, explore it, and train a model on it — all through the
same "Run when you click Run" pattern as the rest of the platform.

Tabs:
  1. Upload & Fuse   — CSV + optional image ZIP, column selection, row
                        matching with a downloadable skipped-rows report
                        and a Continue/Stop gate, text vectorization method
                        (auto Sentence Transformer / manual TF-IDF/Count),
                        builds a versioned fused table.
  2. Preprocessing   — pick a fused/preprocessed version, pick a downstream-
                        algorithm hint (defaults to a safe "assume scaling
                        needed" choice — no extra click required), Run.
                        Produces a NEW version; the source is never mutated,
                        so you can compare a model trained on either.
  3. EDA             — pick a version, optionally pick a target column,
                        Run. Shows the auto-selected charts, what got
                        skipped and why, and a correlation heatmap.
  4. Train Model     — pick a version, task type, and (for regression/
                        classification) a target column, Run. Shows
                        metrics, the auto-picked model + why, and keeps
                        every run side-by-side for comparison.

WHAT THIS FILE DOES NOT DO:
No new logic lives here — every decision (which model, which chart, which
scaler) is made by fusion_core.py calling into the shared_*.py modules.
This file's job is purely: collect input, call the right fusion_core
function on a button click, render what comes back.
"""

import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import fusion_core as fc


# ---------------------------------------------------------------------------
# Page Config (wrapped like image_app.py/home.py, so this survives being
# imported more than once — e.g. by a future navigation hub)
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Cross-Modal Fusion",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #1a1a2e; }
        .stApp { background-color: #FAF6EF; }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #ECE6D8;
            box-shadow: 2px 0 12px rgba(0,0,0,0.03);
        }
        .dashboard-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.2rem; font-weight: 800; color: #1A1A1A; margin-bottom: 0.2rem;
        }
        .dashboard-subtitle { font-size: 0.95rem; color: #6B6B6B; margin-bottom: 1.2rem; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px; background: #FFFFFF; padding: 6px; border-radius: 12px;
            border: 1px solid #ECE6D8; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent; color: #6B6B6B; border-radius: 8px;
            font-size: 13px; font-weight: 500; padding: 8px 16px; transition: all 0.15s ease;
        }
        .stTabs [aria-selected="true"] { background: #1A1A2E !important; color: #FFFFFF !important; }

        .stButton > button {
            background: #1A1A2E; color: #FAF6EF; border: none; border-radius: 8px;
            font-weight: 600; font-size: 14px; padding: 8px 20px; transition: all 0.2s ease;
        }
        .stButton > button:hover { background: #3A3A3A; color: #FFFFFF; transform: translateY(-1px); }

        .active-banner {
            background: linear-gradient(135deg, #FFFFFF, #F5F1E8);
            border: 1px solid #ECE6D8; border-left: 4px solid #1A1A2E;
            border-radius: 8px; padding: 12px 18px; font-family: 'DM Mono', monospace;
            font-size: 12px; color: #374151; margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        .log-entry {
            font-family: 'DM Mono', monospace; font-size: 11px; color: #374151;
            background: #FAF6EF; border: 1px solid #ECE6D8; border-radius: 6px;
            padding: 5px 10px; margin: 3px 0;
        }
        .source-chip {
            display: inline-block; font-size: 0.78rem; font-weight: 600;
            padding: 0.28rem 0.7rem; border-radius: 999px; margin-right: 0.35rem;
        }
        .chip-struct { background: #EAF1FB; color: #3B5B8C; border: 1px solid #DCE7F7; }
        .chip-text   { background: #FBEAF5; color: #9C3E77; border: 1px solid #F5DDEC; }
        .chip-img    { background: #E6F7F5; color: #0F766E; border: 1px solid #CFEFEA; }

        .status-badge {
            display: inline-block; border-radius: 6px; padding: 3px 10px;
            font-size: 12px; font-weight: 600; margin-top: 6px; margin-bottom: 2px;
        }
        .status-badge-ok   { background: #d1fae5; color: #065f46; border: 1px solid #10b981; }
        .status-badge-fail { background: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
        .status-detail { font-size: 11px; color: #6b7280; margin-left: 6px; }

        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #ECE6D8; border-radius: 10px;
            padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        hr { border-color: #ECE6D8; margin: 20px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state():
    if "fusion_session" not in st.session_state:
        st.session_state.fusion_session = fc.FusionSession()
    defaults = {
        "fusion_pending_matched": None,
        "fusion_pending_unmatched": None,
        "fusion_pending_id_col": None,
        "fusion_pending_text_col": None,
        "fusion_pending_image_col": None,
        "fusion_gate_passed": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()
session: fc.FusionSession = st.session_state.fusion_session


# ---------------------------------------------------------------------------
# Small shared helpers (self-contained copies of the pattern already used
# in app12.py/nlp_app2.py — persistent status badge across reruns, and a
# CSV download button)
# ---------------------------------------------------------------------------

def stateful_apply_button(label, status_key, key=None, **button_kwargs):
    clicked = st.button(label, key=key or f"btn_{status_key}", **button_kwargs)
    status = st.session_state.get(status_key)
    if status:
        cls = "status-badge-ok" if status["ok"] else "status-badge-fail"
        icon = "✅" if status["ok"] else "❌"
        st.markdown(
            f"<span class='status-badge {cls}'>{icon} {status['msg_head']}</span>"
            f"<span class='status-detail'>{status['msg']}</span>",
            unsafe_allow_html=True,
        )
    return clicked


def set_apply_status(status_key, ok, msg_head="", msg=""):
    st.session_state[status_key] = {"ok": bool(ok), "msg_head": msg_head, "msg": msg}


def download_csv_button(df, filename, label="⬇ Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def render_log(log: list[str]):
    for line in log:
        st.markdown(f"<div class='log-entry'>✓ {line}</div>", unsafe_allow_html=True)


def version_picker(key: str, label: str = "📁 Version to use"):
    """Shared version dropdown used by Preprocessing/EDA/Train tabs.
    Returns (version_int, df) or (None, None) if nothing is available yet."""
    # "Original upload" isn't a fused table (no struct__/text__/img__
    # prefixes) — only offer actual combined_versions here.
    version_labels = {
        f"v{v} — {session.version_labels.get(v, 'Fused table')}": v
        for v in session.combined_versions
    }
    if not version_labels:
        st.info("No fused table yet — build one in the **Upload & Fuse** tab first.")
        return None, None
    choice = st.selectbox(label, list(version_labels.keys()), key=key)
    v = version_labels[choice]
    return v, session.combined_versions[v]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧬 Cross-Modal Fusion")
        st.caption("Combine structured, text, and image data into one model.")
        st.markdown("---")

        if session.combined_versions:
            st.markdown("**Fused table versions**")
            for v in session.combined_versions:
                label = session.version_labels.get(v, "Fused table")
                src = session.version_sources.get(v)
                src_note = f" (from v{src})" if src is not None else ""
                df = session.combined_versions[v]
                st.markdown(
                    f"<div class='log-entry'>v{v} — {label}{src_note}<br>"
                    f"{df.shape[0]:,} rows × {df.shape[1]} cols</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No fused table yet.")

        if session.training_runs:
            st.markdown("---")
            st.markdown(f"**Training runs** ({len(session.training_runs)})")
            for r in session.training_runs[-5:]:
                metric = (
                    f"{r['metric_name']}={r['metric_value']:.3f}"
                    if r["metric_value"] is not None else "n/a"
                )
                st.markdown(
                    f"<div class='log-entry'>v{r['version']} · {r['task_type']} · {metric}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        if st.button("🗑 Reset Fusion Session", key="btn_reset_fusion"):
            st.session_state.fusion_session = fc.FusionSession()
            for k in ("fusion_pending_matched", "fusion_pending_unmatched", "fusion_gate_passed"):
                st.session_state[k] = None if "matched" in k or "unmatched" in k else False
            st.rerun()

        st.markdown("---")
        st.markdown("""
**Navigation**
1. 📤 Upload & Fuse
2. 🔧 Preprocessing
3. 📊 EDA
4. 🤖 Train Model
        """)


# ══════════════════════════════════════════════════════════════
# TAB 1 — Upload & Fuse
# ══════════════════════════════════════════════════════════════
def tab_upload_fuse():
    st.header("📤 Upload & Fuse")
    st.markdown(
        "Upload a CSV of structured data, optionally with a text column and/or "
        "a ZIP of images referenced by filename. Rows get matched to their "
        "images, then everything is combined into one table with "
        "<span class='source-chip chip-struct'>struct__</span>"
        "<span class='source-chip chip-text'>text__</span>"
        "<span class='source-chip chip-img'>img__</span> prefixed columns.",
        unsafe_allow_html=True,
    )

    # ── 1. Upload ──
    st.subheader("1. Upload Files")
    c1, c2 = st.columns(2)
    with c1:
        csv_file = st.file_uploader("CSV file (required)", type=["csv"], key="fusion_csv_upload")
    with c2:
        zip_file = st.file_uploader("Image ZIP (optional)", type=["zip"], key="fusion_zip_upload")

    if csv_file is None:
        st.info("Upload a CSV to get started.")
        return

    try:
        raw_csv = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        return
    session.raw_csv = raw_csv
    st.success(f"Loaded **{csv_file.name}** — {raw_csv.shape[0]:,} rows × {raw_csv.shape[1]} cols")
    with st.expander("📋 Preview", expanded=False):
        st.dataframe(raw_csv.head(20), use_container_width=True)

    image_files = {}
    if zip_file is not None:
        try:
            image_files = fc.load_images_from_zip(zip_file.read())
            session.image_files = image_files
            st.success(f"Loaded **{len(image_files):,}** image(s) from {zip_file.name}")
        except Exception as e:
            st.error(f"Couldn't read ZIP: {e}")

    st.divider()

    # ── 2. Column selection ──
    st.subheader("2. Select Columns")
    cols = raw_csv.columns.tolist()
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        id_col = st.selectbox("ID column (required)", cols, key="fusion_id_col")
    with cc2:
        text_col = st.selectbox("Text column (optional)", ["None"] + cols, key="fusion_text_col")
        text_col = None if text_col == "None" else text_col
    with cc3:
        image_col_options = ["None"] + cols
        image_col = st.selectbox(
            "Image filename column (optional)", image_col_options, key="fusion_image_col",
            disabled=not image_files,
            help="Requires an image ZIP to be uploaded first." if not image_files else None,
        )
        image_col = None if image_col == "None" else image_col

    session.id_col = id_col
    session.text_col = text_col
    session.image_filename_col = image_col

    st.divider()

    # ── 3. Match records ──
    st.subheader("3. Match Records")
    if st.button("🔗 Match Rows to Images", key="btn_match_records"):
        matched_df, unmatched_df = fc.match_records(raw_csv, image_files, image_col)
        st.session_state.fusion_pending_matched = matched_df
        st.session_state.fusion_pending_unmatched = unmatched_df
        st.session_state.fusion_gate_passed = len(unmatched_df) == 0
        st.rerun()

    matched_df = st.session_state.fusion_pending_matched
    unmatched_df = st.session_state.fusion_pending_unmatched

    if matched_df is None:
        st.info("Click **Match Rows to Images** to continue.")
        return

    m1, m2 = st.columns(2)
    m1.metric("Matched rows", f"{len(matched_df):,}")
    m2.metric("Unmatched rows", f"{len(unmatched_df):,}")

    if len(unmatched_df) > 0:
        st.warning(
            f"{len(unmatched_df):,} row(s) named an image that wasn't found in the ZIP."
        )
        with st.expander("📋 View unmatched rows", expanded=True):
            st.dataframe(unmatched_df.head(50), use_container_width=True)
        st.download_button(
            "⬇ Download skipped-rows report", fc.unmatched_report(unmatched_df),
            file_name="unmatched_rows.csv", mime="text/csv",
        )
        gc1, gc2 = st.columns(2)
        with gc1:
            if st.button("✅ Continue with matched rows only", key="btn_gate_continue"):
                st.session_state.fusion_gate_passed = True
                st.rerun()
        with gc2:
            if st.button("🛑 Stop — I'll fix the CSV and re-upload", key="btn_gate_stop"):
                st.session_state.fusion_gate_passed = False
                st.session_state.fusion_pending_matched = None
                st.session_state.fusion_pending_unmatched = None
                st.rerun()
        if not st.session_state.fusion_gate_passed:
            return
    else:
        st.success("Every row matched cleanly.")

    st.divider()

    # ── 4. Text vectorization method (only relevant if a text column was picked) ──
    text_pipeline_fn = None
    if text_col is not None:
        st.subheader("4. Text Vectorization Method")
        method_choice = st.radio(
            "Method", ["Auto (Sentence Transformer — semantic, GPU-aware)", "Manual (TF-IDF / Count)"],
            key="fusion_text_method",
        )
        if method_choice.startswith("Auto"):
            text_pipeline_fn = fc.default_text_pipeline_fn
        else:
            tm1, tm2, tm3 = st.columns(3)
            with tm1:
                vec_type = st.selectbox("Vectorizer", ["tfidf", "count"], key="fusion_text_vec_type")
            with tm2:
                max_feat = st.number_input("Max features", 100, 20000, 2000, step=100, key="fusion_text_max_feat")
            with tm3:
                ngram_max = st.select_slider("N-gram max", options=[1, 2, 3], value=1, key="fusion_text_ngram")
            text_pipeline_fn = lambda s: fc.manual_text_pipeline_fn(
                s, method=vec_type, max_features=max_feat, ngram_range=(1, ngram_max)
            )
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
            set_apply_status(
                "status_build_fusion", True, "Built",
                f"v{new_version} — {combined.shape[0]:,} rows × {combined.shape[1]} cols",
            )
        except Exception as e:
            set_apply_status("status_build_fusion", False, "Failed", str(e))
        st.rerun()

    if session.combined_versions:
        latest_v = max(session.combined_versions)
        latest_df = session.combined_versions[latest_v]
        st.markdown(f"**Latest: v{latest_v}** — {latest_df.shape[0]:,} rows × {latest_df.shape[1]} cols")
        with st.expander("📋 Preview latest fused table", expanded=False):
            st.dataframe(latest_df.head(20), use_container_width=True)
        download_csv_button(latest_df, f"fused_v{latest_v}.csv")


# ══════════════════════════════════════════════════════════════
# TAB 2 — Preprocessing
# ══════════════════════════════════════════════════════════════
_ALGO_HINT_OPTIONS = {
    "Linear / Ridge / Lasso Regression (default)": "linear_regression",
    "Logistic Regression": "logistic_regression",
    "K-Nearest Neighbors": "knn",
    "SVM": "svm",
    "K-Means Clustering": "kmeans",
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
    "Gradient Boosting": "gradient_boosting",
}


def tab_preprocessing():
    st.header("🔧 Preprocessing")
    st.markdown(
        "Runs automatically: numeric missing values → median, categorical → "
        "most-frequent, columns over ~55% missing are dropped, outliers are "
        "flagged via IQR (only clearly-impossible values like a negative "
        "price get auto-corrected), and scaling is chosen based on the "
        "algorithm hint below. Produces a **new version** — your source "
        "table is never modified, so you can compare either."
    )

    source_version, source_df = version_picker("pp_version_select", "📁 Version to preprocess")
    if source_version is None:
        return

    st.markdown(f"**{source_df.shape[0]:,} rows × {source_df.shape[1]} cols**")
    with st.expander("📋 Preview", expanded=False):
        st.dataframe(source_df.head(20), use_container_width=True)

    algo_label = st.selectbox(
        "Downstream algorithm hint (affects the scaling decision only)",
        list(_ALGO_HINT_OPTIONS.keys()), key="pp_algo_hint",
        help="Tree-based algorithms skip scaling entirely. Everything else "
             "gets Standard or Robust scaling depending on outlier severity.",
    )
    algo_hint = _ALGO_HINT_OPTIONS[algo_label]

    if stateful_apply_button("🚀 Run Preprocessing", "status_run_preprocessing"):
        try:
            with st.spinner("Preprocessing…"):
                new_version = fc.run_preprocessing(session, source_version, downstream_algorithm=algo_hint)
            set_apply_status(
                "status_run_preprocessing", True, "Done",
                f"Created v{new_version} from v{source_version}",
            )
            st.session_state["pp_last_result_version"] = new_version
        except Exception as e:
            set_apply_status("status_run_preprocessing", False, "Failed", str(e))
        st.rerun()

    result_version = st.session_state.get("pp_last_result_version")
    if result_version and result_version in session.combined_versions:
        st.divider()
        st.subheader(f"Result — v{result_version}")
        render_log(session.preprocessing_logs.get(result_version, []))
        result_df = session.combined_versions[result_version]
        with st.expander("📋 View preprocessed table", expanded=False):
            st.dataframe(result_df.head(50), use_container_width=True)
        download_csv_button(result_df, f"preprocessed_v{result_version}.csv")


# ══════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════
def tab_eda():
    st.header("📊 EDA")
    st.markdown(
        "Charts are picked automatically by column type, ranked by relevance "
        "to a target (if you pick one), and near-constant/ID-like columns "
        "are skipped — see why below."
    )

    version, df = version_picker("eda_version_select", "📁 Version to explore")
    if version is None:
        return

    numeric_and_other = df.columns.tolist()
    target_choice = st.selectbox(
        "Target column (optional — improves chart ranking)",
        ["None"] + numeric_and_other, key="eda_target_select",
    )
    target = None if target_choice == "None" else target_choice
    top_n = st.slider("Charts to show by default", 2, 12, 6, key="eda_top_n")

    if stateful_apply_button("🚀 Run EDA", "status_run_eda"):
        try:
            with st.spinner("Analyzing…"):
                fc.run_eda(session, version, target=target, top_n=top_n)
            set_apply_status("status_run_eda", True, "Done", f"v{version}")
        except Exception as e:
            set_apply_status("status_run_eda", False, "Failed", str(e))
        st.rerun()

    cached = session.eda_results.get(version)
    if not cached:
        return

    result = cached["result"]
    used_target = cached["target"]
    st.divider()
    st.subheader(f"Results — v{version}" + (f" (target: {used_target})" if used_target else " (no target)"))

    st.markdown("**Overview**")
    st.dataframe(result["overview"], use_container_width=True)

    if result["skipped"]:
        with st.expander(f"⏭ Skipped columns ({len(result['skipped'])})", expanded=False):
            for s in result["skipped"]:
                st.markdown(f"<div class='log-entry'>{s['column']} — {s['reason']}</div>", unsafe_allow_html=True)

    st.markdown("**Charts**")
    for entry in result["shown"]:
        st.caption(entry["description"])
        if entry["figure"] is not None:
            st.plotly_chart(entry["figure"], use_container_width=True)

    if result["more"]:
        with st.expander(f"➕ Show {len(result['more'])} more chart(s)", expanded=False):
            for entry in result["more"]:
                st.caption(entry["description"])
                if entry["figure"] is not None:
                    st.plotly_chart(entry["figure"], use_container_width=True)

    if result["correlation"]:
        st.markdown("**Correlation**")
        st.plotly_chart(result["correlation"]["heatmap"], use_container_width=True)
        st.dataframe(result["correlation"]["top_pairs"], use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Train Model
# ══════════════════════════════════════════════════════════════
_TASK_TYPES = {"Regression": "regression", "Classification": "classification", "Clustering": "clustering"}


def _render_regression_result(result: dict):
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{result['r2']:.4f}")
    m2.metric("RMSE", f"{result['rmse']:.4f}")
    m3.metric("MAE", f"{result['mae']:.4f}")

    fig = px.scatter(
        x=result["y_test"], y=result["y_pred"], template="plotly_white",
        labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted",
        color_discrete_sequence=["#1a1a2e"],
    )
    lo, hi = min(result["y_test"].min(), result["y_pred"].min()), max(result["y_test"].max(), result["y_pred"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#ef4444", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    if result.get("cv"):
        cv = result["cv"]
        c1, c2 = st.columns(2)
        c1.metric("CV mean R²", f"{cv['cv_r2_mean']:.4f}")
        c2.metric("CV std R²", f"±{cv['cv_r2_std']:.4f}")

    if result.get("coefficients"):
        coef = result["coefficients"]
        fig_c = px.bar(
            x=coef["features"], y=coef["coefficients"], template="plotly_white",
            title="Feature Coefficients", color_discrete_sequence=["#374151"],
        )
        st.plotly_chart(fig_c, use_container_width=True)


def _render_classification_result(result: dict):
    st.metric("Accuracy", f"{result['acc']:.4f}")
    report_df = pd.DataFrame(result["report"]).T.round(4)
    st.dataframe(report_df, use_container_width=True)

    fig_cm = px.imshow(
        result["cm"], text_auto=True, template="plotly_white",
        x=result["present_class_names"], y=result["present_class_names"],
        color_continuous_scale="Blues", title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    if result.get("cv"):
        cv = result["cv"]
        c1, c2 = st.columns(2)
        c1.metric("CV mean accuracy", f"{cv['cv_accuracy_mean']:.4f}")
        c2.metric("CV std accuracy", f"±{cv['cv_accuracy_std']:.4f}")

    if result.get("feature_importances") is not None:
        fig_imp = px.bar(
            x=result["features"], y=result["feature_importances"], template="plotly_white",
            title="Feature Importances", color_discrete_sequence=["#374151"],
        )
        st.plotly_chart(fig_imp, use_container_width=True)


def _render_clustering_result(result: dict):
    if result["algorithm"] == "K-Means":
        c1, c2 = st.columns(2)
        c1.metric("k", result["k"])
        c2.metric("Silhouette", f"{result['silhouette']:.4f}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters found", result["n_clusters"])
        c2.metric("Noise points", result["n_noise"])
        c3.metric("Silhouette", f"{result['silhouette']:.4f}" if result["silhouette"] is not None else "n/a")

    labels = pd.Series(result["labels"]).astype(str)
    counts = labels.value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    fig = px.bar(counts, x="cluster", y="count", template="plotly_white",
                 title="Points per Cluster", color_discrete_sequence=["#1a1a2e"])
    st.plotly_chart(fig, use_container_width=True)

    feats = result["feature_cols"]
    if len(feats) >= 2:
        plot_df = result["cleaned_df"][[feats[0], feats[1]]].copy()
        plot_df["cluster"] = labels.values
        fig2 = px.scatter(
            plot_df, x=feats[0], y=feats[1], color="cluster", template="plotly_white",
            title=f"Clusters — {feats[0]} vs {feats[1]}",
        )
        st.plotly_chart(fig2, use_container_width=True)


def tab_training():
    st.header("🤖 Train Model")
    st.markdown(
        "Pick a version, a task, and (for regression/classification) a "
        "target — the model and its hyperparameters are chosen "
        "automatically and explained below."
    )

    version, df = version_picker("train_version_select", "📁 Version to train on")
    if version is None:
        return

    task_label = st.selectbox("Task", list(_TASK_TYPES.keys()), key="train_task_type")
    task_type = _TASK_TYPES[task_label]

    target = None
    if task_type != "clustering":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_targets = [c for c in df.columns if c != session.id_col]
        target = st.selectbox(
            f"Target column to {'predict' if task_type == 'regression' else 'classify'}",
            candidate_targets, key="train_target_select",
        )
        if task_type == "regression" and target not in numeric_cols:
            st.warning("Regression needs a numeric target column.")

    with st.expander("⚙️ Advanced: override the automatic feature selection", expanded=False):
        override_features = st.multiselect(
            "Feature columns (leave empty to auto-select)",
            [c for c in df.columns if c != target], key="train_feature_override",
        )

    if stateful_apply_button("🚀 Run Training", "status_run_training"):
        try:
            with st.spinner(f"Training ({task_label})…"):
                result = fc.run_training(
                    session, version, target, task_type,
                    feature_columns=override_features or None,
                )
            st.session_state["train_last_result"] = result
            st.session_state["train_last_task"] = task_type
            set_apply_status("status_run_training", True, "Done", result.get("model_choice", ""))
        except Exception as e:
            set_apply_status("status_run_training", False, "Failed", str(e))
        st.rerun()

    result = st.session_state.get("train_last_result")
    last_task = st.session_state.get("train_last_task")
    if not result:
        pass
    else:
        st.divider()
        st.subheader(f"Result — {result.get('model_choice', '')}")
        render_log(result.get("log", []))
        st.markdown(f"**Features used:** {', '.join(result.get('features', result.get('feature_cols', []))) or '—'}")

        if last_task == "regression":
            _render_regression_result(result)
        elif last_task == "classification":
            _render_classification_result(result)
        elif last_task == "clustering":
            _render_clustering_result(result)

    if session.training_runs:
        st.divider()
        st.subheader(f"All Training Runs ({len(session.training_runs)})")
        runs_df = pd.DataFrame([
            {
                "version": r["version"],
                "task": r["task_type"],
                "target": r["target"] or "—",
                "model": r["model_choice"],
                "metric": r["metric_name"],
                "value": round(r["metric_value"], 4) if r["metric_value"] is not None else None,
                "when": time.strftime("%H:%M:%S", time.localtime(r["timestamp"])),
            }
            for r in session.training_runs
        ])
        st.dataframe(runs_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    st.markdown("<div class='dashboard-title'>Cross-Modal Fusion</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dashboard-subtitle'>"
        "Structured &nbsp;·&nbsp; Text &nbsp;·&nbsp; Images"
        "&nbsp;&nbsp;|&nbsp;&nbsp;"
        "Fuse &nbsp;·&nbsp; Preprocess &nbsp;·&nbsp; Explore &nbsp;·&nbsp; Train"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tabs = st.tabs(["📤 Upload & Fuse", "🔧 Preprocessing", "📊 EDA", "🤖 Train Model"])
    with tabs[0]:
        tab_upload_fuse()
    with tabs[1]:
        tab_preprocessing()
    with tabs[2]:
        tab_eda()
    with tabs[3]:
        tab_training()


if __name__ == "__main__":
    main()
