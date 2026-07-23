"""
Universal ML Dashboard — app12.py
===================================
Changes over app11:
  1. Preprocessing is now tracked PER SOURCE DATASET, not globally. Each uploaded/
     merged/saved file has its own independent preprocessing log, so you can see
     exactly which steps were applied to which file (Preprocessing tab + sidebar).
  2. PCA no longer runs silently on page load — it only commits a reduced dataset
     once you click "Save This PCA Result". PCA is also tracked per input dataset,
     so you can run it on more than one file/preprocessed dataset independently.
  3. Dataset selectors (EDA, Regression, Classification, Clustering) now list EVERY
     preprocessed and PCA-reduced dataset that exists — clearly labelled with which
     source file they came from — instead of only the single latest one. The most
     recently created/updated dataset is selected by default.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix, classification_report,
    silhouette_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
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
.stat-zero   { border-left: 4px solid #6366f1; }
.stat-warn   { border-left: 4px solid #f59e0b; }

[data-testid="stExpander"] {
    background: #ffffff; border: 1px solid #e8e4de;
    border-radius: 10px; margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

.stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div {
    background: #ffffff; border: 1px solid #d1cdc7; border-radius: 8px; color: #1a1a2e;
}
.stSelectbox label, .stMultiSelect label, .stTextInput label,
.stSlider label, .stRadio label, .stCheckbox label {
    color: #374151 !important; font-weight: 500 !important; font-size: 13px !important;
}

[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e8e4de; border-radius: 10px;
    padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #1a1a2e !important; font-weight: 700 !important; }

hr { border-color: #e8e4de; margin: 20px 0; }

.log-entry {
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: #374151; background: #f8f7f4;
    border: 1px solid #e8e4de; border-radius: 6px;
    padding: 5px 10px; margin: 3px 0;
}

.save-badge {
    display:inline-block; background:#10b981; color:#fff;
    border-radius:6px; padding:2px 8px; font-size:11px; font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
def init_session_state():
    defaults = {
        "uploaded_files": {},        # name -> DataFrame (user-uploaded)
        "merged_df": None,           # actively merged/selected df (legacy convenience pointer)
        "merged_dfs": {},            # name -> DataFrame (user-merged datasets)
        "saved_dfs": {},             # name -> DataFrame (result tables saved by user)
        "active_source_name": None,  # label of the dataset last marked "active" in Data Manager

        # ── Per-source preprocessing state ──
        # Each raw dataset (uploaded/merged/saved) keeps its OWN preprocessed
        # dataframe and its OWN step-by-step log, keyed by that dataset's label.
        "preprocessed_dfs": {},      # {source_label: DataFrame}
        "preprocessing_logs": {},    # {source_label: [step strings]}

        # ── Per-source PCA state ──
        # Keyed by whichever dataset (raw or preprocessed) PCA was run on, so
        # running PCA on multiple files keeps each result separately available.
        "reduced_dfs": {},           # {input_label: DataFrame}
        "reduced_info": {},          # {input_label: {"columns":..., "component_names":...}}

        # ── Recency tracking, so selectors can default to the newest dataset ──
        "dataset_versions": {},      # {label: version_int}
        "version_counter": 0,

        "primary_key_col": None,
        "label_enc_mappings": {},
        "rows_to_delete": set(),
        # Regression
        "trained_reg_model": None, "trained_reg_features": None,
        "trained_reg_target": None, "trained_reg_name": None,
        "reg_train_results": None,
        # Classification
        "trained_cls_model": None, "trained_cls_features": None,
        "trained_cls_target": None, "trained_cls_name": None,
        "trained_cls_label_encoder": None, "trained_cls_class_names": None,
        "cls_train_results": None,
        # Clustering
        "trained_clust_model": None, "trained_clust_features": None,
        "trained_clust_scaler": None, "trained_clust_name": None,
        "trained_clust_k": None, "clust_train_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ============================================================
# Utilities
# ============================================================

def all_available_datasets():
    """Return dict of all datasets available: uploaded + merged + saved."""
    dfs = {}
    for name, df in st.session_state.uploaded_files.items():
        dfs[f"📂 {name}"] = df
    for name, df in st.session_state.merged_dfs.items():
        dfs[f"🔗 {name}"] = df
    for name, df in st.session_state.saved_dfs.items():
        dfs[f"💾 {name}"] = df
    return dfs


def bump_version(label):
    """Mark `label` as the most recently created/updated dataset, so dataset
    selectors elsewhere can default to it."""
    st.session_state.version_counter = st.session_state.get("version_counter", 0) + 1
    st.session_state.dataset_versions[label] = st.session_state.version_counter


def purge_dependents(raw_label):
    """When a raw source dataset is deleted, remove every preprocessed / PCA
    dataset that was derived from it, so stale entries never linger in selectors."""
    st.session_state.preprocessed_dfs.pop(raw_label, None)
    st.session_state.preprocessing_logs.pop(raw_label, None)
    for store in (st.session_state.reduced_dfs, st.session_state.reduced_info):
        for k in [k for k in store if raw_label in k]:
            store.pop(k, None)
    for k in [k for k in st.session_state.dataset_versions if raw_label in k]:
        st.session_state.dataset_versions.pop(k, None)
    if st.session_state.active_source_name == raw_label:
        st.session_state.active_source_name = None


def get_pca_input_options():
    """Datasets eligible as PCA input: every raw source + every preprocessed dataset."""
    options = {}
    for label, df in all_available_datasets().items():
        options[label] = df
    for label, df in st.session_state.preprocessed_dfs.items():
        options[f"Preprocessed: {label}"] = df
    return options


def build_dataset_catalog():
    """Every dataset available anywhere in the app, uniquely labelled with its
    source lineage: raw sources, EVERY preprocessed dataset (one per source file
    that has been preprocessed), and EVERY PCA-reduced dataset (one per input it
    was run on)."""
    catalog = {}
    for label, df in all_available_datasets().items():
        catalog[label] = df
    for label, df in st.session_state.preprocessed_dfs.items():
        catalog[f"Preprocessed: {label}"] = df
    for label, df in st.session_state.reduced_dfs.items():
        catalog[f"PCA: {label}"] = df
    return catalog


def smart_default_label(labels):
    """Pick the most recently created/updated label among the given options."""
    if not labels:
        return None
    versions = st.session_state.dataset_versions
    return max(labels, key=lambda l: versions.get(l, 0))


def get_eda_df():
    """For EDA: let user pick from every available dataset, defaulting to the
    most recently created/updated one."""
    catalog = build_dataset_catalog()
    if not catalog:
        return None, "No Dataset Loaded"
    labels = list(catalog.keys())
    default_label = smart_default_label(labels)
    idx = labels.index(default_label) if default_label in labels else 0
    if len(labels) == 1:
        chosen = labels[0]
    else:
        chosen = st.selectbox("📂 Dataset for EDA", labels, index=idx, key="eda_df_selector")
    return catalog[chosen], chosen


def get_model_df(tab_key="default"):
    """
    Smart dataset picker for ML tabs.
    Rules:
      - Always shows every raw dataset (uploaded / merged / saved).
      - Shows a *separate* preprocessed entry for every source that has been
        preprocessed (not just the latest one).
      - Shows a *separate* PCA-reduced entry for every input PCA was run on.
      - Defaults the selection to whichever dataset was created/updated most recently.
    """
    catalog = build_dataset_catalog()
    if not catalog:
        return None, "No Dataset Loaded", []

    labels = list(catalog.keys())
    default_label = smart_default_label(labels)
    idx = labels.index(default_label) if default_label in labels else 0

    hints = []
    if not st.session_state.preprocessed_dfs:
        hints.append("run Preprocessing to unlock a _Preprocessed_ dataset")
    if not st.session_state.reduced_dfs:
        hints.append("run PCA to unlock a _Dimensionality-Reduced_ dataset")

    if len(labels) == 1:
        chosen = labels[0]
    else:
        chosen = st.selectbox(
            "📂 Dataset to use for modelling",
            labels, index=idx, key=f"model_df_selector_{tab_key}"
        )
        if hints:
            st.caption("💡 " + " · ".join(hints))

    return catalog[chosen], chosen, labels


def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_cols(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def df_info_card(df, label="Dataset"):
    rows, cols_ = df.shape
    n_num = len(numeric_cols(df))
    n_cat = len(categorical_cols(df))
    missing = int(df.isnull().sum().sum())
    st.markdown(f"""
    <div class='active-banner'>
        🔬 <b>{label}</b> &nbsp;·&nbsp;
        {rows:,} rows &nbsp;×&nbsp; {cols_} cols &nbsp;·&nbsp;
        {n_num} numeric &nbsp;·&nbsp; {n_cat} categorical &nbsp;·&nbsp;
        {missing:,} missing values
    </div>
    """, unsafe_allow_html=True)


def download_csv_button(df, filename="dataset.csv", label="⬇ Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def save_result_table_widget(df, default_name, widget_key):
    """Offer user option to save a result dataframe to session saved_dfs."""
    with st.expander("💾 Save this result to session?", expanded=False):
        save_name = st.text_input(
            "Name for saved dataset", value=default_name, key=f"save_name_{widget_key}"
        )
        if st.button("✅ Save to Session", key=f"save_btn_{widget_key}"):
            if save_name.strip():
                key = save_name.strip()
                st.session_state.saved_dfs[key] = df.copy()
                bump_version(f"💾 {key}")
                st.success(f"✅ Saved as **{key}** — visible in sidebar under 'Saved Files'.")
            else:
                st.warning("Please enter a name.")


def interactive_result_table(df, key_prefix, title="Results", save_default_name=None):
    """Searchable, filterable, sortable dataframe with save option."""
    st.markdown(f"**{title}** — {len(df):,} rows × {df.shape[1]} cols")
    num_cols_df = df.select_dtypes(include=[np.number]).columns.tolist()

    with st.expander("🔍 Filter / Search / Sort", expanded=False):
        s_col1, s_col2 = st.columns([2, 2])
        with s_col1:
            search_text = st.text_input("Search value", key=f"{key_prefix}_search", placeholder="e.g. 1, yes…")
        with s_col2:
            search_cols = st.multiselect("Search in columns (empty = all)", df.columns.tolist(), key=f"{key_prefix}_scols")

        if num_cols_df:
            f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
            with f1:
                filter_col = st.selectbox("Filter column", ["None"] + num_cols_df, key=f"{key_prefix}_fcol")
            with f2:
                filter_op = st.selectbox("Op", [">=", "<=", "==", ">", "<", "!="], key=f"{key_prefix}_fop")
            with f3:
                filter_val = st.number_input("Value", value=0.0, key=f"{key_prefix}_fval")
            with f4:
                st.markdown("<br>", unsafe_allow_html=True)
                apply_f = st.button("Apply", key=f"{key_prefix}_fapply")
            if apply_f:
                st.session_state[f"{key_prefix}_filter_active"] = True
                st.session_state[f"{key_prefix}_filter_col"] = filter_col
                st.session_state[f"{key_prefix}_filter_op"] = filter_op
                st.session_state[f"{key_prefix}_filter_val"] = filter_val

        s1, s2 = st.columns(2)
        with s1:
            sort_col = st.selectbox("Sort by", ["None"] + df.columns.tolist(), key=f"{key_prefix}_sortcol")
        with s2:
            sort_asc = st.selectbox("Order", ["Ascending", "Descending"], key=f"{key_prefix}_sortdir") == "Ascending"

        if st.button("🔄 Reset", key=f"{key_prefix}_reset"):
            st.session_state.pop(f"{key_prefix}_filter_active", None)
            st.rerun()

    filtered = df.copy()

    if search_text:
        scope = search_cols if search_cols else df.columns.tolist()
        mask = pd.Series([False] * len(filtered), index=filtered.index)
        for col in scope:
            mask |= filtered[col].astype(str).str.contains(search_text, case=False, na=False)
        filtered = filtered[mask]

    if st.session_state.get(f"{key_prefix}_filter_active") and num_cols_df:
        fc = st.session_state.get(f"{key_prefix}_filter_col", "None")
        fo = st.session_state.get(f"{key_prefix}_filter_op", ">=")
        fv = st.session_state.get(f"{key_prefix}_filter_val", 0.0)
        if fc != "None" and fc in filtered.columns:
            ops = {
                ">=": filtered[fc] >= fv, "<=": filtered[fc] <= fv,
                "==": filtered[fc] == fv, ">":  filtered[fc] > fv,
                "<":  filtered[fc] < fv,  "!=": filtered[fc] != fv,
            }
            filtered = filtered[ops[fo]]

    if sort_col != "None" and sort_col in filtered.columns:
        filtered = filtered.sort_values(by=sort_col, ascending=sort_asc)

    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

    # Save option
    sn = save_default_name or f"{key_prefix}_result"
    save_result_table_widget(filtered, sn, key_prefix)

    return filtered


def flatten_json_record(record, parent_key="", sep="__"):
    items = {}
    if isinstance(record, dict):
        for k, v in record.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.update(flatten_json_record(v, new_key, sep=sep))
            else:
                items[new_key] = v
    elif isinstance(record, list):
        for i, v in enumerate(record):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list)):
                items.update(flatten_json_record(v, new_key, sep=sep))
            else:
                items[new_key] = v
    else:
        items[parent_key] = record
    return items


def load_file(uploaded_file):
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(uploaded_file), None
    elif ext in ("xlsx", "xls"):
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        if len(sheet_names) == 1:
            return xl.parse(sheet_names[0]), f"Sheet: **{sheet_names[0]}**"
        else:
            return xl, sheet_names
    elif ext == "json":
        import json
        raw = json.load(uploaded_file)
        if isinstance(raw, dict):
            list_keys = [k for k, v in raw.items() if isinstance(v, list)]
            records = raw[list_keys[0]] if list_keys else [raw]
            wrap_key = list_keys[0] if list_keys else None
        elif isinstance(raw, list):
            records = raw
            wrap_key = None
        else:
            raise ValueError("Unsupported JSON structure.")
        is_nested = any(
            isinstance(v, (dict, list))
            for rec in records if isinstance(rec, dict)
            for v in rec.values()
        )
        if is_nested:
            df = pd.DataFrame([flatten_json_record(r) for r in records])
            return df, f"Nested JSON flattened → {df.shape[1]} cols"
        else:
            df = pd.json_normalize(records)
            return df, f"Flat JSON → {df.shape[1]} cols"
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def pt():
    return "plotly_white"


def show_before_after(label, before_val, after_val, unit="items"):
    reduced = before_val - after_val
    is_zero = after_val == 0
    card_class = "stat-zero" if is_zero else "stat-after"
    zero_badge = "&nbsp;·&nbsp;<b style='color:#6366f1'>✓ None remaining!</b>" if is_zero else ""
    st.markdown(f"""
    <div class='stat-card {card_class}'>
        <b>{label}</b><br>
        <span style='color:#f59e0b'>Before: <b>{before_val:,} {unit}</b></span>
        &nbsp;→&nbsp;
        <span style='color:#10b981'>After: <b>{after_val:,} {unit}</b></span>
        &nbsp;·&nbsp;
        <span style='color:#374151'>Removed: <b>{reduced:,}</b></span>
        {zero_badge}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Sidebar — three tabs: Uploaded | Merged | Saved
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔬 ML Dashboard")
        st.markdown("---")

        sb_t1, sb_t2, sb_t3 = st.tabs(["📂 Uploaded", "🔗 Merged", "💾 Saved"])

        with sb_t1:
            files = st.session_state.uploaded_files
            if not files:
                st.info("No files uploaded yet.")
            else:
                for name, df in files.items():
                    st.markdown(f"**{name}**")
                    st.caption(f"{df.shape[0]:,} × {df.shape[1]}")
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ {name}", csv, file_name=name, mime="text/csv",
                                       key=f"sb_dl_up_{name}")

        with sb_t2:
            merged = st.session_state.merged_dfs
            if not merged:
                st.info("No merged datasets yet.")
            else:
                for name, df in merged.items():
                    st.markdown(f"**{name}**")
                    st.caption(f"{df.shape[0]:,} × {df.shape[1]}")
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ {name}", csv, file_name=f"{name}.csv", mime="text/csv",
                                       key=f"sb_dl_mg_{name}")
                    if st.button(f"🗑 Remove", key=f"sb_rm_mg_{name}"):
                        purge_dependents(f"🔗 {name}")
                        del st.session_state.merged_dfs[name]
                        st.rerun()

        with sb_t3:
            saved = st.session_state.saved_dfs
            if not saved:
                st.info("No result tables saved yet.")
            else:
                for name, df in saved.items():
                    st.markdown(f"**{name}**")
                    st.caption(f"{df.shape[0]:,} × {df.shape[1]}")
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ {name}", csv, file_name=f"{name}.csv", mime="text/csv",
                                       key=f"sb_dl_sv_{name}")
                    if st.button(f"🗑 Remove", key=f"sb_rm_sv_{name}"):
                        purge_dependents(f"💾 {name}")
                        del st.session_state.saved_dfs[name]
                        st.rerun()

        st.markdown("---")
        st.markdown("""
**Navigation**
1. 📁 Data Manager
2. 🔧 Preprocessing
3. 📊 EDA & Visualization
4. 🧠 Dimensionality Reduction
5. 📈 Regression
6. 🎯 Classification
7. 🔗 Clustering
        """)

        active_logs = {k: v for k, v in st.session_state.preprocessing_logs.items() if v}
        if active_logs:
            st.markdown("---")
            st.markdown("**Preprocessing Activity** — by file")
            st.caption("Which techniques were applied to which dataset.")
            for src, steps in active_logs.items():
                with st.expander(f"{src} — {len(steps)} step(s)", expanded=False):
                    for i, step in enumerate(steps, 1):
                        st.markdown(f"<div class='log-entry'>✓ Step {i}: {step}</div>", unsafe_allow_html=True)

        if st.session_state.reduced_dfs:
            st.markdown("---")
            st.markdown("**PCA Results** — by input dataset")
            for src in st.session_state.reduced_dfs:
                st.markdown(f"<div class='log-entry'>✓ PCA: {src}</div>", unsafe_allow_html=True)


# ============================================================
# TAB 0 — Data Manager
# ============================================================
def tab_data_manager():
    st.header("📁 Data Manager")
    st.markdown(
        "Upload **CSV**, **Excel**, or **JSON** files. Merge any uploaded/saved datasets. "
        "Set an active dataset for preprocessing & ML."
    )

    # ── 1. Upload ──
    st.subheader("1. Upload Datasets")
    uploaded = st.file_uploader(
        "Upload file(s)", type=["csv", "xlsx", "xls", "json"],
        accept_multiple_files=True
    )

    if uploaded is not None:
        current_names = {f.name for f in uploaded}
        removed = set(st.session_state.uploaded_files.keys()) - current_names
        for r in removed:
            del st.session_state.uploaded_files[r]
            purge_dependents(f"📂 {r}")
            if not st.session_state.uploaded_files:
                st.session_state.merged_df = None
                st.session_state.label_enc_mappings = {}
                st.session_state.primary_key_col = None
            st.info(f"Removed **{r}** from session.")

        for f in uploaded:
            if f.name not in st.session_state.uploaded_files:
                try:
                    result = load_file(f)
                    if isinstance(result[0], pd.ExcelFile):
                        xl, sheet_names = result
                        chosen = st.selectbox(
                            f"📊 **{f.name}** — choose sheet:",
                            sheet_names, key=f"sheet_{f.name}"
                        )
                        if st.button(f"Load sheet: {chosen}", key=f"load_sheet_{f.name}"):
                            df = xl.parse(chosen)
                            st.session_state.uploaded_files[f.name] = df
                            bump_version(f"📂 {f.name}")
                            st.success(f"Loaded **{f.name}** ({chosen}) — {df.shape[0]:,} × {df.shape[1]}")
                    else:
                        df, info_msg = result
                        st.session_state.uploaded_files[f.name] = df
                        bump_version(f"📂 {f.name}")
                        msg = f"Loaded **{f.name}** — {df.shape[0]:,} × {df.shape[1]}"
                        st.success(f"{msg}  ·  {info_msg}" if info_msg else msg)
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")

    if not st.session_state.uploaded_files:
        st.info("Upload at least one file to begin.")
        return

    # ── 2. Preview ──
    st.subheader("2. Preview")
    file_names = list(st.session_state.uploaded_files.keys())
    prev_choice = st.selectbox("Select file to preview", file_names, key="dm_preview")
    prev_df = st.session_state.uploaded_files[prev_choice]
    df_info_card(prev_df, prev_choice)
    st.dataframe(prev_df.head(50), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Column Types**")
        st.dataframe(pd.DataFrame({
            "Column": prev_df.columns,
            "Type": prev_df.dtypes.astype(str).values,
            "Non-Null": prev_df.notnull().sum().values,
            "Null %": (prev_df.isnull().mean() * 100).round(2).values
        }), use_container_width=True)
    with c2:
        st.markdown("**Summary Statistics**")
        st.dataframe(prev_df.describe(include="all").T, use_container_width=True)

    st.divider()

    # ── 3. Merge Any Datasets ──
    st.subheader("3. Merge Datasets")
    st.markdown(
        "You can merge **any** uploaded or saved datasets that share a common key column. "
        "Merged datasets are saved to the session and appear in the sidebar."
    )

    all_ds = all_available_datasets()
    all_ds_names = list(all_ds.keys())

    if len(all_ds_names) < 2:
        st.info("Need at least 2 datasets (uploaded or saved) to merge.")
    else:
        files_to_merge = st.multiselect(
            "Select datasets to merge (in order)", all_ds_names,
            default=all_ds_names[:2], key="dm_files_to_merge"
        )
        if len(files_to_merge) >= 2:
            join_type = st.selectbox("Join Type", ["inner", "outer", "left", "right"], key="dm_join_type")
            merge_keys_list = []
            for i in range(len(files_to_merge) - 1):
                left_df = all_ds[files_to_merge[i]]
                right_df = all_ds[files_to_merge[i + 1]]
                common = list(set(left_df.columns) & set(right_df.columns))
                if not common:
                    st.warning(f"No common columns between **{files_to_merge[i]}** and **{files_to_merge[i+1]}**.")
                    merge_keys_list.append([])
                else:
                    keys = st.multiselect(
                        f"Join key(s): `{files_to_merge[i]}` ➜ `{files_to_merge[i+1]}`",
                        common, default=[common[0]], key=f"dm_merge_key_{i}"
                    )
                    merge_keys_list.append(keys)

            merge_result_name = st.text_input(
                "Name for merged dataset", value="merged_dataset", key="dm_merge_name"
            )

            if st.button("🔗 Merge Datasets", key="btn_dm_merge"):
                with st.spinner("Merging…"):
                    try:
                        result = all_ds[files_to_merge[0]].copy()
                        for i in range(1, len(files_to_merge)):
                            right = all_ds[files_to_merge[i]]
                            keys = merge_keys_list[i - 1]
                            if not keys:
                                st.error(f"No keys for step {i}.")
                                continue
                            result = pd.merge(result, right, on=keys, how=join_type)
                        name = merge_result_name.strip() or "merged_dataset"
                        st.session_state.merged_dfs[name] = result
                        st.session_state.merged_df = result
                        st.session_state.active_source_name = f"🔗 {name}"
                        bump_version(f"🔗 {name}")
                        st.success(f"✅ Merged → **{name}** — {result.shape[0]:,} × {result.shape[1]}")
                        st.dataframe(result.head(20), use_container_width=True)
                        download_csv_button(result, f"{name}.csv")
                    except Exception as e:
                        st.error(f"Merge failed: {e}")

    st.divider()

    # ── 4. Set Active Dataset ──
    st.subheader("4. Set Active Dataset")
    st.caption(
        "Optional — marks a dataset as the default pick in other tabs' selectors. "
        "You can always choose any dataset directly in the Preprocessing, PCA, EDA, "
        "and Modelling tabs regardless of what's set here."
    )
    all_ds_now = all_available_datasets()
    if all_ds_now:
        single_choice = st.selectbox("Choose dataset", list(all_ds_now.keys()), key="dm_active_single")
        if st.button("✅ Set as Active Dataset", key="btn_set_active"):
            st.session_state.merged_df = all_ds_now[single_choice].copy()
            st.session_state.active_source_name = single_choice
            bump_version(single_choice)
            st.success(f"**{single_choice}** is now the active dataset.")
    else:
        if file_names:
            single_choice = st.selectbox("Choose dataset", file_names, key="dm_active_single2")
            if st.button("✅ Set as Active Dataset", key="btn_set_active2"):
                st.session_state.merged_df = st.session_state.uploaded_files[single_choice].copy()
                label_key = f"📂 {single_choice}"
                st.session_state.active_source_name = label_key
                bump_version(label_key)
                st.success(f"**{single_choice}** is now the active dataset.")


def commit_preprocess(source_label, work_df, entries):
    """Save the working dataframe as the current preprocessed state for THIS
    source dataset, and append the step(s) to THIS source's own log."""
    if isinstance(entries, str):
        entries = [entries]
    st.session_state.preprocessed_dfs[source_label] = work_df
    st.session_state.preprocessing_logs.setdefault(source_label, [])
    st.session_state.preprocessing_logs[source_label].extend(entries)
    bump_version(f"Preprocessed: {source_label}")


def tab_preprocessing():
    st.header("🔧 Data Preprocessing")

    raw_sources = all_available_datasets()
    if not raw_sources:
        st.warning("Please upload a dataset in the Data Manager first.")
        return

    labels = list(raw_sources.keys())
    default_label = st.session_state.get("pp_last_source")
    if default_label not in labels:
        default_label = (
            st.session_state.active_source_name
            if st.session_state.active_source_name in labels
            else labels[0]
        )
    idx = labels.index(default_label) if default_label in labels else 0
    source_label = st.selectbox(
        "📁 Dataset to preprocess", labels, index=idx, key="pp_source_select"
    )
    st.session_state.pp_last_source = source_label
    st.caption(
        "Each dataset here keeps its **own, separate** preprocessing history — "
        "switch the dropdown above to work on a different file."
    )

    raw_df = raw_sources[source_label]
    has_existing = source_label in st.session_state.preprocessed_dfs
    existing_steps = st.session_state.preprocessing_logs.get(source_label, [])

    top_c1, top_c2 = st.columns([4, 1])
    with top_c1:
        if has_existing:
            st.info(f"✏️ Continuing preprocessing on **{source_label}** — {len(existing_steps)} step(s) already applied.")
        else:
            st.caption(f"No preprocessing applied to **{source_label}** yet.")
    with top_c2:
        if has_existing and st.button("↺ Start Fresh", key="pp_restart"):
            st.session_state.preprocessed_dfs.pop(source_label, None)
            st.session_state.preprocessing_logs.pop(source_label, None)
            st.rerun()

    active_df = st.session_state.preprocessed_dfs.get(source_label, raw_df)
    display_label = f"Preprocessed: {source_label}" if has_existing else source_label
    df_info_card(active_df, display_label)

    # Show selected dataset as table at top
    with st.expander("📋 View Current Dataset", expanded=False):
        st.dataframe(active_df.head(100), use_container_width=True)

    work_df = active_df.copy()
    st.markdown("---")

    with st.expander("0. 🔑 Primary Key & Row Search / Delete"):
        all_cols_pk = work_df.columns.tolist()
        if st.session_state.primary_key_col not in all_cols_pk:
            st.session_state.primary_key_col = all_cols_pk[0] if all_cols_pk else None
        pk_col = st.selectbox("Primary Key Column", all_cols_pk,
            index=all_cols_pk.index(st.session_state.primary_key_col) if st.session_state.primary_key_col in all_cols_pk else 0,
            key="pk_select")
        if pk_col != st.session_state.primary_key_col:
            st.session_state.primary_key_col = pk_col
        st.markdown(f"**Current primary key:** `{st.session_state.primary_key_col}`")
        st.divider()
        st.markdown("**Search Rows by Value**")
        search_val = st.text_input("Search value", key="row_search_val")
        search_scope_cols = st.multiselect("Columns (empty=all)", work_df.columns.tolist(), key="row_search_cols")
        if search_val:
            scope = search_scope_cols if search_scope_cols else work_df.columns.tolist()
            mask = pd.Series([False]*len(work_df), index=work_df.index)
            for col in scope:
                mask |= work_df[col].astype(str).str.contains(search_val, case=False, na=False)
            found_df = work_df[mask]
            st.dataframe(found_df, use_container_width=True)
            save_result_table_widget(found_df, "search_results", "pp_search")
        st.divider()
        delete_input = st.text_input("Row indices to delete (comma-separated)", key="manual_delete_input")
        if st.button("Delete Selected Rows", key="btn_manual_delete"):
            try:
                idx = [int(x.strip()) for x in delete_input.split(",") if x.strip()]
                valid = [i for i in idx if i in work_df.index]
                if valid:
                    deleted_pk_vals = work_df.loc[valid, st.session_state.primary_key_col].tolist() if st.session_state.primary_key_col else valid
                    work_df = work_df.drop(index=valid)
                    commit_preprocess(
                        source_label, work_df,
                        f"Deleted {len(valid)} row(s) — {st.session_state.primary_key_col}: {deleted_pk_vals}"
                    )
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    # ── 1. Missing Values ──
    with st.expander("1. 🧹 Handle Missing Values", expanded=True):
        missing_summary = work_df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0].index.tolist()
        if not missing_cols:
            st.success("No missing values.")
        else:
            pk = st.session_state.primary_key_col

            # ── Summary table ──
            miss_data = {
                "Column": missing_cols,
                "Missing Count": missing_summary[missing_cols].values,
                "Missing %": (work_df[missing_cols].isnull().mean()*100).round(2).values,
            }
            st.dataframe(pd.DataFrame(miss_data), use_container_width=True)

            # ── Per-column PK viewer ──
            if pk and pk in work_df.columns:
                st.markdown("**🔍 Rows with Missing Values (by column)**")
                st.caption(
                    "Expand a column below to see the rows (identified by Primary Key) "
                    "that have missing values. You can then select specific rows to handle "
                    "differently from the rest."
                )
                # Store per-column row selections in session state
                if "mv_row_selections" not in st.session_state:
                    st.session_state["mv_row_selections"] = {}

                for c in missing_cols:
                    mask_miss = work_df[c].isnull()
                    miss_rows = work_df[mask_miss][[pk]].copy()
                    miss_rows.index.name = "row_idx"
                    miss_rows = miss_rows.reset_index()
                    with st.expander(
                        f"📋 **{c}** — {int(mask_miss.sum())} missing row(s)", expanded=False
                    ):
                        # Show table of PK values
                        st.dataframe(
                            miss_rows.rename(columns={"row_idx": "Row Index", pk: pk}),
                            use_container_width=True,
                            height=min(300, 40 + 35 * len(miss_rows)),
                        )
                        pk_options = miss_rows[pk].astype(str).tolist()
                        sel_key = f"mv_sel_{c}"
                        selected_pks = st.multiselect(
                            f"Select {pk} values to handle separately (leave empty = apply global strategy to all)",
                            options=pk_options,
                            default=st.session_state["mv_row_selections"].get(c, []),
                            key=sel_key,
                        )
                        st.session_state["mv_row_selections"][c] = selected_pks

                        if selected_pks:
                            # Per-row strategy
                            row_strategy = st.selectbox(
                                f"Strategy for selected {pk} values in `{c}`",
                                ["Drop these rows", "Fill with Mean", "Fill with Median",
                                 "Fill with Mode", "Fill with Custom Value"],
                                key=f"mv_row_strat_{c}",
                            )
                            row_custom = (
                                st.text_input(
                                    f"Custom value for selected rows in `{c}`",
                                    key=f"mv_row_custom_{c}",
                                )
                                if row_strategy == "Fill with Custom Value"
                                else None
                            )
                            if st.button(
                                f"Apply to selected rows in `{c}`",
                                key=f"btn_mv_row_{c}",
                            ):
                                try:
                                    # Identify row indices matching selected PK values
                                    sel_pk_typed = []
                                    for pv in selected_pks:
                                        try:
                                            sel_pk_typed.append(type(work_df[pk].iloc[0])(pv))
                                        except Exception:
                                            sel_pk_typed.append(pv)
                                    row_mask = work_df[pk].isin(sel_pk_typed) & work_df[c].isnull()
                                    before_r = int(row_mask.sum())
                                    if row_strategy == "Drop these rows":
                                        work_df = work_df[~row_mask]
                                    elif row_strategy == "Fill with Mean":
                                        if pd.api.types.is_numeric_dtype(work_df[c]):
                                            work_df.loc[row_mask, c] = work_df[c].mean()
                                    elif row_strategy == "Fill with Median":
                                        if pd.api.types.is_numeric_dtype(work_df[c]):
                                            work_df.loc[row_mask, c] = work_df[c].median()
                                    elif row_strategy == "Fill with Mode":
                                        m_ = work_df[c].mode()
                                        if len(m_) > 0:
                                            work_df.loc[row_mask, c] = m_[0]
                                    elif row_strategy == "Fill with Custom Value":
                                        work_df.loc[row_mask, c] = row_custom
                                    commit_preprocess(
                                        source_label, work_df,
                                        f"Missing values (row-level): '{row_strategy}' on "
                                        f"{before_r} row(s) in col '{c}' for {pk} in {selected_pks}"
                                    )
                                    st.session_state["mv_row_selections"].pop(c, None)
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

            st.markdown("---")
            st.markdown("**🌐 Global Strategy — apply to column(s) as a whole**")
            target_cols_miss = st.multiselect(
                "Columns to fix", missing_cols, default=missing_cols, key="target_cols_miss"
            )
            strategy = st.selectbox(
                "Strategy",
                ["Drop rows with nulls", "Fill with Mean", "Fill with Median",
                 "Fill with Mode", "Fill with Custom Value", "Drop columns"],
                key="missing_strategy",
            )
            custom_val = (
                st.text_input("Custom value", key="missing_custom_val")
                if strategy == "Fill with Custom Value"
                else None
            )
            if st.button("Apply Global Strategy", key="apply_missing"):
                cols_to_fix = target_cols_miss if target_cols_miss else missing_cols
                before_miss = int(work_df[cols_to_fix].isnull().sum().sum())
                try:
                    if strategy == "Drop rows with nulls":
                        work_df = work_df.dropna(subset=cols_to_fix)
                    elif strategy == "Fill with Mean":
                        for c in cols_to_fix:
                            if pd.api.types.is_numeric_dtype(work_df[c]):
                                work_df[c] = work_df[c].fillna(work_df[c].mean())
                    elif strategy == "Fill with Median":
                        for c in cols_to_fix:
                            if pd.api.types.is_numeric_dtype(work_df[c]):
                                work_df[c] = work_df[c].fillna(work_df[c].median())
                    elif strategy == "Fill with Mode":
                        for c in cols_to_fix:
                            m = work_df[c].mode()
                            if len(m) > 0:
                                work_df[c] = work_df[c].fillna(m[0])
                    elif strategy == "Fill with Custom Value":
                        for c in cols_to_fix:
                            work_df[c] = work_df[c].fillna(custom_val)
                    elif strategy == "Drop columns":
                        work_df = work_df.drop(columns=cols_to_fix)
                    commit_preprocess(
                        source_label, work_df,
                        f"Missing values: {strategy} on {len(cols_to_fix)} col(s)"
                    )
                    after_miss = int(work_df.isnull().sum().sum()) if strategy != "Drop columns" else 0
                    show_before_after("Missing values", before_miss, after_miss, "missing cells")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    # ── 2. Fix Data Types ──
    with st.expander("2. 🔄 Fix Data Types"):
        type_cols = st.multiselect("Columns", work_df.columns.tolist(), key="type_cols")
        if type_cols:
            # Show current types for selected columns
            cur_types_df = pd.DataFrame({
                "Column": type_cols,
                "Current Type": [str(work_df[c].dtype) for c in type_cols],
            })
            st.dataframe(cur_types_df, use_container_width=True)
        target_type = st.selectbox("Convert to", ["numeric (float)","numeric (int)","string (object)","datetime","category","boolean"], key="target_type")
        if type_cols and st.button("Apply", key="apply_types"):
            try:
                old_types = {c: str(work_df[c].dtype) for c in type_cols}
                for c in type_cols:
                    if target_type == "numeric (float)": work_df[c] = pd.to_numeric(work_df[c], errors="coerce").astype(float)
                    elif target_type == "numeric (int)": work_df[c] = pd.to_numeric(work_df[c], errors="coerce").astype("Int64")
                    elif target_type == "string (object)": work_df[c] = work_df[c].astype(str)
                    elif target_type == "datetime": work_df[c] = pd.to_datetime(work_df[c], errors="coerce")
                    elif target_type == "category": work_df[c] = work_df[c].astype("category")
                    elif target_type == "boolean": work_df[c] = work_df[c].astype(bool)
                commit_preprocess(
                    source_label, work_df,
                    [f"Type conversion: '{c}' {old_types[c]} → {target_type}" for c in type_cols]
                )
                # Show updated types
                new_types_df = pd.DataFrame({
                    "Column": type_cols,
                    "Old Type": [old_types[c] for c in type_cols],
                    "New Type": [str(work_df[c].dtype) for c in type_cols],
                })
                st.dataframe(new_types_df, use_container_width=True)
                st.rerun()
            except Exception as e: st.error(str(e))

    # ── 3. Remove Duplicates ──
    with st.expander("3. 🔁 Remove Duplicates"):
        n_full_dups = int(work_df.duplicated().sum())
        st.markdown(f"Full-row duplicates: **{n_full_dups}**")
        dup_subset = st.multiselect("Subset columns", work_df.columns.tolist(), key="dup_subset")
        dup_keep = st.selectbox("Keep", ["first","last","none (drop all)"], key="dup_keep")
        keep_val = False if dup_keep == "none (drop all)" else dup_keep
        if st.button("Remove Duplicates", key="btn_dup"):
            before_rows = len(work_df)
            work_df = work_df.drop_duplicates(subset=dup_subset if dup_subset else None, keep=keep_val)
            commit_preprocess(source_label, work_df, f"Removed {before_rows-len(work_df)} duplicate row(s)")
            show_before_after("Rows", before_rows, len(work_df), "rows")
            st.rerun()

    # ── 4. Handle Outliers ──
    with st.expander("4. 📦 Handle Outliers"):
        num_cols_o = numeric_cols(work_df)
        if not num_cols_o: st.info("No numeric columns.")
        else:
            out_cols = st.multiselect("Columns", num_cols_o, default=num_cols_o[:5], key="out_cols")
            out_method = st.selectbox("Method", ["Z-Score (|z| > 3)","IQR (1.5x IQR)","Percentile (1%-99%)"], key="out_method")
            out_action = st.selectbox("Action", ["Remove outlier rows","Cap to boundary"], key="out_action")

            # Show rows that contain outliers in selected columns
            if out_cols:
                try:
                    mask_preview = pd.Series([False]*len(work_df), index=work_df.index)
                    for c in out_cols:
                        cd = work_df[c].dropna()
                        if out_method == "Z-Score (|z| > 3)":
                            m_, s_ = cd.mean(), cd.std()
                            mask_preview |= (work_df[c]-m_).abs() > 3*s_
                        elif out_method == "IQR (1.5x IQR)":
                            q1, q3 = cd.quantile(0.25), cd.quantile(0.75)
                            iqr = q3-q1
                            mask_preview |= (work_df[c]<q1-1.5*iqr)|(work_df[c]>q3+1.5*iqr)
                        else:
                            l_, u_ = cd.quantile(0.01), cd.quantile(0.99)
                            mask_preview |= (work_df[c]<l_)|(work_df[c]>u_)
                    outlier_df = work_df[mask_preview]
                    st.markdown(f"**{len(outlier_df):,} row(s)** contain outliers in selected column(s) using **{out_method}**:")
                    st.dataframe(outlier_df[out_cols + (
                        [st.session_state.primary_key_col] if st.session_state.primary_key_col and
                        st.session_state.primary_key_col in work_df.columns and
                        st.session_state.primary_key_col not in out_cols else []
                    )].head(200), use_container_width=True)
                except Exception:
                    pass

            if out_cols and st.button("Apply", key="apply_out"):
                before_rows = len(work_df)
                try:
                    mask_all = pd.Series([False]*len(work_df), index=work_df.index)
                    for c in out_cols:
                        cd = work_df[c].dropna()
                        if out_method == "Z-Score (|z| > 3)":
                            m_,s_ = cd.mean(), cd.std()
                            mask = (work_df[c]-m_).abs() > 3*s_
                            if out_action == "Cap to boundary": work_df[c] = work_df[c].clip(m_-3*s_, m_+3*s_)
                            else: mask_all |= mask
                        elif out_method == "IQR (1.5x IQR)":
                            q1,q3 = cd.quantile(0.25), cd.quantile(0.75)
                            iqr = q3-q1
                            lo,hi = q1-1.5*iqr, q3+1.5*iqr
                            mask = (work_df[c]<lo)|(work_df[c]>hi)
                            if out_action == "Cap to boundary": work_df[c] = work_df[c].clip(lo,hi)
                            else: mask_all |= mask
                        else:
                            l,u = cd.quantile(0.01), cd.quantile(0.99)
                            mask = (work_df[c]<l)|(work_df[c]>u)
                            if out_action == "Cap to boundary": work_df[c] = work_df[c].clip(l,u)
                            else: mask_all |= mask
                    if out_action == "Remove outlier rows": work_df = work_df[~mask_all]
                    commit_preprocess(source_label, work_df, f"Outliers: {out_action} on {len(out_cols)} col(s) via {out_method}")
                    show_before_after("Rows", before_rows, len(work_df), "rows")
                    st.rerun()
                except Exception as e: st.error(str(e))

    # ── 5. Encode Categorical ──
    with st.expander("5. 🔤 Encode Categorical Variables"):
        cat_cols = categorical_cols(work_df)
        if not cat_cols: st.info("No categorical columns.")
        else:
            enc_cols = st.multiselect("Columns", cat_cols, default=cat_cols[:3], key="enc_cols")
            enc_method = st.selectbox("Method", ["Label Encoding","One-Hot Encoding"], key="enc_method")
            label_enc_new_names = {}
            if enc_method == "Label Encoding" and enc_cols:
                for c in enc_cols:
                    nn = st.text_input(f"Name for `{c}`", value=f"{c}_enc", key=f"le_newname_{c}")
                    label_enc_new_names[c] = nn.strip() if nn.strip() else f"{c}_enc"
            if enc_cols and st.button("Apply", key="apply_enc"):
                try:
                    if enc_method == "Label Encoding":
                        le = LabelEncoder()
                        for c in enc_cols:
                            out_col = label_enc_new_names.get(c, f"{c}_enc")
                            le.fit(work_df[c].astype(str))
                            work_df[out_col] = le.transform(work_df[c].astype(str))
                            st.session_state.label_enc_mappings[c] = {"output_col": out_col, "mapping": {cls:idx for idx,cls in enumerate(le.classes_)}}
                    else:
                        dummies = pd.get_dummies(work_df[enc_cols], prefix=enc_cols, drop_first=True)
                        work_df = pd.concat([work_df, dummies], axis=1)
                    commit_preprocess(source_label, work_df, f"Encoding: {enc_method} on {len(enc_cols)} col(s)")
                    st.rerun()
                except Exception as e: st.error(str(e))

    # ── 6. Scale / Normalize ──
    with st.expander("6. ⚖️ Scale / Normalize"):
        num_cols_s = numeric_cols(work_df)
        if not num_cols_s: st.info("No numeric columns.")
        else:
            scale_cols = st.multiselect("Columns", num_cols_s, default=num_cols_s[:5], key="scale_cols")
            scale_method = st.selectbox("Method", ["Standard Scaler (Z-score)","Min-Max Scaler (0-1)","Robust Scaler (median/IQR)"], key="scale_method")
            if scale_cols and st.button("Apply", key="apply_scale"):
                try:
                    sm = {"Standard Scaler (Z-score)": StandardScaler(), "Min-Max Scaler (0-1)": MinMaxScaler(), "Robust Scaler (median/IQR)": RobustScaler()}
                    work_df[scale_cols] = sm[scale_method].fit_transform(work_df[scale_cols])
                    commit_preprocess(source_label, work_df, f"Scaled: {scale_method} on {len(scale_cols)} col(s)")
                    st.rerun()
                except Exception as e: st.error(str(e))

    # ── 7. Drop Columns ──
    with st.expander("7. ✂️ Drop Columns"):
        drop_cols = st.multiselect("Columns to drop", work_df.columns.tolist(), key="drop_cols")
        if drop_cols and st.button("Drop", key="btn_drop_cols"):
            before_c = work_df.shape[1]
            work_df = work_df.drop(columns=drop_cols)
            commit_preprocess(source_label, work_df, f"Dropped columns: {drop_cols}")
            show_before_after("Columns", before_c, work_df.shape[1], "columns")
            st.rerun()

    # ── 8. Rename Columns ──
    with st.expander("8. ✏️ Rename Columns"):
        rename_map = {}
        for col in work_df.columns.tolist()[:20]:
            nn = st.text_input(f"`{col}`", value="", key=f"rename_{col}")
            if nn.strip(): rename_map[col] = nn.strip()
        if rename_map and st.button("Apply Renames", key="apply_rename"):
            work_df = work_df.rename(columns=rename_map)
            commit_preprocess(
                source_label, work_df,
                [f'Renamed column "{old_c}" → "{new_c}"' for old_c, new_c in rename_map.items()]
            )
            st.rerun()

    st.divider()

    # ── Preprocessing Steps Log — for the CURRENTLY selected source only ──
    current_steps = st.session_state.preprocessing_logs.get(source_label, [])
    if current_steps:
        st.subheader(f"📋 Preprocessing Steps Taken — {source_label}")
        for i, step in enumerate(current_steps, 1):
            st.markdown(
                f"<div class='log-entry'>✓ <b>Step {i}:</b> {step}</div>",
                unsafe_allow_html=True
            )
        st.markdown("")

    st.subheader("Current Dataset State")
    current_df = st.session_state.preprocessed_dfs.get(source_label, work_df)
    df_info_card(current_df, display_label)
    st.dataframe(current_df, use_container_width=True)
    download_csv_button(current_df, "preprocessed.csv")
    if st.button(f"🔄 Reset Preprocessing for {source_label}", key="reset_preprocessing"):
        st.session_state.preprocessed_dfs.pop(source_label, None)
        st.session_state.preprocessing_logs.pop(source_label, None)
        st.success("Reset done.")
        st.rerun()

    # ── All-files overview: which techniques were applied to which files ──
    all_logs = {k: v for k, v in st.session_state.preprocessing_logs.items() if v}
    if len(all_logs) > 1 or (all_logs and source_label not in all_logs):
        st.divider()
        with st.expander("📚 Preprocessing History — All Files", expanded=False):
            st.caption("A complete record of exactly which steps were applied to which dataset.")
            for src, steps in all_logs.items():
                st.markdown(f"**{src}** — {len(steps)} step(s)")
                for i, step in enumerate(steps, 1):
                    st.markdown(
                        f"<div class='log-entry'>✓ Step {i}: {step}</div>",
                        unsafe_allow_html=True
                    )



# ============================================================
# TAB 2 — EDA & Visualization
# ============================================================
def tab_eda():
    st.header("📊 EDA & Visualization")

    df, label = get_eda_df()
    if df is None:
        st.warning("No dataset available. Upload a file first.")
        return

    df_info_card(df, label)
    with st.expander("📋 View Selected Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
    num_c = numeric_cols(df)
    cat_c = categorical_cols(df)

    eda_t1, eda_t_search, eda_t2, eda_t3, eda_t4, eda_t5 = st.tabs([
        "📋 Overview", "🔍 Search & Filter", "📈 Distributions",
        "🔗 Correlations", "🎨 Custom Chart", "📐 Pair Plot"
    ])

    with eda_t1:
        st.subheader("Dataset Overview")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Numeric Cols", len(num_c))
        c4.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        overview = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values,
            "Unique": df.nunique().values, "Missing": df.isnull().sum().values,
            "Missing %": (df.isnull().mean()*100).round(1).values})
        st.dataframe(overview, use_container_width=True)
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with eda_t_search:
        st.subheader("Search, Filter & Sort")
        filtered_df = df.copy()
        st.markdown("**Text Search**")
        s1,s2 = st.columns([2,2])
        with s1: search_text = st.text_input("Search value", key="eda_search_text", placeholder="e.g. India, 42…")
        with s2: search_cols = st.multiselect("Columns (empty=all)", df.columns.tolist(), key="eda_search_cols")
        if search_text:
            scope = search_cols if search_cols else df.columns.tolist()
            mask = pd.Series([False]*len(filtered_df), index=filtered_df.index)
            for col in scope:
                mask |= filtered_df[col].astype(str).str.contains(search_text, case=False, na=False)
            filtered_df = filtered_df[mask]
        st.divider()
        st.markdown("**Filter Numeric Columns**")
        if num_c:
            filter_col = st.selectbox("Column", ["None"]+num_c, key="eda_filter_col")
            if filter_col != "None":
                f1,f2,f3 = st.columns(3)
                with f1: filter_op = st.selectbox("Condition", [">=","<=","==",">","<","!="], key="eda_filter_op")
                with f2: filter_val = st.number_input("Value", value=float(df[filter_col].min()), key="eda_filter_val")
                with f3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Apply Filter", key="eda_apply_filter"):
                        st.session_state["eda_filter_active"] = True
                if st.session_state.get("eda_filter_active"):
                    ops = {">=": filtered_df[filter_col]>=filter_val, "<=": filtered_df[filter_col]<=filter_val,
                           "==": filtered_df[filter_col]==filter_val, ">": filtered_df[filter_col]>filter_val,
                           "<": filtered_df[filter_col]<filter_val, "!=": filtered_df[filter_col]!=filter_val}
                    filtered_df = filtered_df[ops[filter_op]]
        st.divider()
        st.markdown("**Sort**")
        sc1,sc2 = st.columns(2)
        with sc1: sort_col = st.selectbox("Sort by", ["None"]+df.columns.tolist(), key="eda_sort_col")
        with sc2:
            if sort_col != "None":
                sort_ord = st.selectbox("Order", ["Ascending","Descending"], key="eda_sort_order")
                ascending = sort_ord == "Ascending"
        if sort_col != "None": filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending)
        st.divider()
        st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} rows**")
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)
        download_csv_button(filtered_df, "filtered_data.csv", "⬇ Download Filtered Data")
        save_result_table_widget(filtered_df, "eda_filter_result", "eda_search_save")
        if st.button("🔄 Reset Filters", key="eda_reset_filters"):
            st.session_state["eda_filter_active"] = False
            st.rerun()

    with eda_t2:
        st.subheader("Feature Distributions")
        if not num_c: st.info("No numeric columns.")
        else:
            dist_col = st.selectbox("Column", num_c, key="dist_col")
            d1,d2 = st.columns(2)
            with d1:
                fig = px.histogram(df, x=dist_col, marginal="box", template=pt(),
                    title=f"Distribution — {dist_col}", color_discrete_sequence=["#1a1a2e"])
                st.plotly_chart(fig, use_container_width=True)
            with d2:
                fig2 = px.box(df, y=dist_col, template=pt(), title=f"Box Plot — {dist_col}",
                    color_discrete_sequence=["#374151"])
                st.plotly_chart(fig2, use_container_width=True)
            if cat_c:
                grp = st.selectbox("Group by", ["None"]+cat_c, key="grp_violin")
                if grp != "None":
                    fig3 = px.violin(df, y=dist_col, x=grp, box=True, template=pt(), title=f"{dist_col} by {grp}")
                    st.plotly_chart(fig3, use_container_width=True)

    with eda_t3:
        st.subheader("Correlation Analysis")
        if len(num_c) < 2: st.info("Need at least 2 numeric columns.")
        else:
            corr_method = st.selectbox("Method", ["pearson","spearman","kendall"], key="corr_method")
            selected_num = st.multiselect("Columns", num_c, default=num_c[:min(10,len(num_c))], key="corr_selected_num")
            if len(selected_num) >= 2:
                corr = df[selected_num].corr(method=corr_method)
                fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title=f"Correlation Matrix ({corr_method})", template=pt())
                fig_c.update_layout(height=500)
                st.plotly_chart(fig_c, use_container_width=True)
                corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
                corr_pairs.columns = ["Feature A","Feature B","Correlation"]
                corr_pairs = corr_pairs.iloc[corr_pairs["Correlation"].abs().argsort()[::-1].values]
                st.dataframe(corr_pairs.head(20), use_container_width=True)

    # ── Custom Chart — multi-column support ──────────────────────────────
    with eda_t4:
        st.subheader("Custom Chart Builder")
        st.markdown("**Multi-column comparison supported** — e.g. multi-line, grouped bar, multi-scatter.")

        chart_type = st.selectbox("Chart Type", [
            "Scatter", "Multi-Scatter (multiple Y)",
            "Line", "Multi-Line (multiple Y)",
            "Bar", "Multi-Bar (multiple Y)",
            "Histogram", "Box", "Violin",
            "Pie", "Heatmap (Pivot)"
        ], key="custom_chart_type")

        MULTI_CHARTS = ["Multi-Scatter (multiple Y)", "Multi-Line (multiple Y)", "Multi-Bar (multiple Y)"]

        if chart_type in MULTI_CHARTS:
            x_col = st.selectbox("X Axis", ["None"] + df.columns.tolist(), key="cc_x_multi")
            y_cols_multi = st.multiselect("Y Columns (multiple allowed)", num_c, key="cc_y_multi",
                default=num_c[:min(3, len(num_c))])
            color_col = st.selectbox("Color By (optional)", ["None"] + df.columns.tolist(), key="cc_color_multi")
            color = None if color_col == "None" else color_col
            x = None if x_col == "None" else x_col

            if not y_cols_multi:
                st.info("Select at least one Y column.")
            elif x is None:
                st.info("Select an X column.")
            else:
                try:
                    if chart_type == "Multi-Line (multiple Y)":
                        # Melt for multi-line
                        plot_df = df[[x] + y_cols_multi].copy().dropna()
                        melted = plot_df.melt(id_vars=x, value_vars=y_cols_multi, var_name="Series", value_name="Value")
                        fig = px.line(melted, x=x, y="Value", color="Series", template=pt(),
                            title=f"Multi-Line: {', '.join(y_cols_multi)} vs {x}")
                    elif chart_type == "Multi-Bar (multiple Y)":
                        plot_df = df[[x] + y_cols_multi].copy().dropna()
                        melted = plot_df.melt(id_vars=x, value_vars=y_cols_multi, var_name="Series", value_name="Value")
                        fig = px.bar(melted, x=x, y="Value", color="Series", barmode="group", template=pt(),
                            title=f"Multi-Bar: {', '.join(y_cols_multi)} vs {x}")
                    else:  # Multi-Scatter
                        plot_df = df[[x] + y_cols_multi].copy().dropna()
                        melted = plot_df.melt(id_vars=x, value_vars=y_cols_multi, var_name="Series", value_name="Value")
                        fig = px.scatter(melted, x=x, y="Value", color="Series", template=pt(),
                            title=f"Multi-Scatter: {', '.join(y_cols_multi)} vs {x}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
        else:
            c1,c2,c3 = st.columns(3)
            with c1: x_col = st.selectbox("X Axis", ["None"]+df.columns.tolist(), key="cc_x")
            with c2: y_col = st.selectbox("Y Axis", ["None"]+num_c, key="cc_y")
            with c3: color_col = st.selectbox("Color By", ["None"]+df.columns.tolist(), key="cc_color")
            x = None if x_col == "None" else x_col
            y = None if y_col == "None" else y_col
            color = None if color_col == "None" else color_col
            try:
                if chart_type == "Scatter" and x and y:
                    size_col = st.selectbox("Size By", ["None"]+num_c, key="cc_size")
                    size = None if size_col == "None" else size_col
                    fig = px.scatter(df, x=x, y=y, color=color, size=size, template=pt())
                elif chart_type == "Line" and x and y:
                    fig = px.line(df, x=x, y=y, color=color, template=pt())
                elif chart_type == "Bar" and x and y:
                    fig = px.bar(df, x=x, y=y, color=color, barmode="group", template=pt())
                elif chart_type == "Histogram" and x:
                    fig = px.histogram(df, x=x, color=color, template=pt())
                elif chart_type == "Box" and y:
                    fig = px.box(df, x=x, y=y, color=color, template=pt())
                elif chart_type == "Violin" and y:
                    fig = px.violin(df, x=x, y=y, color=color, box=True, template=pt())
                elif chart_type == "Pie" and x and y:
                    fig = px.pie(df, names=x, values=y, template=pt())
                elif chart_type == "Heatmap (Pivot)" and x and y:
                    pv = st.selectbox("Value column", num_c, key="cc_pivot")
                    pivot = df.pivot_table(index=y, columns=x, values=pv, aggfunc="mean")
                    fig = px.imshow(pivot, template=pt())
                else:
                    st.info("Select X and Y axes.")
                    fig = None
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

    with eda_t5:
        st.subheader("Pair Plot")
        pair_cols = st.multiselect("Columns (3–6 recommended)", num_c,
            default=num_c[:min(4,len(num_c))], key="pair_cols")
        pair_color = st.selectbox("Color By", ["None"]+cat_c, key="pair_color")
        if len(pair_cols) >= 2:
            if st.button("Generate Pair Plot", key="btn_pair_plot"):
                with st.spinner("Generating…"):
                    fig_p = px.scatter_matrix(df, dimensions=pair_cols,
                        color=None if pair_color == "None" else pair_color,
                        template=pt(), title="Pair Plot")
                    fig_p.update_traces(diagonal_visible=False, showupperhalf=False)
                    fig_p.update_layout(height=700)
                    st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Select at least 2 columns.")


# ============================================================
# TAB 3 — PCA / Dimensionality Reduction
# ============================================================
def tab_pca():
    st.header("🧠 Dimensionality Reduction — PCA")

    input_options = get_pca_input_options()
    if not input_options:
        st.warning("No dataset available. Upload a file first.")
        return

    labels = list(input_options.keys())
    default_label = smart_default_label(labels)
    idx = labels.index(default_label) if default_label in labels else 0
    source_label = st.selectbox(
        "📁 Dataset to reduce", labels, index=idx, key="pca_source_select"
    )
    st.caption(
        "You can run PCA on more than one dataset — each is saved separately and "
        "will all show up (clearly labelled) in the EDA and Modelling tabs."
    )
    df = input_options[source_label]

    if source_label in st.session_state.reduced_dfs:
        st.info(f"💾 A saved PCA result already exists for **{source_label}**. Adjust settings below and click **Save** to update it.")

    df_info_card(df, source_label)
    with st.expander("📋 View Selected Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
    num_c = numeric_cols(df)
    c1,c2 = st.columns(2)
    with c1: pca_cols = st.multiselect("Feature columns for PCA", num_c, default=num_c, key="pca_cols")
    with c2: pca_missing = st.selectbox("Handle missing values", ["Fill with Mean","Drop Rows","Fill with 0"], key="pca_missing")
    if len(pca_cols) < 2:
        st.info("Select at least 2 columns.")
        return
    pca_df = df[pca_cols].copy()
    if pca_missing == "Drop Rows": pca_df = pca_df.dropna()
    elif pca_missing == "Fill with Mean": pca_df = pca_df.fillna(pca_df.mean())
    else: pca_df = pca_df.fillna(0)
    if pca_df.empty:
        st.error("No data after handling missing values.")
        return
    scaler = StandardScaler()
    scaled = scaler.fit_transform(pca_df)
    max_comp = min(len(pca_cols), pca_df.shape[0])
    st.subheader("Explained Variance Ratio (all components)")
    pca_full = PCA(n_components=max_comp)
    pca_full.fit(scaled)
    full_exp_var = pca_full.explained_variance_ratio_ * 100
    full_cumulative = np.cumsum(full_exp_var)
    ev_df = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(max_comp)],
        "Explained Variance (%)": full_exp_var.round(2), "Cumulative (%)": full_cumulative.round(2)})
    st.dataframe(ev_df, use_container_width=True)
    fig_full_var = go.Figure()
    fig_full_var.add_bar(x=ev_df["Component"], y=ev_df["Explained Variance (%)"], name="Individual", marker_color="#1a1a2e")
    fig_full_var.add_scatter(x=ev_df["Component"], y=ev_df["Cumulative (%)"],
        name="Cumulative", line=dict(color="#10b981", width=2), mode="lines+markers")
    fig_full_var.update_layout(template=pt(), title="Full Explained Variance Ratio", yaxis_title="Variance (%)", xaxis_title="Component")
    st.plotly_chart(fig_full_var, use_container_width=True)
    st.caption("💡 Pick # components where cumulative variance exceeds 90%.")
    n_comp = st.slider("Number of components", 1, min(max_comp, 10), min(2, max_comp), key="pca_n_comp")
    pca = PCA(n_components=n_comp)
    pca_result = pca.fit_transform(scaled)
    exp_var = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(exp_var)
    fig_var = go.Figure()
    fig_var.add_bar(x=[f"PC{i+1}" for i in range(n_comp)], y=exp_var, name="Individual", marker_color="#1a1a2e")
    fig_var.add_scatter(x=[f"PC{i+1}" for i in range(n_comp)], y=cumulative,
        name="Cumulative", line=dict(color="#10b981", width=2), mode="lines+markers")
    fig_var.update_layout(template=pt(), title="Selected Components Variance", yaxis_title="Variance (%)")
    st.plotly_chart(fig_var, use_container_width=True)
    st.subheader("Name Your Components")
    custom_names = []
    if n_comp <= 8:
        name_cols = st.columns(n_comp)
        for i, col in enumerate(name_cols):
            with col:
                name = st.text_input(f"PC{i+1}", value=f"PC{i+1}", key=f"pc_name_{i}")
                custom_names.append(name)
    else:
        for i in range(n_comp):
            name = st.text_input(f"PC{i+1}", value=f"PC{i+1}", key=f"pc_name_{i}")
            custom_names.append(name)
    df_pca = pd.DataFrame(pca_result, columns=custom_names)
    non_num = [c for c in df.columns if c not in num_c]
    if non_num:
        label_col = st.selectbox("Label column (hover)", non_num, key="pca_label")
        df_pca[label_col] = df[label_col].values[:len(df_pca)]
        hover_col = label_col
    else:
        df_pca["Index"] = range(len(df_pca))
        hover_col = "Index"

    st.markdown("---")
    st.caption("Nothing above has been saved yet — this is a live preview. Click below to make it available elsewhere.")
    if st.button("💾 Save This PCA Result", key="btn_save_pca"):
        st.session_state.reduced_dfs[source_label] = df_pca
        st.session_state.reduced_info[source_label] = {
            "input_label": source_label,
            "columns": pca_cols,
            "n_components": n_comp,
            "component_names": custom_names,
        }
        bump_version(f"PCA: {source_label}")
        st.success(f"✅ Saved as **PCA: {source_label}** — available in EDA, Regression, Classification, and Clustering tabs.")

    if n_comp >= 2:
        pc_x = st.selectbox("X Component", custom_names, index=0, key="pca_x")
        pc_y = st.selectbox("Y Component", custom_names, index=1, key="pca_y")
        fig_pca = px.scatter(df_pca, x=pc_x, y=pc_y, hover_name=hover_col,
            color=pc_x, color_continuous_scale="Blues", template=pt(), title=f"{pc_x} vs {pc_y}")
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        fig_1d = px.histogram(df_pca, x=custom_names[0], marginal="box",
            template=pt(), color_discrete_sequence=["#1a1a2e"])
        st.plotly_chart(fig_1d, use_container_width=True)
    with st.expander("📊 Feature Loadings"):
        loadings = pd.DataFrame(pca.components_.T, index=pca_cols, columns=custom_names)
        if n_comp > 1:
            fig_load = px.imshow(loadings, text_auto=".2f", color_continuous_scale="RdBu_r", template=pt())
            st.plotly_chart(fig_load, use_container_width=True)
        st.dataframe(loadings, use_container_width=True)
    # Allow EDA on the reduced dataset, once saved
    if source_label in st.session_state.reduced_dfs:
        st.info(f"💡 View this in the **EDA & Visualization** tab — select **PCA: {source_label}** from the dataset dropdown.")
    else:
        st.caption("⚠️ Not saved yet — click **Save This PCA Result** above to unlock it in other tabs.")


# ============================================================
# TAB 4 — Regression
# ============================================================
def tab_regression():
    st.header("📈 Regression")
    df, label, _ = get_model_df(tab_key="reg")
    if df is None:
        st.warning("No dataset available.")
        return
    df_info_card(df, label)
    with st.expander("📋 View Selected Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
    num_c = numeric_cols(df)
    if len(num_c) < 2:
        st.error("Need at least 2 numeric columns.")
        return
    c1,c2 = st.columns(2)
    with c1: target = st.selectbox("Target (Y)", num_c, key="reg_target")
    with c2: features = st.multiselect("Feature columns (X)", [c for c in num_c if c!=target],
        default=[c for c in num_c if c!=target][:5], key="reg_features")
    if not features:
        st.info("Select at least one feature.")
        return
    model_df = df[features+[target]].dropna()
    X = model_df[features].values
    y = model_df[target].values
    st.markdown(f"**{len(model_df):,} samples** after dropping NaN rows.")
    c3,c4,c5 = st.columns(3)
    with c3: model_choice = st.selectbox("Model", ["Linear Regression","Ridge Regression","Lasso Regression","Polynomial Regression"], key="reg_model")
    with c4: test_size = st.slider("Test Size (%)", 10, 40, 20, key="reg_test_size") / 100
    with c5: eval_mode = st.selectbox("Evaluation", ["Train/Test Split","Cross-Validation","Both"], key="reg_eval")
    alpha = 1.0
    poly_degree = 2
    if model_choice == "Ridge Regression": alpha = st.slider("Alpha", 0.01, 100.0, 1.0, key="ridge_alpha")
    elif model_choice == "Lasso Regression": alpha = st.slider("Alpha", 0.0001, 10.0, 0.1, key="lasso_alpha")
    elif model_choice == "Polynomial Regression": poly_degree = st.slider("Degree", 2, 5, 2, key="poly_deg")
    cv_folds_reg = None
    if eval_mode in ["Cross-Validation","Both"]: cv_folds_reg = st.slider("CV Folds", 3, 10, 5, key="reg_cv_folds")
    if st.button("🚀 Train Model", key="train_reg"):
        with st.spinner("Training…"):
            if model_choice == "Linear Regression": model = LinearRegression()
            elif model_choice == "Ridge Regression": model = Ridge(alpha=alpha)
            elif model_choice == "Lasso Regression": model = Lasso(alpha=alpha)
            else: model = Pipeline([("poly", PolynomialFeatures(degree=poly_degree,include_bias=False)),("lin",LinearRegression())])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores_data = cv_rmse_data = cv_folds_used = None
            if eval_mode in ["Cross-Validation","Both"]:
                cv_folds_used = cv_folds_reg or 5
                kf = KFold(n_splits=cv_folds_used, shuffle=True, random_state=42)
                cv_scores_data = cross_val_score(model, X, y, cv=kf, scoring="r2")
                cv_rmse_data = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
            coef_data = None
            if model_choice in ["Linear Regression","Ridge Regression","Lasso Regression"]: coef_data = model.coef_
            st.session_state.trained_reg_model = model
            st.session_state.trained_reg_features = features
            st.session_state.trained_reg_target = target
            st.session_state.trained_reg_name = model_choice
            st.session_state.reg_train_results = {
                "model_choice":model_choice,"eval_mode":eval_mode,
                "X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"y_pred":y_pred,
                "r2":r2,"rmse":rmse,"mae":mae,"features":features,"target":target,
                "cv_scores":cv_scores_data,"cv_rmse":cv_rmse_data,"cv_folds":cv_folds_used,"coef_data":coef_data,
            }
    res = st.session_state.get("reg_train_results")
    if res:
        st.markdown("---")
        st.subheader("📊 Training Results")
        if res["eval_mode"] in ["Train/Test Split","Both"]:
            m1,m2,m3 = st.columns(3)
            m1.metric("R²", f"{res['r2']:.4f}")
            m2.metric("RMSE", f"{res['rmse']:.4f}")
            m3.metric("MAE", f"{res['mae']:.4f}")
            with st.expander("📂 View Test Dataset"):
                test_show = pd.DataFrame(res["X_test"], columns=res["features"])
                test_show[res["target"]] = res["y_test"]
                test_show["Predicted"] = res["y_pred"]
                test_show["Residual"] = res["y_test"] - res["y_pred"]
                filtered_test = interactive_result_table(test_show, "reg_test", "Test Dataset", "reg_test_data")
                download_csv_button(filtered_test, "reg_test_results.csv")
            fig_p = go.Figure()
            fig_p.add_scatter(x=res["y_test"], y=res["y_pred"], mode="markers",
                marker=dict(color="#1a1a2e", opacity=0.5), name="Predictions")
            fig_p.add_scatter(x=[res["y_test"].min(), res["y_test"].max()],
                y=[res["y_test"].min(), res["y_test"].max()],
                mode="lines", line=dict(color="#ef4444", dash="dash"), name="Perfect")
            fig_p.update_layout(template=pt(), title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig_p, use_container_width=True)
            residuals = res["y_test"] - res["y_pred"]
            fig_r = px.scatter(x=res["y_pred"], y=residuals, template=pt(), title="Residual Plot",
                labels={"x":"Predicted","y":"Residuals"}, color_discrete_sequence=["#374151"])
            fig_r.add_hline(y=0, line_dash="dash", line_color="#ef4444")
            st.plotly_chart(fig_r, use_container_width=True)
        if res["eval_mode"] in ["Cross-Validation","Both"] and res["cv_scores"] is not None:
            cv_scores = res["cv_scores"]
            cv_rmse = res["cv_rmse"]
            cv_folds = res["cv_folds"]
            c1,c2,c3 = st.columns(3)
            c1.metric("Mean R²", f"{cv_scores.mean():.4f}")
            c2.metric("Std R²", f"±{cv_scores.std():.4f}")
            c3.metric("Mean RMSE", f"{cv_rmse.mean():.4f}")
            fig_cv = px.bar(x=[f"Fold {i+1}" for i in range(cv_folds)], y=cv_scores,
                template=pt(), title="R² per Fold", color_discrete_sequence=["#1a1a2e"])
            st.plotly_chart(fig_cv, use_container_width=True)
        if res["coef_data"] is not None:
            coefs = res["coef_data"]
            fig_coef = px.bar(x=res["features"], y=coefs, template=pt(), title="Feature Coefficients",
                color_discrete_sequence=["#374151"])
            st.plotly_chart(fig_coef, use_container_width=True)
        st.success("✅ Model saved! Predict on future data below.")
    st.divider()
    st.subheader("🔮 Predict on Future Data")
    if st.session_state.trained_reg_model is None:
        st.info("Train a model above first.")
        return
    saved_features = st.session_state.trained_reg_features
    saved_target = st.session_state.trained_reg_target
    saved_model_name = st.session_state.trained_reg_name
    st.markdown(f"<div class='active-banner'>📦 <b>{saved_model_name}</b> · Target: <b>{saved_target}</b> · Features: <b>{', '.join(saved_features)}</b></div>", unsafe_allow_html=True)

    pred_tab_manual, pred_tab_file = st.tabs(["✏️ Enter Values Manually", "📂 Upload Dataset"])

    with pred_tab_manual:
        st.markdown("Enter values for each feature, then click **Predict**.")
        manual_vals = {}
        cols_per_row = 3
        feat_chunks = [saved_features[i:i+cols_per_row] for i in range(0, len(saved_features), cols_per_row)]
        for chunk in feat_chunks:
            row_cols = st.columns(len(chunk))
            for col_widget, feat in zip(row_cols, chunk):
                with col_widget:
                    manual_vals[feat] = st.number_input(feat, value=0.0, key=f"manual_feat_{feat}")
        if st.button("🎯 Predict", key="btn_manual_predict_reg"):
            X_manual = np.array([[manual_vals[f] for f in saved_features]])
            pred_val = st.session_state.trained_reg_model.predict(X_manual)[0]
            st.success(f"**Predicted {saved_target}:** `{pred_val:.4f}`")
            st.markdown(
                f"<div class='stat-card stat-after'>🎯 <b>Prediction result</b><br>"
                f"Target: <b>{saved_target}</b> &nbsp;·&nbsp; Value: <b>{pred_val:.4f}</b></div>",
                unsafe_allow_html=True
            )

    with pred_tab_file:
        future_file = st.file_uploader("Upload future dataset", type=["csv","xlsx","xls","json"], key="reg_future_upload")
        if future_file:
            try:
                result = load_file(future_file)
                if isinstance(result[0], pd.ExcelFile): st.error("Multi-sheet Excel: export to CSV first.")
                else:
                    future_df, _ = result
                    missing_feats = [f for f in saved_features if f not in future_df.columns]
                    if missing_feats: st.error(f"Missing columns: **{', '.join(missing_feats)}**")
                    else:
                        X_future = future_df[saved_features].fillna(0).values
                        preds = st.session_state.trained_reg_model.predict(X_future)
                        future_df[f"Predicted_{saved_target}"] = preds
                        filtered_future = interactive_result_table(future_df, "reg_future", "Predictions", "reg_future_preds")
                        download_csv_button(filtered_future, "regression_predictions.csv")
                        if saved_target in future_df.columns:
                            y_true = future_df[saved_target].values
                            valid = ~pd.isnull(y_true)
                            if valid.sum() > 0:
                                f_r2 = r2_score(y_true[valid].astype(float), preds[valid])
                                f_rmse = np.sqrt(mean_squared_error(y_true[valid].astype(float), preds[valid]))
                                f_mae = mean_absolute_error(y_true[valid].astype(float), preds[valid])
                                fm1,fm2,fm3 = st.columns(3)
                                fm1.metric("R²", f"{f_r2:.4f}")
                                fm2.metric("RMSE", f"{f_rmse:.4f}")
                                fm3.metric("MAE", f"{f_mae:.4f}")
            except Exception as e: st.error(f"Error: {e}")



# ============================================================
# TAB 5 — Classification
# ============================================================
def tab_classification():
    st.header("🎯 Classification")
    df, label, _ = get_model_df(tab_key="cls")
    if df is None:
        st.warning("No dataset available.")
        return
    df_info_card(df, label)
    with st.expander("📋 View Selected Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
    num_c = numeric_cols(df)
    all_c = df.columns.tolist()
    c1,c2 = st.columns(2)
    with c1: target = st.selectbox("Target column", all_c, key="cls_target")
    with c2: features = st.multiselect("Feature columns (X)", [c for c in num_c if c!=target],
        default=[c for c in num_c if c!=target][:6], key="cls_features")
    if not features:
        st.info("Select at least one feature.")
        return
    model_df = df[features+[target]].dropna()
    X = model_df[features].values
    y_raw = model_df[target]
    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        class_names = le.classes_
    else:
        le = None
        y = y_raw.values.astype(int)
        class_names = np.unique(y).astype(str)
    st.markdown(f"**{len(model_df):,} samples** | **{len(class_names)} classes**: `{', '.join(class_names[:10])}`")
    c3,c4,c5 = st.columns(3)
    with c3: model_choice = st.selectbox("Model", ["Logistic Regression","K-Nearest Neighbors","SVM (RBF)","Decision Tree","Random Forest"], key="cls_model")
    with c4: test_size = st.slider("Test Size (%)", 10, 40, 20, key="cls_test") / 100
    with c5: eval_mode = st.selectbox("Evaluation", ["Train/Test Split","Cross-Validation","Both"], key="cls_eval")
    k_val=5; svm_c=1.0; dt_depth=5; rf_n=100; rf_d=5
    if model_choice == "K-Nearest Neighbors": k_val = st.slider("Neighbors (k)", 1, 25, 5, key="knn_k")
    elif model_choice == "SVM (RBF)": svm_c = st.slider("SVM C", 0.01, 100.0, 1.0, key="svm_c")
    elif model_choice == "Decision Tree": dt_depth = st.slider("Max Depth", 1, 20, 5, key="dt_depth")
    elif model_choice == "Random Forest":
        rf_n = st.slider("Trees", 10, 300, 100, key="rf_n")
        rf_d = st.slider("Max Depth", 1, 20, 5, key="rf_depth")
    cv_folds_cls = None
    if eval_mode in ["Cross-Validation","Both"]: cv_folds_cls = st.slider("CV Folds", 3, 10, 5, key="cls_cv_folds")
    if st.button("🚀 Train Classifier", key="train_cls"):
        with st.spinner("Training…"):
            if model_choice == "Logistic Regression": model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_choice == "K-Nearest Neighbors": model = KNeighborsClassifier(n_neighbors=k_val)
            elif model_choice == "SVM (RBF)": model = SVC(C=svm_c, kernel="rbf", random_state=42, probability=True)
            elif model_choice == "Decision Tree": model = DecisionTreeClassifier(max_depth=dt_depth, random_state=42)
            else: model = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d, random_state=42)
            min_class_count = pd.Series(y).value_counts().min()
            can_stratify = len(np.unique(y)) > 1 and min_class_count >= 2
            if not can_stratify and len(np.unique(y)) > 1:
                st.warning("⚠️ Stratified split disabled: some classes have only 1 sample.")
            strat = y if can_stratify else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            present_idx = np.unique(np.concatenate([y_test, y_pred]))
            present_class_names = [class_names[i] for i in present_idx]
            report = classification_report(y_test, y_pred, target_names=present_class_names, labels=present_idx, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            cv_scores_data = cv_folds_used = None
            if eval_mode in ["Cross-Validation","Both"]:
                cv_folds_used = cv_folds_cls or 5
                skf = StratifiedKFold(n_splits=cv_folds_used, shuffle=True, random_state=42)
                cv_scores_data = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
            feat_importances = model.feature_importances_ if model_choice in ["Decision Tree","Random Forest"] else None
            st.session_state.trained_cls_model = model
            st.session_state.trained_cls_features = features
            st.session_state.trained_cls_target = target
            st.session_state.trained_cls_name = model_choice
            st.session_state.trained_cls_label_encoder = le
            st.session_state.trained_cls_class_names = class_names
            st.session_state.cls_train_results = {
                "model_choice":model_choice,"eval_mode":eval_mode,
                "X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"y_pred":y_pred,
                "acc":acc,"report":report,"cm":cm,"class_names":class_names,
                "present_class_names":present_class_names,"features":features,"target":target,
                "cv_scores":cv_scores_data,"cv_folds":cv_folds_used,"feat_importances":feat_importances,
            }
    res = st.session_state.get("cls_train_results")
    if res:
        st.markdown("---")
        st.subheader("📊 Training Results")
        if res["eval_mode"] in ["Train/Test Split","Both"]:
            st.metric("Accuracy", f"{res['acc']:.4f}  ({res['acc']*100:.1f}%)")
            with st.expander("📂 View Test Dataset"):
                test_show = pd.DataFrame(res["X_test"], columns=res["features"])
                test_show[res["target"]] = res["y_test"]
                test_show["Predicted"] = res["y_pred"]
                test_show["Correct"] = (res["y_test"] == res["y_pred"])
                filtered_test = interactive_result_table(test_show, "cls_test", "Test Dataset", "cls_test_data")
                download_csv_button(filtered_test, "cls_test_results.csv")
            st.markdown("**Classification Report**")
            st.dataframe(pd.DataFrame(res["report"]).T.style.format("{:.3f}"), use_container_width=True)
            fig_cm = px.imshow(res["cm"], text_auto=True, template=pt(), labels=dict(x="Predicted",y="Actual"),
                x=res["class_names"], y=res["class_names"], color_continuous_scale="Blues", title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        if res["eval_mode"] in ["Cross-Validation","Both"] and res["cv_scores"] is not None:
            cv_scores = res["cv_scores"]
            cv_folds = res["cv_folds"]
            c1,c2 = st.columns(2)
            c1.metric("Mean Accuracy", f"{cv_scores.mean():.4f}")
            c2.metric("Std", f"±{cv_scores.std():.4f}")
            fig_cv = px.bar(x=[f"Fold {i+1}" for i in range(cv_folds)], y=cv_scores,
                template=pt(), title="Accuracy per Fold", color_discrete_sequence=["#1a1a2e"])
            st.plotly_chart(fig_cv, use_container_width=True)
        if res["feat_importances"] is not None:
            fig_imp = px.bar(x=res["features"], y=res["feat_importances"],
                template=pt(), title="Feature Importances", color_discrete_sequence=["#374151"])
            st.plotly_chart(fig_imp, use_container_width=True)
        st.success("✅ Model saved!")
    st.divider()
    st.subheader("🔮 Classify Future Data")
    if st.session_state.trained_cls_model is None:
        st.info("Train a classifier above first.")
        return
    saved_features = st.session_state.trained_cls_features
    saved_target = st.session_state.trained_cls_target
    saved_model_name = st.session_state.trained_cls_name
    saved_le = st.session_state.trained_cls_label_encoder
    saved_class_names = st.session_state.trained_cls_class_names
    st.markdown(f"<div class='active-banner'>📦 <b>{saved_model_name}</b> · Target: <b>{saved_target}</b></div>", unsafe_allow_html=True)
    future_file = st.file_uploader("Upload future dataset", type=["csv","xlsx","xls","json"], key="cls_future_upload")
    if future_file:
        try:
            result = load_file(future_file)
            if isinstance(result[0], pd.ExcelFile): st.error("Multi-sheet Excel: export to CSV first.")
            else:
                future_df, _ = result
                missing_feats = [f for f in saved_features if f not in future_df.columns]
                if missing_feats: st.error(f"Missing columns: **{', '.join(missing_feats)}**")
                else:
                    X_future = future_df[saved_features].fillna(0).values
                    preds_encoded = st.session_state.trained_cls_model.predict(X_future)
                    preds_labels = saved_le.inverse_transform(preds_encoded) if saved_le is not None else preds_encoded.astype(str)
                    future_df[f"Predicted_{saved_target}"] = preds_labels
                    filtered_future = interactive_result_table(future_df, "cls_future", "Classifications", "cls_future_preds")
                    download_csv_button(filtered_future, "classification_predictions.csv")
                    pred_counts = pd.Series(preds_labels).value_counts().reset_index()
                    pred_counts.columns = ["Class","Count"]
                    fig_dist = px.bar(pred_counts, x="Class", y="Count", template=pt(), color_discrete_sequence=["#1a1a2e"])
                    st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e: st.error(f"Error: {e}")


# ============================================================
# TAB 6 — Clustering
# ============================================================
def tab_clustering():
    st.header("🔗 Clustering")
    df, label, _ = get_model_df(tab_key="clust")
    if df is None:
        st.warning("No dataset available.")
        return
    df_info_card(df, label)
    with st.expander("📋 View Selected Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
    num_c = numeric_cols(df)
    if len(num_c) < 2:
        st.error("Need at least 2 numeric columns.")
        return
    feature_cols = st.multiselect(
        "Feature columns for clustering", num_c, default=num_c, key="clust_feature_cols"
    )
    if len(feature_cols) < 2:
        st.info("Select at least 2 feature columns.")
        return
    non_num_c = [c for c in df.columns if c not in num_c]
    label_col_clust = None
    if non_num_c:
        label_col_clust = st.selectbox("Label column for hover", ["None"]+non_num_c, key="clust_label_col")
        if label_col_clust == "None": label_col_clust = None
    missing_strat = st.selectbox("Handle missing values", ["Fill with Mean","Drop Rows","Fill with 0"], key="clust_missing")
    cluster_df = df[feature_cols].copy()
    if missing_strat == "Drop Rows": cluster_df = cluster_df.dropna()
    elif missing_strat == "Fill with Mean": cluster_df = cluster_df.fillna(cluster_df.mean())
    else: cluster_df = cluster_df.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)
    algorithm = st.selectbox("Algorithm", ["K-Means","DBSCAN"], key="clust_algo")
    st.divider()

    if algorithm == "K-Means":
        n_samples = X_scaled.shape[0]
        max_k = min(11, n_samples-1)
        if max_k < 2:
            st.error(f"Not enough samples ({n_samples}) after handling missing values.")
            return
        st.subheader("1. Find Optimal k")
        with st.spinner("Computing elbow and silhouette curves…"):
            k_range = range(2, max_k+1)
            inertias, silhouettes = [], []
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbs = km.fit_predict(X_scaled)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(X_scaled, lbs))
        c1,c2 = st.columns(2)
        with c1:
            fig_e = px.line(x=list(k_range), y=inertias, markers=True, template=pt(),
                title="Elbow Method", labels={"x":"k","y":"Inertia"}, color_discrete_sequence=["#1a1a2e"])
            st.plotly_chart(fig_e, use_container_width=True)
        with c2:
            fig_s = px.line(x=list(k_range), y=silhouettes, markers=True, template=pt(),
                title="Silhouette Score", labels={"x":"k","y":"Silhouette"}, color_discrete_sequence=["#374151"])
            st.plotly_chart(fig_s, use_container_width=True)
        st.subheader("2. Run K-Means")
        k_val = st.slider("Number of Clusters (k)", 2, max_k, min(3,max_k), key="kmeans_k")
        if st.button("🚀 Run K-Means", key="run_kmeans"):
            with st.spinner(f"Running K-Means k={k_val}…"):
                km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                sil = silhouette_score(X_scaled, labels)
            st.session_state.trained_clust_model = km
            st.session_state.trained_clust_features = feature_cols
            st.session_state.trained_clust_scaler = scaler
            st.session_state.trained_clust_name = "K-Means"
            st.session_state.trained_clust_k = k_val
            result_df = cluster_df.copy()
            result_df["Cluster"] = labels.astype(str)
            if label_col_clust: result_df[label_col_clust] = df[label_col_clust].values[:len(result_df)]
            full_result_df = df.copy().reset_index(drop=True).iloc[:len(result_df)]
            full_result_df["Cluster"] = labels.astype(str)
            st.session_state.clust_train_results = {
                "algo":"K-Means","k_val":k_val,"sil":sil,"labels":labels,
                "result_df":result_df,"full_result_df":full_result_df,
                "feature_cols":feature_cols,"label_col_clust":label_col_clust,
            }
        res = st.session_state.get("clust_train_results")
        if res and res.get("algo") == "K-Means":
            st.metric("Silhouette Score", f"{res['sil']:.4f}")
            result_df = res["result_df"]
            full_result_df = res["full_result_df"]
            viz_x = st.selectbox("X Axis", feature_cols, index=0, key="km_viz_x")
            viz_y = st.selectbox("Y Axis", feature_cols, index=1, key="km_viz_y")
            fig_k = px.scatter(result_df, x=viz_x, y=viz_y, color="Cluster", symbol="Cluster",
                template=pt(), hover_name=res["label_col_clust"],
                title=f"K-Means Clusters (k={res['k_val']})")
            st.plotly_chart(fig_k, use_container_width=True)
            st.subheader("Cluster Statistics")
            st.dataframe(result_df.groupby("Cluster")[feature_cols].mean().round(3), use_container_width=True)
            with st.expander("View & Filter Cluster Assignments"):
                filtered_clust = interactive_result_table(full_result_df, "kmeans_result", "K-Means Cluster Assignments", "kmeans_clusters")
                download_csv_button(filtered_clust, "kmeans_clusters.csv")
            st.success("✅ Model saved!")
    else:
        st.subheader("DBSCAN Parameters")
        c1,c2 = st.columns(2)
        with c1: eps = st.slider("Epsilon", 0.1, 5.0, 0.5, step=0.1, key="dbscan_eps")
        with c2: min_samples = st.slider("Min Samples", 2, 20, 5, key="dbscan_min")
        if st.button("🚀 Run DBSCAN", key="run_dbscan"):
            with st.spinner("Running DBSCAN…"):
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db.fit_predict(X_scaled)
            st.session_state.trained_clust_model = db
            st.session_state.trained_clust_features = feature_cols
            st.session_state.trained_clust_scaler = scaler
            st.session_state.trained_clust_name = "DBSCAN"
            st.session_state.trained_clust_k = None
            result_df = cluster_df.copy()
            result_df["Cluster"] = labels.astype(str)
            if label_col_clust: result_df[label_col_clust] = df[label_col_clust].values[:len(result_df)]
            full_result_df_db = df.copy().reset_index(drop=True).iloc[:len(result_df)]
            full_result_df_db["Cluster"] = labels.astype(str)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            sil_db = None
            if n_clusters > 1:
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > 0 and len(np.unique(labels[non_noise_mask])) > 1:
                    sil_db = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
            st.session_state.clust_train_results = {
                "algo":"DBSCAN","eps":eps,"min_samples":min_samples,
                "labels":labels,"n_clusters":n_clusters,"n_noise":n_noise,"sil":sil_db,
                "result_df":result_df,"full_result_df":full_result_df_db,
                "feature_cols":feature_cols,"label_col_clust":label_col_clust,
            }
        res = st.session_state.get("clust_train_results")
        if res and res.get("algo") == "DBSCAN":
            c1,c2 = st.columns(2)
            c1.metric("Clusters Found", res["n_clusters"])
            c2.metric("Noise Points", res["n_noise"])
            if res["sil"] is not None: st.metric("Silhouette (excl. noise)", f"{res['sil']:.4f}")
            result_df = res["result_df"]
            full_result_df_db = res["full_result_df"]
            viz_x = st.selectbox("X Axis", feature_cols, index=0, key="db_viz_x")
            viz_y = st.selectbox("Y Axis", feature_cols, index=1, key="db_viz_y")
            fig_db = px.scatter(result_df, x=viz_x, y=viz_y, color="Cluster",
                template=pt(), hover_name=res["label_col_clust"],
                title=f"DBSCAN (eps={res['eps']}, min_samples={res['min_samples']})")
            st.plotly_chart(fig_db, use_container_width=True)
            with st.expander("View & Filter Cluster Assignments"):
                filtered_db = interactive_result_table(full_result_df_db, "dbscan_result", "DBSCAN Cluster Assignments", "dbscan_clusters")
                download_csv_button(filtered_db, "dbscan_clusters.csv")
            st.success("✅ Model saved!")

    # Future data prediction
    st.divider()
    st.subheader("🔮 Assign Future Data to Clusters")
    if st.session_state.trained_clust_model is None:
        st.info("Run a clustering algorithm above first.")
        return
    saved_features = st.session_state.trained_clust_features
    saved_scaler = st.session_state.trained_clust_scaler
    saved_model = st.session_state.trained_clust_model
    saved_clust_name = st.session_state.trained_clust_name
    st.markdown(f"<div class='active-banner'>📦 <b>{saved_clust_name}</b> · Features: <b>{', '.join(saved_features)}</b></div>", unsafe_allow_html=True)
    future_file = st.file_uploader("Upload future dataset", type=["csv","xlsx","xls","json"], key="clust_future_upload")
    if future_file:
        try:
            result = load_file(future_file)
            if isinstance(result[0], pd.ExcelFile): st.error("Multi-sheet Excel: export to CSV first.")
            else:
                future_df, _ = result
                missing_feats = [f for f in saved_features if f not in future_df.columns]
                if missing_feats: st.error(f"Missing columns: **{', '.join(missing_feats)}**")
                else:
                    X_future_raw = future_df[saved_features].fillna(0).values
                    X_future_scaled = saved_scaler.transform(X_future_raw)
                    if saved_clust_name == "K-Means":
                        cluster_labels = saved_model.predict(X_future_scaled)
                    else:
                        st.info("ℹ️ DBSCAN: assigning via nearest training point.")
                        from sklearn.neighbors import NearestNeighbors
                        train_labels = saved_model.labels_
                        train_X = saved_scaler.transform(df[saved_features].fillna(0).values[:len(train_labels)])
                        nn = NearestNeighbors(n_neighbors=1)
                        nn.fit(train_X)
                        _, indices = nn.kneighbors(X_future_scaled)
                        cluster_labels = train_labels[indices.flatten()]
                    future_df["Assigned_Cluster"] = cluster_labels.astype(str)
                    filtered_future_clust = interactive_result_table(future_df, "clust_future", "Future Cluster Assignments", "clust_future_preds")
                    download_csv_button(filtered_future_clust, "cluster_assignments.csv")
                    dist_counts = pd.Series(cluster_labels.astype(str)).value_counts().reset_index()
                    dist_counts.columns = ["Cluster","Count"]
                    fig_cdist = px.bar(dist_counts, x="Cluster", y="Count", template=pt(), color_discrete_sequence=["#1a1a2e"])
                    st.plotly_chart(fig_cdist, use_container_width=True)
        except Exception as e: st.error(f"Error: {e}")


# ============================================================
# Main
# ============================================================
def main():
    render_sidebar()

    st.markdown("<div class='dashboard-title'>Universal ML Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dashboard-subtitle'>"
        "CSV &nbsp;·&nbsp; Excel &nbsp;·&nbsp; JSON"
        "&nbsp;&nbsp;|&nbsp;&nbsp;"
        "Preprocess &nbsp;·&nbsp; Explore &nbsp;·&nbsp; Model &nbsp;·&nbsp; Cluster"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    tabs = st.tabs([
        "📁 Data Manager",
        "🔧 Preprocessing",
        "📊 EDA & Visualization",
        "🧠 Dimensionality Reduction",
        "📈 Regression",
        "🎯 Classification",
        "🔗 Clustering"
    ])

    with tabs[0]: tab_data_manager()
    with tabs[1]: tab_preprocessing()
    with tabs[2]: tab_eda()
    with tabs[3]: tab_pca()
    with tabs[4]: tab_regression()
    with tabs[5]: tab_classification()
    with tabs[6]: tab_clustering()


if __name__ == "__main__":
    main()
