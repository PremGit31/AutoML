"""
NLP Dashboard — nlp_app2.py
============================
Standalone text / NLP analysis companion to the Universal ML Dashboard.
Tabs:
  1. Corpus Manager      — upload .txt/.log/.csv/.json/.xlsx/.pdf/.docx/.pptx,
                           append files, build master corpus, persistent file log
  2. Text Cleaning       — sequential spaCy-driven pipeline with per-step expanders,
                           live banner, and persistent Applied/Unsuccessful status badges
  3. Text EDA            — length distributions, n-gram charts, word clouds, KWIC search
  4. Vectorization       — TF-IDF / CountVectorizer  +  Sentence Transformers (GPU-aware,
                           explicit device selection, FP16), PCA dimensionality reduction
                           (any number of components, for downstream tasks) and separate
                           PCA / t-SNE 2D/3D projections (for plotting only)
  5. Text Classification — Naïve Bayes, LinearSVC, Logistic Regression, Random Forest;
                           confusion matrix, classification report, bulk inference
                           (correctly PCA-aware at inference time)
  6. Topic Modeling      — LDA (CountVec) / NMF (TF-IDF) + K-Means clustering on embeddings;
                           guarded against negative-value inputs (embeddings/PCA-reduced)

Requires:
pip install streamlit spacy textblob wordcloud scikit-learn gensim
              sentence-transformers plotly pandas numpy torch
              PyMuPDF python-docx python-pptx
  python -m spacy download en_core_web_sm
"""

import warnings
warnings.filterwarnings("ignore")

import re
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ──────────────────────────────────────────────────────────────
# GPU detection (single source of truth, checked once)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def detect_gpu():
    """Detect CUDA availability once per session. Returns dict with status info."""
    info = {"available": False, "name": None, "reason": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
        else:
            info["reason"] = "PyTorch installed but no CUDA device detected."
    except ImportError:
        info["reason"] = "PyTorch not installed."
    return info

GPU_INFO = detect_gpu()


# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="NLP Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
# Custom CSS  (mirrors app12 palette exactly)
# ──────────────────────────────────────────────────────────────
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
.stat-info   { border-left: 4px solid #3b82f6; }

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
.status-badge {
    display:inline-block; border-radius:6px; padding:3px 10px;
    font-size:12px; font-weight:600; margin-top:6px; margin-bottom:2px;
}
.status-badge-ok   { background:#d1fae5; color:#065f46; border:1px solid #10b981; }
.status-badge-fail { background:#fee2e2; color:#991b1b; border:1px solid #ef4444; }
.status-detail { font-size:11px; color:#6b7280; margin-left:6px; }
.gpu-badge {
    display:inline-block; border-radius:6px; padding:3px 10px;
    font-size:12px; font-weight:600; margin-bottom:8px;
}
.gpu-badge-on  { background:#d1fae5; color:#065f46; border:1px solid #10b981; }
.gpu-badge-off { background:#f3f4f6; color:#4b5563; border:1px solid #d1cdc7; }
.kwic-context { color: #6b7280; font-size: 13px; }
.kwic-hit     { color: #1a1a2e; font-weight: 700; background: #fef3c7;
                 padding: 0 3px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        # Corpus
        "nlp_corpus_files": {},          # {filename: {"text": [str], "source": str}}
        "nlp_raw_corpus": None,          # DataFrame with columns: doc_id, text, source
        "nlp_corpus_file_log": [],       # [{"filename","n_docs","appended_at"}] — running upload history
        "nlp_last_append_msg": None,
        # Cleaning
        "nlp_cleaned_corpus": None,      # DataFrame: doc_id, text, cleaned_text, source
        "nlp_cleaning_log": [],          # [step strings]
        # Vectorization
        "nlp_vectorized": None,          # dense np.ndarray or sparse matrix
        "nlp_vectorizer_obj": None,      # fitted sklearn vectorizer or SentenceTransformer
        "nlp_vectorizer_type": None,     # "tfidf" | "count" | "sentence_transformer"
        "nlp_feature_names": None,       # list[str] for TF-IDF / Count
        "nlp_vec_reduced": None,         # 2D/3D projection for scatter
        # PCA dimensionality reduction (distinct from the 2D/3D projection above)
        "nlp_vectorized_pre_pca": None,  # backup of matrix/type/features before PCA reduction
        "nlp_vectorizer_type_pre_pca": None,
        "nlp_feature_names_pre_pca": None,
        "nlp_pca_info": None,            # dict: n_components, explained_variance, cumulative
        "nlp_pca_transformer": None,     # the fitted sklearn PCA object, needed to project new text at inference time
        # Classification
        "nlp_labels": None,              # Series — target column
        "nlp_label_col": None,
        "nlp_trained_clf": None,
        "nlp_clf_features": None,        # vectorizer used for the trained model
        "nlp_clf_name": None,
        "nlp_clf_results": None,         # dict with metrics
        "nlp_label_encoder": None,
        "nlp_clf_pca_transformer": None, # PCA object used at train time, if any — needed to project new text at inference
        "nlp_clf_vectorizer_type": None, # base vectorizer type ("tfidf"/"count"/"sentence_transformer") at train time
        # Topic Modeling / Clustering
        "nlp_topic_model": None,
        "nlp_topic_type": None,
        "nlp_topic_results": None,
        "nlp_cluster_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ──────────────────────────────────────────────────────────────
# Lazy imports (heavy libs loaded once, cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_spacy():
    """Load spaCy with only the components needed for lemmatisation.
    Disabling 'ner' and 'parser' gives a significant speed boost."""
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        st.error(
            "spaCy model not found. Run: `python -m spacy download en_core_web_sm`"
        )
        return None

@st.cache_resource
def load_sentence_transformer(model_name="all-MiniLM-L6-v2", device=None):
    """device: 'cuda', 'cpu', or None (auto: GPU if available)."""
    try:
        from sentence_transformers import SentenceTransformer
        device = device or ("cuda" if GPU_INFO["available"] else "cpu")
        return SentenceTransformer(model_name, device=device)
    except Exception as e:
        st.error(f"Could not load Sentence Transformer: {e}")
        return None

def get_stopwords_spacy():
    nlp = load_spacy()
    if nlp:
        return nlp.Defaults.stop_words
    # fallback
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return set(ENGLISH_STOP_WORDS)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def pt():
    return "plotly_white"

COLOR_SEQ = ["#1a1a2e", "#374151", "#6366f1", "#10b981", "#f59e0b", "#3b82f6", "#ec4899"]

def corpus_info_banner(df, label="Corpus"):
    n_docs  = len(df)
    avg_w   = df["text"].apply(lambda x: len(str(x).split())).mean()
    avg_c   = df["text"].apply(lambda x: len(str(x))).mean()
    sources = df["source"].nunique() if "source" in df.columns else 1
    st.markdown(f"""
    <div class='active-banner'>
        📚 <b>{label}</b> &nbsp;·&nbsp;
        {n_docs:,} documents &nbsp;·&nbsp;
        avg {avg_w:.0f} words / doc &nbsp;·&nbsp;
        avg {avg_c:.0f} chars / doc &nbsp;·&nbsp;
        {sources} source(s)
    </div>
    """, unsafe_allow_html=True)

def download_csv_button(df, filename="result.csv", label="⬇ Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def save_result_widget(df, default_name, widget_key):
    with st.expander("💾 Save this result to session?", expanded=False):
        save_name = st.text_input("Name", value=default_name, key=f"sv_name_{widget_key}")
        if st.button("✅ Save", key=f"sv_btn_{widget_key}"):
            if save_name.strip():
                st.session_state[f"nlp_saved_{save_name.strip()}"] = df.copy()
                st.success(f"Saved as **{save_name.strip()}**")
            else:
                st.warning("Enter a name first.")

def get_active_text_series():
    """Return (series, label) of the best available text column."""
    if st.session_state.nlp_cleaned_corpus is not None:
        df = st.session_state.nlp_cleaned_corpus
        return df["cleaned_text"], "cleaned_text"
    elif st.session_state.nlp_raw_corpus is not None:
        df = st.session_state.nlp_raw_corpus
        return df["text"], "text (raw)"
    return None, None

def get_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_ngram_freq(series, n, top_k=20):
    all_ngrams = []
    for doc in series.dropna():
        tokens = str(doc).lower().split()
        all_ngrams.extend(get_ngrams(tokens, n))
    counter = Counter(all_ngrams)
    df = pd.DataFrame(counter.most_common(top_k), columns=["ngram", "count"])
    return df

def cleaning_log_banner():
    steps = st.session_state.nlp_cleaning_log
    if steps:
        lines = "".join(
            f"<div class='log-entry'>✓ Step {i}: {s}</div>"
            for i, s in enumerate(steps, 1)
        )
        st.markdown(f"<div style='margin-bottom:12px'><b>Active Cleaning Pipeline</b><br>{lines}</div>",
                    unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Stateful "Apply" buttons
# Every technique-applying button in the app uses this pair of
# helpers so the outcome (success / failure) is written to
# session_state and therefore SURVIVES the st.rerun() that follows
# it — instead of a st.success() banner that flashes and vanishes.
# ──────────────────────────────────────────────────────────────
def stateful_apply_button(label, status_key, key=None, **button_kwargs):
    """Render a button and, right below it, a persistent status badge
    reflecting the outcome of the LAST time it was clicked (if any).
    Returns True exactly when the button was clicked this run."""
    clicked = st.button(label, key=key or f"btn_{status_key}", **button_kwargs)
    status = st.session_state.get(status_key)
    if status:
        if status["ok"]:
            st.markdown(
                f"<span class='status-badge status-badge-ok'>✅ Applied Successfully</span>"
                f"<span class='status-detail'>{status['msg']}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<span class='status-badge status-badge-fail'>❌ Unsuccessful</span>"
                f"<span class='status-detail'>{status['msg']}</span>",
                unsafe_allow_html=True,
            )
    return clicked

def set_apply_status(status_key, ok, msg=""):
    st.session_state[status_key] = {"ok": bool(ok), "msg": msg, "ts": time.time()}

def is_nonnegative(X):
    """True if the matrix/array has no negative entries (NMF/MultinomialNB requirement)."""
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.min() >= 0 if X.nnz > 0 else True
    except ImportError:
        pass
    return bool(np.all(np.asarray(X) >= 0))

def gpu_status_badge():
    """Small persistent badge showing whether GPU acceleration is active."""
    if GPU_INFO["available"]:
        st.markdown(
            f"<span class='gpu-badge gpu-badge-on'>🎮 GPU active — {GPU_INFO['name']}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span class='gpu-badge gpu-badge-off'>💻 CPU only "
            f"({GPU_INFO['reason'] or 'no GPU detected'})</span>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────
# File loading
# ──────────────────────────────────────────────────────────────
import io
def load_text_file(uploaded_file):
    """Return list[str] of documents from any supported file type."""
    name = uploaded_file.name
    ext  = name.rsplit(".", 1)[-1].lower()

    if ext in ("txt", "log"):
        raw  = uploaded_file.read().decode("utf-8", errors="replace")
        docs = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return docs, f"{len(docs):,} lines"

    elif ext == "pdf":
        import fitz  # PyMuPDF
        doc  = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        docs = [page.get_text().strip() for page in doc if page.get_text().strip()]
        return docs, f"{len(docs):,} pages"

    elif ext == "docx":
        import docx
        doc  = docx.Document(uploaded_file)
        docs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return docs, f"{len(docs):,} paragraphs"

    elif ext == "pptx":
        import pptx
        prs  = pptx.Presentation(uploaded_file)
        docs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    docs.append(shape.text.strip())
        return docs, f"{len(docs):,} shapes"

    elif ext == "csv":
        df = pd.read_csv(uploaded_file)
        return df, list(df.columns)

    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(uploaded_file)
        return df, list(df.columns)

    elif ext == "json":
        raw = json.load(uploaded_file)
        if isinstance(raw, list):
            df = pd.json_normalize(raw)
        elif isinstance(raw, dict):
            list_keys = [k for k, v in raw.items() if isinstance(v, list)]
            records   = raw[list_keys[0]] if list_keys else [raw]
            df = pd.json_normalize(records)
        else:
            raise ValueError("Unsupported JSON structure.")
        return df, list(df.columns)

    else:
        raise ValueError(f"Unsupported extension: .{ext}")

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 NLP Dashboard")
        gpu_status_badge()
        st.markdown("---")

        sb_t1, sb_t2 = st.tabs(["📚 Corpus", "💾 Results"])

        with sb_t1:
            raw = st.session_state.nlp_raw_corpus
            if raw is None:
                st.info("No corpus loaded yet.")
            else:
                st.markdown(f"**Documents:** {len(raw):,}")
                src_count = raw['source'].nunique() if 'source' in raw.columns else 'N/A'
                st.markdown(f"**Sources:** {src_count}")
                cleaned = st.session_state.nlp_cleaned_corpus
                if cleaned is not None:
                    st.markdown(f"**Cleaned:** ✅ {len(st.session_state.nlp_cleaning_log)} step(s)")
                vec_type = st.session_state.nlp_vectorizer_type
                if vec_type:
                    vec_label = {
                        "tfidf": "TF-IDF", "count": "Count (BoW)",
                        "sentence_transformer": "Sentence Transformer embeddings",
                        "pca_reduced": "PCA-reduced",
                    }.get(vec_type, vec_type)
                    pca_info = st.session_state.nlp_pca_info
                    if vec_type == "pca_reduced" and pca_info:
                        base = {
                            "tfidf": "TF-IDF", "count": "Count (BoW)",
                            "sentence_transformer": "Sentence Transformer",
                        }.get(st.session_state.nlp_vectorizer_type_pre_pca, "features")
                        vec_label = f"{base} → PCA ({pca_info['n_components']} dims)"
                    st.markdown(f"**Vectorized:** ✅ {vec_label}")

        with sb_t2:
            saved_keys = [k for k in st.session_state if k.startswith("nlp_saved_")]
            if not saved_keys:
                st.info("No results saved yet.")
            else:
                for k in saved_keys:
                    name = k.replace("nlp_saved_", "")
                    df   = st.session_state[k]
                    st.markdown(f"**{name}**")
                    st.caption(f"{df.shape[0]:,} × {df.shape[1]}")
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ {name}", csv, file_name=f"{name}.csv",
                                       mime="text/csv", key=f"sb_dl_{k}")

        st.markdown("---")
        st.markdown("""
**Navigation**
1. 📚 Corpus Manager
2. 🧹 Text Cleaning
3. 📊 Text EDA
4. 🔢 Vectorization
5. 🎯 Text Classification
6. 🗂 Topic Modeling
        """)

        steps = st.session_state.nlp_cleaning_log
        if steps:
            st.markdown("---")
            st.markdown("**Cleaning Pipeline**")
            for i, s in enumerate(steps, 1):
                st.markdown(f"<div class='log-entry'>✓ {i}. {s}</div>",
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 1 — Corpus Manager
# ══════════════════════════════════════════════════════════════
def tab_corpus_manager():
    st.header("📚 Corpus Manager")
    st.markdown(
        "Upload `.txt`, `.pdf`, `.docx`, `.pptx`, `.log`, `.csv`, `.json`, or `.xlsx` files. "
        "Select the text column for structured files. "
        "Use **Append** to concatenate multiple files into a single master corpus."
    )

    # ── 1. Upload ──
    st.subheader("1. Upload Files")
    uploaded = st.file_uploader(
        "Upload file(s)",
        type=["txt", "log", "csv", "json", "xlsx", "xls", "pdf", "docx", "pptx"],
        accept_multiple_files=True,
        key="corpus_uploader",
    )

    pending = {}   # {filename: list[str]}  awaiting append


    for f in uploaded:
            ext = f.name.rsplit(".", 1)[-1].lower()
            try:
                result, meta = load_text_file(f)
            except Exception as e:
                st.error(f"**{f.name}**: {e}")
                continue

            if isinstance(result, list):
                # txt, log, pdf, docx, pptx — meta is an info string
                docs = result
                st.success(f"**{f.name}** — {meta}")
                pending[f.name] = docs

            else:
                # csv, xlsx, json — result is a DataFrame, meta is column list
                df_file = result
                st.markdown(f"**{f.name}** — {df_file.shape[0]:,} rows, {df_file.shape[1]} cols")
                text_col = st.selectbox(
                    f"Select text column from **{f.name}**",
                    meta, key=f"col_sel_{f.name}"
                )
                label_col = st.selectbox(
                    f"Select label column from **{f.name}** (optional)",
                    ["None"] + meta, key=f"lbl_sel_{f.name}"
                )
                docs = df_file[text_col].dropna().astype(str).tolist()
                pending[f.name] = docs

                if label_col != "None":
                    st.session_state["nlp_pending_labels"] = df_file[label_col]
                    st.session_state["nlp_label_col"] = label_col
                    st.caption(f"Label column **{label_col}** will be used in Classification tab.")

                st.caption(f"↳ {len(docs):,} non-empty text documents found in `{text_col}`.")
    
    st.divider()

    # ── 2. Append to Corpus ──
    st.subheader("2. Build / Append to Master Corpus")
    st.caption(
        "Each click appends the currently uploaded files to the existing corpus. "
        "Use 'Reset Corpus' to start fresh."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if st.button("➕ Append to Corpus", key="btn_append_corpus") and pending:
            new_rows = []
            for fname, docs in pending.items():
                for doc in docs:
                    new_rows.append({"text": doc, "source": fname})
            new_df = pd.DataFrame(new_rows, columns=["text", "source"]) if new_rows else pd.DataFrame(columns=["text", "source"])
            existing = st.session_state.nlp_raw_corpus
            if existing is not None:
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined["doc_id"] = range(len(combined))
            st.session_state.nlp_raw_corpus = combined

            # Track which files were appended, so the corpus contents are
            # always visible — not just a one-time upload confirmation.
            file_log = st.session_state.get("nlp_corpus_file_log") or []
            for fname, docs in pending.items():
                file_log.append({
                    "filename": fname,
                    "n_docs": len(docs),
                    "appended_at": time.strftime("%H:%M:%S"),
                })
            st.session_state.nlp_corpus_file_log = file_log

            # Reset downstream state when corpus changes
            st.session_state.nlp_cleaned_corpus = None
            st.session_state.nlp_cleaning_log   = []
            st.session_state.nlp_vectorized      = None
            st.session_state.nlp_vectorizer_type = None
            new_names = ", ".join(pending.keys())
            st.session_state["nlp_last_append_msg"] = (
                f"Appended **{len(pending)} file(s)** ({new_names}) — "
                f"{len(new_rows):,} documents. Corpus now contains "
                f"**{combined['source'].nunique()} file(s)**, {len(combined):,} documents total."
            )
            st.rerun()
    with col_b:
        if st.button("🗑 Reset Corpus", key="btn_reset_corpus"):
            for key in ["nlp_raw_corpus", "nlp_cleaned_corpus", "nlp_vectorized",
                        "nlp_vectorizer_type", "nlp_vectorizer_obj", "nlp_vec_reduced",
                        "nlp_topic_results", "nlp_cluster_results", "nlp_clf_results",
                        "nlp_last_append_msg", "nlp_feature_names",
                        "nlp_vectorized_pre_pca", "nlp_vectorizer_type_pre_pca",
                        "nlp_feature_names_pre_pca", "nlp_pca_info", "nlp_pca_transformer",
                        "nlp_clf_pca_transformer", "nlp_clf_vectorizer_type",
                        "nlp_trained_clf", "nlp_clf_features", "nlp_label_encoder"]:
                st.session_state[key] = None
            st.session_state.nlp_cleaning_log = []
            st.session_state.nlp_corpus_file_log = []
            st.rerun()

    if st.session_state.get("nlp_last_append_msg"):
        st.success(f"✅ {st.session_state['nlp_last_append_msg']}")

    # ── Persistent "files in corpus" list — grows with every append ──
    file_log = st.session_state.get("nlp_corpus_file_log") or []
    if file_log:
        with st.expander(f"📁 Files in Corpus ({len(file_log)} upload event(s))", expanded=True):
            log_df = pd.DataFrame(file_log)[["filename", "n_docs", "appended_at"]]
            log_df.columns = ["File", "Documents Added", "Appended At"]
            st.dataframe(log_df, use_container_width=True, hide_index=True)

    # ── 3. Preview ──
    corpus = st.session_state.nlp_raw_corpus
    if corpus is None:
        st.info("No corpus loaded yet — upload files and click **Append to Corpus**.")
        return

    st.divider()
    st.subheader("3. Corpus Preview")
    corpus_info_banner(corpus, "Master Corpus")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Documents", f"{len(corpus):,}")
    c2.metric("Unique Sources", corpus["source"].nunique())
    c3.metric("Avg Words / Doc",
              f"{corpus['text'].apply(lambda x: len(str(x).split())).mean():.1f}")
    c4.metric("Avg Chars / Doc",
              f"{corpus['text'].apply(lambda x: len(str(x))).mean():.1f}")

    with st.expander("📋 Browse Documents", expanded=True):
        st.dataframe(corpus[["doc_id", "source", "text"]].head(100), use_container_width=True)

    if corpus["source"].nunique() > 1:
        src_counts = corpus["source"].value_counts().reset_index()
        src_counts.columns = ["source", "count"]
        fig_src = px.bar(
            src_counts, x="source", y="count", template=pt(),
            title="Documents per Source", color_discrete_sequence=COLOR_SEQ,
        )
        st.plotly_chart(fig_src, use_container_width=True)

    download_csv_button(corpus, "master_corpus.csv", "⬇ Download Corpus as CSV")


# ══════════════════════════════════════════════════════════════
# TAB 2 — Text Cleaning & Normalisation
# ══════════════════════════════════════════════════════════════
def tab_text_cleaning():
    st.header("🧹 Text Cleaning & Normalisation")

    corpus = st.session_state.nlp_raw_corpus
    if corpus is None:
        st.warning("Load a corpus in the **Corpus Manager** tab first.")
        return

    has_cleaned = st.session_state.nlp_cleaned_corpus is not None
    steps_done  = st.session_state.nlp_cleaning_log

    top_c1, top_c2 = st.columns([4, 1])
    with top_c1:
        if has_cleaned:
            st.info(f"✏️ Continuing pipeline — {len(steps_done)} step(s) already applied.")
        else:
            st.caption("No cleaning applied yet. Apply steps sequentially using the expanders below.")
    with top_c2:
        if has_cleaned and st.button("↺ Start Fresh", key="clean_restart"):
            st.session_state.nlp_cleaned_corpus = None
            st.session_state.nlp_cleaning_log   = []
            st.rerun()

    # Work on cleaned if exists, otherwise raw
    work_series = (
        st.session_state.nlp_cleaned_corpus["cleaned_text"].copy()
        if has_cleaned
        else corpus["text"].copy()
    )

    corpus_info_banner(corpus, "Source Corpus")
    cleaning_log_banner()

    st.markdown("---")
    st.caption("Apply steps in order. Each step commits immediately and is logged.")

    # ── Step 1: Noise Removal ──
    with st.expander("Step 1 — 🔇 Noise Removal", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        do_lower   = c1.checkbox("Lowercase",          value=True,  key="cl_lower")
        do_html    = c2.checkbox("Strip HTML tags",     value=True,  key="cl_html")
        do_urls    = c3.checkbox("Remove URLs",         value=True,  key="cl_urls")
        do_punct   = c4.checkbox("Remove punctuation",  value=True,  key="cl_punct")
        do_digits  = c5.checkbox("Remove digits",       value=False, key="cl_digits")
        do_extra   = c1.checkbox("Collapse whitespace", value=True,  key="cl_ws")

        # Live preview
        sample = str(work_series.iloc[0])[:300]
        preview = sample
        if do_lower:   preview = preview.lower()
        if do_html:    preview = re.sub(r"<[^>]+>", " ", preview)
        if do_urls:    preview = re.sub(r"https?://\S+|www\.\S+", " ", preview)
        if do_punct:   preview = re.sub(r"[^\w\s]", " ", preview)
        if do_digits:  preview = re.sub(r"\d+", " ", preview)
        if do_extra:   preview = re.sub(r"\s+", " ", preview).strip()

        st.markdown("**Preview (first doc):**")
        st.code(preview, language=None)

        if stateful_apply_button("✅ Apply Noise Removal", "status_cl_noise"):
            try:
                def noise_clean(text):
                    t = str(text)
                    if do_lower:  t = t.lower()
                    if do_html:   t = re.sub(r"<[^>]+>", " ", t)
                    if do_urls:   t = re.sub(r"https?://\S+|www\.\S+", " ", t)
                    if do_punct:  t = re.sub(r"[^\w\s]", " ", t)
                    if do_digits: t = re.sub(r"\d+", " ", t)
                    if do_extra:  t = re.sub(r"\s+", " ", t).strip()
                    return t
                with st.spinner("Applying noise removal…"):
                    work_series = work_series.apply(noise_clean)
                ops = [x for x, f in [
                    ("lowercase", do_lower), ("strip HTML", do_html), ("remove URLs", do_urls),
                    ("remove punctuation", do_punct), ("remove digits", do_digits),
                    ("collapse whitespace", do_extra)
                ] if f]
                _commit_cleaning(corpus, work_series, f"Noise removal: {', '.join(ops)}")
                set_apply_status("status_cl_noise", True,
                                  f"{len(ops)} operation(s) on {len(work_series):,} docs")
            except Exception as e:
                set_apply_status("status_cl_noise", False, str(e))
            st.rerun()

    # ── Step 2: Stopword Removal ──
    with st.expander("Step 2 — 🛑 Stopword Removal"):
        sw_source = st.radio(
            "Stopword list", ["Default (spaCy)", "Upload custom list"],
            horizontal=True, key="cl_sw_src"
        )
        custom_sw = set()
        if sw_source == "Upload custom list":
            sw_file = st.file_uploader("Upload .txt — one word per line", type=["txt"],
                                       key="cl_sw_file")
            if sw_file:
                custom_sw = set(
                    sw_file.read().decode("utf-8").splitlines()
                )
                st.caption(f"{len(custom_sw):,} custom stopwords loaded.")

        # Preview
        sw_set = custom_sw if custom_sw else get_stopwords_spacy()
        sample_toks = str(work_series.iloc[0]).split()
        filtered_sample = [t for t in sample_toks if t.lower() not in sw_set]
        st.markdown("**Preview (first doc after removal):**")
        st.code(" ".join(filtered_sample[:60]), language=None)

        if stateful_apply_button("✅ Apply Stopword Removal", "status_cl_sw"):
            try:
                sw = custom_sw if custom_sw else get_stopwords_spacy()
                with st.spinner("Removing stopwords…"):
                    work_series = work_series.apply(
                        lambda t: " ".join(tok for tok in str(t).split() if tok.lower() not in sw)
                    )
                src_label = "custom" if custom_sw else "spaCy default"
                _commit_cleaning(corpus, work_series, f"Stopword removal ({src_label}, {len(sw):,} words)")
                set_apply_status("status_cl_sw", True, f"{src_label} list, {len(sw):,} words")
            except Exception as e:
                set_apply_status("status_cl_sw", False, str(e))
            st.rerun()

    # ── Step 3: Stemming / Lemmatisation ──
    with st.expander("Step 3 — 🌱 Stemming / Lemmatisation"):
        morph_method = st.radio(
            "Method",
            ["spaCy Lemmatisation (accurate)", "Porter Stemmer (fast)"],
            key="cl_morph",
        )
        st.caption(
            "Lemmatisation: _running_ → _run_ (context-aware). "
            "Stemming: _running_ → _run_ (rule-based, faster)."
        )

        if stateful_apply_button("✅ Apply", "status_cl_morph"):
            if "Lemmatisation" in morph_method:
                try:
                    nlp = load_spacy()
                    if nlp is None:
                        raise RuntimeError("spaCy model not available (run: python -m spacy download en_core_web_sm).")
                    n_docs = len(work_series)
                    with st.spinner(f"Lemmatising {n_docs:,} docs with spaCy (batched)…"):
                        texts = work_series.astype(str).tolist()
                        lemmatised = [
                            " ".join(tok.lemma_ for tok in doc if not tok.is_space)
                            for doc in nlp.pipe(texts, batch_size=512, n_process=1)
                        ]
                        work_series = pd.Series(lemmatised, index=work_series.index)
                    _commit_cleaning(corpus, work_series, "spaCy lemmatisation")
                    set_apply_status("status_cl_morph", True, f"Lemmatised {n_docs:,} docs")
                except Exception as e:
                    set_apply_status("status_cl_morph", False, str(e))
            else:
                try:
                    try:
                        from nltk.stem import PorterStemmer
                        import nltk; nltk.download("punkt", quiet=True)
                    except ImportError:
                        raise RuntimeError("NLTK not installed. Run: pip install nltk")
                    ps = PorterStemmer()
                    with st.spinner("Stemming…"):
                        work_series = work_series.apply(
                            lambda t: " ".join(ps.stem(tok) for tok in str(t).split())
                        )
                    _commit_cleaning(corpus, work_series, "Porter stemming")
                    set_apply_status("status_cl_morph", True, f"Stemmed {len(work_series):,} docs")
                except Exception as e:
                    set_apply_status("status_cl_morph", False, str(e))
            st.rerun()

    # ── Step 4: Tokenisation preview ──
    with st.expander("Step 4 — 🔤 Tokenisation Preview"):
        st.caption(
            "Tokenisation is handled internally during vectorisation. "
            "This step shows you what the tokens will look like before you proceed."
        )
        n_preview = st.slider("Docs to preview", 1, 10, 3, key="cl_tok_preview")
        for i in range(min(n_preview, len(work_series))):
            tokens = str(work_series.iloc[i]).split()
            st.markdown(f"**Doc {i}** — {len(tokens)} tokens")
            st.code(" | ".join(tokens[:50]) + ("…" if len(tokens) > 50 else ""), language=None)

    # ── Before / After stats ──
    st.divider()
    if has_cleaned:
        st.subheader("Before / After Statistics")
        raw_series = corpus["text"]
        cl_series  = st.session_state.nlp_cleaned_corpus["cleaned_text"]

        raw_wc  = raw_series.apply(lambda x: len(str(x).split())).mean()
        cl_wc   = cl_series.apply(lambda x: len(str(x).split())).mean()
        raw_vc  = len(set(" ".join(raw_series.dropna().astype(str)).split()))
        cl_vc   = len(set(" ".join(cl_series.dropna().astype(str)).split()))

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(f"""
            <div class='stat-card stat-warn'>
                <b>Raw text</b><br>
                Avg words/doc: <b>{raw_wc:.1f}</b><br>
                Vocabulary size: <b>{raw_vc:,}</b>
            </div>""", unsafe_allow_html=True)
        with bc2:
            st.markdown(f"""
            <div class='stat-card stat-after'>
                <b>Cleaned text</b><br>
                Avg words/doc: <b>{cl_wc:.1f}</b><br>
                Vocabulary size: <b>{cl_vc:,}</b>
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 View Cleaned Corpus (sample)", expanded=False):
            view_df = st.session_state.nlp_cleaned_corpus[
                ["doc_id", "source", "text", "cleaned_text"]
            ].head(100)
            st.dataframe(view_df, use_container_width=True)
        download_csv_button(
            st.session_state.nlp_cleaned_corpus,
            "cleaned_corpus.csv",
            "⬇ Download Cleaned Corpus"
        )


def _commit_cleaning(corpus_df, cleaned_series, step_label):
    """Persist cleaned text into session state and log the step."""
    if st.session_state.nlp_cleaned_corpus is not None:
        result_df = st.session_state.nlp_cleaned_corpus.copy()
        result_df["cleaned_text"] = cleaned_series.values
    else:
        result_df = corpus_df.copy()
        result_df["cleaned_text"] = cleaned_series.values

    st.session_state.nlp_cleaned_corpus = result_df
    st.session_state.nlp_cleaning_log.append(step_label)
    # Reset downstream when cleaning changes
    st.session_state.nlp_vectorized      = None
    st.session_state.nlp_vectorizer_type = None
    st.session_state.nlp_vec_reduced     = None


# ══════════════════════════════════════════════════════════════
# TAB 3 — Text EDA & Lexical Analysis
# ══════════════════════════════════════════════════════════════
def tab_text_eda():
    st.header("📊 Text EDA & Lexical Analysis")

    text_series, label = get_active_text_series()
    if text_series is None:
        st.warning("Load a corpus first.")
        return

    corpus_info_banner(st.session_state.nlp_raw_corpus, f"Active text: {label}")

    eda_tabs = st.tabs([
        "📏 Length Distribution",
        "📈 N-gram Frequency",
        "☁ Word Cloud",
        "🔍 KWIC Search",
    ])

    # ── Length Distribution ──
    with eda_tabs[0]:
        st.subheader("Document Length Distributions")
        lengths_df = pd.DataFrame({
            "word_count": text_series.apply(lambda x: len(str(x).split())),
            "char_count": text_series.apply(lambda x: len(str(x))),
        })
        c1, c2 = st.columns(2)
        with c1:
            fig_wc = px.histogram(
                lengths_df, x="word_count", nbins=50, template=pt(),
                title="Word Count Distribution",
                color_discrete_sequence=[COLOR_SEQ[0]],
                labels={"word_count": "Words per Document"},
            )
            st.plotly_chart(fig_wc, use_container_width=True)
        with c2:
            fig_cc = px.histogram(
                lengths_df, x="char_count", nbins=50, template=pt(),
                title="Character Count Distribution",
                color_discrete_sequence=[COLOR_SEQ[2]],
                labels={"char_count": "Characters per Document"},
            )
            st.plotly_chart(fig_cc, use_container_width=True)

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Median Words",    f"{lengths_df['word_count'].median():.0f}")
        dc2.metric("Max Words",       f"{lengths_df['word_count'].max():,}")
        dc3.metric("Median Chars",    f"{lengths_df['char_count'].median():.0f}")
        dc4.metric("Max Chars",       f"{lengths_df['char_count'].max():,}")

    # ── N-gram Frequency ──
    with eda_tabs[1]:
        st.subheader("Most Frequent N-grams")
        ng_col1, ng_col2 = st.columns([1, 3])
        with ng_col1:
            ng_n    = st.selectbox("N-gram type", [1, 2, 3],
                                   format_func=lambda n: {1:"Unigrams",2:"Bigrams",3:"Trigrams"}[n],
                                   key="eda_ngram_n")
            top_k   = st.slider("Top K", 10, 50, 20, key="eda_topk")
        with ng_col2:
            with st.spinner("Computing…"):
                ngram_df = compute_ngram_freq(text_series, ng_n, top_k)
            if ngram_df.empty:
                st.info("Not enough tokens — try cleaning first.")
            else:
                fig_ng = px.bar(
                    ngram_df.sort_values("count"), x="count", y="ngram",
                    orientation="h", template=pt(),
                    title=f"Top {top_k} {['','Uni','Bi','Tri'][ng_n]}grams",
                    color_discrete_sequence=[COLOR_SEQ[0]],
                    labels={"count": "Frequency", "ngram": ""},
                )
                fig_ng.update_layout(height=max(400, top_k * 22))
                st.plotly_chart(fig_ng, use_container_width=True)

    # ── Word Cloud ──
    with eda_tabs[2]:
        st.subheader("Word Cloud")
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
        except ImportError:
            st.error("Install wordcloud: `pip install wordcloud matplotlib`")
            return

        max_words  = st.slider("Max words", 50, 300, 150, key="wc_max")
        bg_color   = st.color_picker("Background", "#f8f7f4", key="wc_bg")
        full_text  = " ".join(text_series.dropna().astype(str).tolist())
        if len(full_text.strip()) < 10:
            st.warning("Not enough text to generate a word cloud.")
        else:
            wc = WordCloud(
                width=1200, height=500, max_words=max_words,
                background_color=bg_color,
                colormap="cividis",
            ).generate(full_text)
            fig_wcloud, ax = plt.subplots(figsize=(14, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wcloud.patch.set_facecolor(bg_color)
            st.pyplot(fig_wcloud, use_container_width=True)

    # ── KWIC Search ──
    with eda_tabs[3]:
        st.subheader("Key Word In Context (KWIC)")
        st.caption(
            "Search for a term and see the surrounding words in each document where it appears."
        )
        kwic_query   = st.text_input("Search term", placeholder="e.g. climate", key="kwic_query")
        kwic_window  = st.slider("Context window (words each side)", 3, 15, 7, key="kwic_window")
        case_sens    = st.checkbox("Case-sensitive", value=False, key="kwic_case")

        if kwic_query.strip():
            results = []
            pattern = re.compile(
                re.escape(kwic_query.strip()),
                flags=0 if case_sens else re.IGNORECASE,
            )
            for idx, doc in text_series.items():
                tokens = str(doc).split()
                for i, tok in enumerate(tokens):
                    if pattern.search(tok):
                        left  = " ".join(tokens[max(0, i-kwic_window):i])
                        right = " ".join(tokens[i+1:i+1+kwic_window])
                        results.append({
                            "doc_id": idx,
                            "left_context":  left,
                            "hit": tok,
                            "right_context": right,
                        })
                if len(results) >= 500:
                    break

            if not results:
                st.info(f"No matches found for **{kwic_query}**.")
            else:
                st.caption(f"{len(results):,} occurrences found (max 500 shown).")
                kwic_df = pd.DataFrame(results)
                # Render with inline HTML for context styling
                rows_html = ""
                for _, row in kwic_df.head(200).iterrows():
                    rows_html += (
                        f"<tr>"
                        f"<td style='color:#6b7280;font-size:12px;padding:4px 8px'>{row['doc_id']}</td>"
                        f"<td style='text-align:right;color:#6b7280;font-size:13px;padding:4px 8px'>{row['left_context']}</td>"
                        f"<td style='padding:4px 8px'>"
                        f"<span class='kwic-hit'>{row['hit']}</span></td>"
                        f"<td style='color:#6b7280;font-size:13px;padding:4px 8px'>{row['right_context']}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse'>"
                    f"<thead><tr>"
                    f"<th style='text-align:left;color:#374151;font-size:12px;padding:4px 8px'>Doc</th>"
                    f"<th style='text-align:right;color:#374151;font-size:12px;padding:4px 8px'>Left</th>"
                    f"<th style='color:#374151;font-size:12px;padding:4px 8px'>Hit</th>"
                    f"<th style='color:#374151;font-size:12px;padding:4px 8px'>Right</th>"
                    f"</tr></thead><tbody>{rows_html}</tbody></table>",
                    unsafe_allow_html=True,
                )
                download_csv_button(kwic_df, "kwic_results.csv", "⬇ Download KWIC Results")


# ══════════════════════════════════════════════════════════════
# TAB 4 — Vectorization & Embeddings
# ══════════════════════════════════════════════════════════════
def tab_vectorization():
    st.header("🔢 Vectorization & Embeddings")
    st.markdown(
        "Convert your text into numerical representations. "
        "This tab bridges the corpus to the ML tabs."
    )

    text_series, label = get_active_text_series()
    if text_series is None:
        st.warning("Load a corpus first.")
        return

    corpus_info_banner(st.session_state.nlp_raw_corpus, f"Active text: {label}")

    vec_tabs = st.tabs([
        "📐 Traditional (TF-IDF / Count)",
        "🤖 Sentence Transformers",
        "🎚 PCA Reduction",
        "🗺 Projection (PCA / t-SNE)",
    ])

    # ── Traditional ──
    with vec_tabs[0]:
        st.subheader("Scikit-learn Vectorizer")
        v1, v2 = st.columns(2)
        with v1:
            vec_type = st.selectbox("Vectorizer", ["TF-IDF", "Count (Bag of Words)"],
                                    key="vec_type_sel")
        with v2:
            max_feat = st.number_input("Max features", 100, 50000, 5000, step=500,
                                       key="vec_max_feat")

        ng_min = st.select_slider("N-gram min", options=[1, 2, 3], value=1, key="vec_ng_min")
        ng_max = st.select_slider("N-gram max", options=[1, 2, 3], value=1, key="vec_ng_max")

        if stateful_apply_button("🚀 Vectorize", "status_vec_trad"):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
                docs = text_series.fillna("").astype(str).tolist()
                with st.spinner(f"Fitting {vec_type} vectorizer on {len(docs):,} docs…"):
                    if vec_type == "TF-IDF":
                        vec = TfidfVectorizer(max_features=max_feat,
                                             ngram_range=(ng_min, ng_max),
                                             sublinear_tf=True)
                    else:
                        vec = CountVectorizer(max_features=max_feat,
                                             ngram_range=(ng_min, ng_max))
                    X = vec.fit_transform(docs)
                st.session_state.nlp_vectorized      = X
                st.session_state.nlp_vectorizer_obj  = vec
                st.session_state.nlp_vectorizer_type = "tfidf" if vec_type == "TF-IDF" else "count"
                st.session_state.nlp_feature_names   = vec.get_feature_names_out().tolist()
                st.session_state.nlp_vec_reduced      = None
                # Clear any stale PCA-reduction state — it referred to the old matrix
                st.session_state.nlp_vectorized_pre_pca = None
                st.session_state.nlp_pca_info = None
                set_apply_status("status_vec_trad", True,
                                  f"matrix shape {X.shape[0]:,} × {X.shape[1]:,}")
            except Exception as e:
                set_apply_status("status_vec_trad", False, str(e))
            st.rerun()

        if (st.session_state.nlp_vectorizer_type in ("tfidf", "count")
                and st.session_state.nlp_vectorized is not None):
            X = st.session_state.nlp_vectorized
            feats = st.session_state.nlp_feature_names
            st.markdown(f"**Matrix:** {X.shape[0]:,} docs × {X.shape[1]:,} features  "
                        f"(sparsity: {100*(1 - X.nnz/(X.shape[0]*X.shape[1])):.1f}%)")
            # Top features by mean weight
            import scipy.sparse as sp
            mean_w = np.asarray(X.mean(axis=0)).flatten()
            top_idx = np.argsort(mean_w)[::-1][:30]
            top_df  = pd.DataFrame({"feature": [feats[i] for i in top_idx],
                                    "mean_weight": mean_w[top_idx]})
            fig_feat = px.bar(
                top_df.sort_values("mean_weight"), x="mean_weight", y="feature",
                orientation="h", template=pt(), title="Top 30 Features by Mean Weight",
                color_discrete_sequence=[COLOR_SEQ[0]],
            )
            fig_feat.update_layout(height=600)
            st.plotly_chart(fig_feat, use_container_width=True)

            # Downloadable dense sample
            dense_sample = pd.DataFrame(
                X[:500].toarray(), columns=feats
            )
            download_csv_button(dense_sample, "vectorized_sample.csv",
                                "⬇ Download Vectorized Matrix (first 500 docs)")

    # ── Sentence Transformers ──
    with vec_tabs[1]:
        st.subheader("Sentence Transformers (Semantic Embeddings)")
        st.caption(
            "Uses a pre-trained transformer to create dense embeddings."
        )
        gpu_status_badge()
        st.info("ℹ️ First run downloads the model (~90 MB for MiniLM). Subsequent runs use cache.")

        s1, s2 = st.columns(2)
        with s1:
            model_name = st.selectbox(
                "Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"],
                key="st_model_name",
            )
        with s2:
            device_choice = st.selectbox(
                "Device",
                ["Auto (GPU if available)", "Force CPU"] + (["Force GPU"] if GPU_INFO["available"] else []),
                key="st_device_choice",
            )

        # GPU lets us push much bigger batches through in one shot
        default_batch = 128 if GPU_INFO["available"] else 32
        batch_size = st.slider("Batch size", 16, 512, default_batch, key="st_batch")
        use_fp16 = False
        if GPU_INFO["available"] and device_choice != "Force CPU":
            use_fp16 = st.checkbox(
                "Use FP16 half-precision (faster on GPU, negligible accuracy loss)",
                value=True, key="st_fp16",
            )

        if stateful_apply_button("🚀 Generate Embeddings", "status_vec_st"):
            try:
                device = ("cpu" if device_choice == "Force CPU"
                          else "cuda" if (device_choice == "Force GPU" or GPU_INFO["available"])
                          else "cpu")
                model = load_sentence_transformer(model_name, device=device)
                if model is None:
                    raise RuntimeError("Could not load model.")
                if use_fp16 and device == "cuda":
                    model = model.half()
                docs = text_series.fillna("").astype(str).tolist()
                t0 = time.time()
                with st.spinner(f"Encoding {len(docs):,} docs with {model_name} on {device.upper()}…"):
                    embeddings = model.encode(
                        docs, batch_size=batch_size, show_progress_bar=False
                    )
                elapsed = time.time() - t0
                st.session_state.nlp_vectorized      = embeddings
                st.session_state.nlp_vectorizer_obj  = model
                st.session_state.nlp_vectorizer_type = "sentence_transformer"
                st.session_state.nlp_feature_names   = None
                st.session_state.nlp_vec_reduced      = None
                st.session_state.nlp_vectorized_pre_pca = None
                st.session_state.nlp_pca_info = None
                set_apply_status(
                    "status_vec_st", True,
                    f"shape {embeddings.shape[0]:,} × {embeddings.shape[1]} on {device.upper()} in {elapsed:.1f}s"
                )
            except Exception as e:
                set_apply_status("status_vec_st", False, str(e))
            st.rerun()

    # ── PCA Dimensionality Reduction ──
    with vec_tabs[2]:
        st.subheader("PCA Dimensionality Reduction")
        st.caption(
            "Distinct from the 2D/3D **Projection** tab (which is purely for plotting). "
            "This reduces the *actual* feature matrix used downstream — useful before "
            "Classification or Clustering when you have hundreds/thousands of features "
            "(e.g. TF-IDF vocab, or 384–768-dim embeddings). Fewer, decorrelated "
            "components can mean faster training and less noise, at the cost of some "
            "explained variance. Note: after reduction, components are no longer tied to "
            "individual words, so **Topic Modeling (LDA/NMF)** — which needs a word "
            "vocabulary — will require the original (un-reduced) TF-IDF/Count matrix."
        )

        if st.session_state.nlp_vectorized is None:
            st.info("Run vectorization first (TF-IDF, Count, or Sentence Transformers).")
        else:
            X_raw = st.session_state.nlp_vectorized
            try:
                import scipy.sparse as sp
                X_check = X_raw.toarray() if sp.issparse(X_raw) else X_raw
            except ImportError:
                X_check = X_raw
            n_samples, n_features = X_check.shape
            max_comp = max(2, min(n_samples, n_features) - 1)

            if st.session_state.nlp_pca_info:
                info = st.session_state.nlp_pca_info
                st.markdown(f"""
                <div class='stat-card stat-after'>
                    ✅ Active matrix is <b>PCA-reduced</b>: {info['n_components']} components,
                    {info['cumulative_variance']*100:.1f}% cumulative variance retained
                    (reduced from {info['original_features']:,} original features).
                </div>""", unsafe_allow_html=True)
                if st.button("↺ Revert to original (un-reduced) features", key="btn_pca_revert"):
                    st.session_state.nlp_vectorized      = st.session_state.nlp_vectorized_pre_pca
                    st.session_state.nlp_vectorizer_type = st.session_state.nlp_vectorizer_type_pre_pca
                    st.session_state.nlp_feature_names   = st.session_state.nlp_feature_names_pre_pca
                    st.session_state.nlp_vectorized_pre_pca = None
                    st.session_state.nlp_pca_info = None
                    st.session_state.nlp_pca_transformer = None
                    st.rerun()

            else:
                if st.button("📈 Compute Explained Variance Curve", key="btn_pca_curve"):
                    from sklearn.decomposition import PCA as skPCA
                    with st.spinner("Fitting PCA across component range…"):
                        curve_n = min(max_comp, 100)
                        pca_curve = skPCA(n_components=curve_n, random_state=42).fit(X_check)
                        cum_var = np.cumsum(pca_curve.explained_variance_ratio_)
                    fig_cum = px.line(
                        x=list(range(1, curve_n + 1)), y=cum_var, markers=True,
                        template=pt(), title="Cumulative Explained Variance vs. Components",
                        labels={"x": "Number of Components", "y": "Cumulative Variance Explained"},
                        color_discrete_sequence=[COLOR_SEQ[0]],
                    )
                    fig_cum.add_hline(y=0.9, line_dash="dot", annotation_text="90%")
                    st.plotly_chart(fig_cum, use_container_width=True)
                    n_for_90 = int(np.argmax(cum_var >= 0.9) + 1) if np.any(cum_var >= 0.9) else curve_n
                    st.caption(f"≈{n_for_90} components needed to reach 90% variance "
                               f"(out of {n_features:,} original features).")

                n_components = st.number_input(
                    f"Number of components (2 – {max_comp:,})",
                    min_value=2, max_value=max_comp,
                    value=min(50, max_comp), step=1, key="pca_n_components",
                )

                if stateful_apply_button("🎚 Reduce with PCA", "status_pca_reduce"):
                    try:
                        from sklearn.decomposition import PCA as skPCA
                        with st.spinner(f"Reducing {n_features:,} → {n_components} components…"):
                            pca = skPCA(n_components=int(n_components), random_state=42)
                            X_reduced = pca.fit_transform(X_check)
                            cum_var = float(np.sum(pca.explained_variance_ratio_))

                        # Back up the pre-PCA matrix so it can be restored
                        st.session_state.nlp_vectorized_pre_pca      = X_raw
                        st.session_state.nlp_vectorizer_type_pre_pca = st.session_state.nlp_vectorizer_type
                        st.session_state.nlp_feature_names_pre_pca   = st.session_state.nlp_feature_names

                        st.session_state.nlp_vectorized      = X_reduced
                        st.session_state.nlp_vectorizer_type = "pca_reduced"
                        st.session_state.nlp_feature_names   = None
                        st.session_state.nlp_vec_reduced      = None
                        st.session_state.nlp_pca_transformer = pca
                        st.session_state.nlp_pca_info = {
                            "n_components": int(n_components),
                            "cumulative_variance": cum_var,
                            "original_features": n_features,
                        }
                        set_apply_status("status_pca_reduce", True,
                                          f"{n_features:,} → {n_components} dims, "
                                          f"{cum_var*100:.1f}% variance retained")
                    except Exception as e:
                        set_apply_status("status_pca_reduce", False, str(e))
                    st.rerun()

    # ── Projection ──
    with vec_tabs[3]:
        st.subheader("Dimensionality Reduction for Visualisation")
        st.caption(
            "Capped at 2–3 dimensions on purpose — this is for plotting a scatter chart, "
            "which only humans-in-3D can read. For reducing the *feature matrix* itself "
            "(any number of components), use the **PCA Reduction** tab instead."
        )

        if st.session_state.nlp_vectorized is None:
            st.info("Run vectorization first (TF-IDF or Sentence Transformers).")
            return

        proj_method  = st.selectbox("Method", ["PCA", "t-SNE"], key="proj_method")
        proj_dims    = st.selectbox("Dimensions", [2, 3], key="proj_dims")
        proj_color   = "source"
        corpus_df    = st.session_state.nlp_raw_corpus

        if proj_method == "t-SNE":
            max_tsne = 5000
            n_docs   = (st.session_state.nlp_vectorized.shape[0]
                        if hasattr(st.session_state.nlp_vectorized, "shape")
                        else len(st.session_state.nlp_vectorized))
            if n_docs > max_tsne:
                st.warning(
                    f"t-SNE is slow on large corpora. Sampling {max_tsne:,} docs. "
                    "Switch to PCA for full corpus."
                )
            perplexity = st.slider("Perplexity", 5, 50, 30, key="tsne_perp")

        if stateful_apply_button("🗺 Run Projection", "status_proj"):
            try:
                from sklearn.decomposition import PCA as skPCA
                X = st.session_state.nlp_vectorized

                # Convert sparse → dense if needed
                try:
                    import scipy.sparse as sp
                    if sp.issparse(X):
                        X = X.toarray()
                except ImportError:
                    pass

                n_samples = X.shape[0]
                with st.spinner("Running projection…"):
                    if proj_method == "PCA":
                        n_comp = proj_dims
                        reducer = skPCA(n_components=n_comp, random_state=42)
                        coords  = reducer.fit_transform(X)
                    else:
                        from sklearn.manifold import TSNE
                        sample_idx = (np.random.choice(n_samples, min(max_tsne, n_samples),
                                                       replace=False)
                                      if n_samples > max_tsne else np.arange(n_samples))
                        X_sub = X[sample_idx]
                        pre_n = min(50, X_sub.shape[1])
                        pre_pca = skPCA(n_components=pre_n, random_state=42)
                        X_pre = pre_pca.fit_transform(X_sub)
                        eff_perplexity = min(perplexity, max(5, len(sample_idx) // 4))
                        tsne  = TSNE(n_components=proj_dims, perplexity=eff_perplexity,
                                     random_state=42, n_iter=300)
                        coords = tsne.fit_transform(X_pre)

                if proj_method == "t-SNE":
                    plot_idx = sample_idx
                else:
                    plot_idx = np.arange(n_samples)

                src_vals  = corpus_df["source"].iloc[plot_idx].values if len(corpus_df) >= len(plot_idx) else ["unknown"] * len(plot_idx)
                text_vals = text_series.iloc[plot_idx].astype(str).str[:80].values if len(text_series) >= len(plot_idx) else [""] * len(plot_idx)

                if proj_dims == 2:
                    plot_df = pd.DataFrame({
                        "x": coords[:, 0], "y": coords[:, 1],
                        "source": src_vals, "text_preview": text_vals,
                    })
                else:
                    plot_df = pd.DataFrame({
                        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
                        "source": src_vals, "text_preview": text_vals,
                    })

                st.session_state.nlp_vec_reduced = plot_df
                st.session_state["_proj_fig_meta"] = {"method": proj_method, "dims": proj_dims}
                if proj_method == "PCA":
                    reducer_pca = skPCA(n_components=proj_dims, random_state=42)
                    reducer_pca.fit_transform(X)
                    st.session_state["_proj_exp_var"] = reducer_pca.explained_variance_ratio_.tolist()
                else:
                    st.session_state["_proj_exp_var"] = None
                set_apply_status("status_proj", True, f"{proj_method} {proj_dims}D on {n_samples:,} docs")
            except Exception as e:
                set_apply_status("status_proj", False, str(e))
            st.rerun()

        plot_df = st.session_state.nlp_vec_reduced
        meta = st.session_state.get("_proj_fig_meta")
        if plot_df is not None and meta:
            if meta["dims"] == 2:
                fig_proj = px.scatter(
                    plot_df, x="x", y="y", color="source",
                    hover_data={"text_preview": True}, template=pt(),
                    title=f"{meta['method']} 2D Projection", color_discrete_sequence=COLOR_SEQ,
                )
            else:
                fig_proj = px.scatter_3d(
                    plot_df, x="x", y="y", z="z", color="source",
                    hover_data={"text_preview": True}, template=pt(),
                    title=f"{meta['method']} 3D Projection", color_discrete_sequence=COLOR_SEQ,
                )
            st.plotly_chart(fig_proj, use_container_width=True)

            exp_var = st.session_state.get("_proj_exp_var")
            if meta["method"] == "PCA" and exp_var:
                ev1, ev2 = st.columns(2)
                ev1.metric("Variance explained (PC1)", f"{exp_var[0]*100:.1f}%")
                ev2.metric("Variance explained (PC2)", f"{exp_var[1]*100:.1f}%")


# ══════════════════════════════════════════════════════════════
# TAB 5 — Text Classification
# ══════════════════════════════════════════════════════════════
def tab_classification():
    st.header("🎯 Text Classification")

    if st.session_state.nlp_vectorized is None:
        st.warning("Run **Vectorization** first — the vectorized matrix is required as input.")
        return

    corpus_df   = st.session_state.nlp_raw_corpus
    text_series, _ = get_active_text_series()

    # ── Label column selection ──
    st.subheader("1. Select Label Column")
    label_col_default = st.session_state.get("nlp_label_col", None)

    if corpus_df is not None and "source" in corpus_df.columns:
        available_label_cols = ["source"] + [
            c for c in corpus_df.columns if c not in ("doc_id", "text", "source")
        ]
    else:
        available_label_cols = []

    # Also check if user provided labels via Corpus Manager
    if label_col_default and label_col_default not in available_label_cols:
        st.info(
            f"Label column **{label_col_default}** was selected in Corpus Manager. "
            "Re-upload or use 'source' as a proxy label."
        )

    if not available_label_cols:
        st.info("No label columns available. Use structured uploads (CSV/JSON) and select a label column in the Corpus Manager.")
        return

    label_col = st.selectbox(
        "Label column", available_label_cols,
        index=0 if label_col_default not in available_label_cols else available_label_cols.index(label_col_default),
        key="clf_label_col",
    )
    y_raw = corpus_df[label_col] if label_col in corpus_df.columns else corpus_df["source"]

    # ── Model selection ──
    st.subheader("2. Choose Model")
    clf_name = st.selectbox(
        "Algorithm",
        ["Multinomial Naïve Bayes", "Linear SVC", "Logistic Regression", "Random Forest"],
        key="clf_model_sel",
    )
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, step=0.05, key="clf_test_size")

    vec_type = st.session_state.nlp_vectorizer_type
    X_active = st.session_state.nlp_vectorized
    matrix_nonneg = is_nonnegative(X_active)

    if clf_name == "Multinomial Naïve Bayes" and not matrix_nonneg:
        reason = ("Sentence Transformer embeddings" if vec_type == "sentence_transformer"
                  else "PCA-reduced features" if vec_type == "pca_reduced"
                  else "the active matrix")
        st.warning(
            f"Multinomial Naïve Bayes requires non-negative input, but {reason} "
            "contain negative values. Use TF-IDF / Count vectorizer (un-reduced) "
            "or choose a different classifier."
        )
        return

    if stateful_apply_button("🚀 Train Classifier", "status_train_clf"):
        try:
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.svm import LinearSVC
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier

            X = st.session_state.nlp_vectorized
            try:
                import scipy.sparse as sp
                if sp.issparse(X):
                    X = X.toarray()
            except ImportError:
                pass

            le = LabelEncoder()
            y  = le.fit_transform(y_raw.fillna("unknown").astype(str))

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
                if pd.Series(y).value_counts().min() >= 2 else None
            )

            clf_map = {
                "Multinomial Naïve Bayes": MultinomialNB(),
                "Linear SVC":             LinearSVC(max_iter=2000, random_state=42),
                "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
            }
            clf = clf_map[clf_name]

            with st.spinner(f"Training {clf_name}…"):
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)

            acc    = accuracy_score(y_te, y_pred)
            cm     = confusion_matrix(y_te, y_pred)
            report = classification_report(y_te, y_pred,
                                           target_names=le.classes_, output_dict=True)

            st.session_state.nlp_trained_clf   = clf
            st.session_state.nlp_clf_features  = st.session_state.nlp_vectorizer_obj
            st.session_state.nlp_clf_name      = clf_name
            st.session_state.nlp_label_encoder = le
            st.session_state.nlp_clf_pca_transformer = (
                st.session_state.nlp_pca_transformer if vec_type == "pca_reduced" else None
            )
            st.session_state.nlp_clf_vectorizer_type = (
                st.session_state.nlp_vectorizer_type_pre_pca if vec_type == "pca_reduced" else vec_type
            )
            st.session_state.nlp_clf_results   = {
                "acc": acc, "cm": cm, "report": report,
                "y_test": y_te, "y_pred": y_pred,
                "classes": le.classes_,
            }
            set_apply_status("status_train_clf", True, f"accuracy {acc:.4f}")
        except Exception as e:
            set_apply_status("status_train_clf", False, str(e))
        st.rerun()

    # ── Results ──
    res = st.session_state.nlp_clf_results
    if res:
        st.subheader("3. Evaluation Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",  f"{res['acc']:.4f}")
        m2.metric("Classifier", st.session_state.nlp_clf_name)
        m3.metric("Classes",   len(res["classes"]))

        # Confusion matrix
        cm_df = pd.DataFrame(
            res["cm"], index=res["classes"], columns=res["classes"]
        )
        fig_cm = px.imshow(
            cm_df, text_auto=True, template=pt(),
            title="Confusion Matrix",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        report_df = pd.DataFrame(res["report"]).T.round(4)
        with st.expander("📋 Classification Report", expanded=True):
            st.dataframe(report_df, use_container_width=True)
            download_csv_button(report_df.reset_index(), "classification_report.csv")

        # Bulk inference
        st.subheader("4. Predict New Text")
        infer_tabs = st.tabs(["✏️ Manual Entry", "📂 Upload CSV"])

        with infer_tabs[0]:
            manual_text = st.text_area("Enter text to classify", key="clf_manual_input",
                                       height=120)
            if st.button("🔮 Predict", key="btn_clf_manual_predict"):
                clf   = st.session_state.nlp_trained_clf
                vect  = st.session_state.nlp_clf_features
                le    = st.session_state.nlp_label_encoder
                vec_t = st.session_state.nlp_clf_vectorizer_type
                pca_t = st.session_state.nlp_clf_pca_transformer
                if manual_text.strip():
                    try:
                        if vec_t in ("tfidf", "count"):
                            X_new = vect.transform([manual_text])
                            try:
                                import scipy.sparse as sp
                                if sp.issparse(X_new): X_new = X_new.toarray()
                            except ImportError: pass
                        else:
                            X_new = vect.encode([manual_text])
                        if pca_t is not None:
                            X_new = pca_t.transform(X_new)
                        pred = clf.predict(X_new)
                        st.markdown(f"""
                        <div class='stat-card stat-after'>
                            Predicted label: <b>{le.inverse_transform(pred)[0]}</b>
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        with infer_tabs[1]:
            bulk_file = st.file_uploader("Upload CSV with text column", type=["csv"],
                                         key="clf_bulk_upload")
            if bulk_file:
                bulk_df = pd.read_csv(bulk_file)
                bulk_text_col = st.selectbox("Text column", bulk_df.columns.tolist(),
                                             key="clf_bulk_col")
                if st.button("🔮 Predict Batch", key="btn_clf_bulk"):
                    try:
                        clf   = st.session_state.nlp_trained_clf
                        vect  = st.session_state.nlp_clf_features
                        le    = st.session_state.nlp_label_encoder
                        vec_t = st.session_state.nlp_clf_vectorizer_type
                        pca_t = st.session_state.nlp_clf_pca_transformer
                        docs  = bulk_df[bulk_text_col].fillna("").astype(str).tolist()
                        if vec_t in ("tfidf", "count"):
                            X_bulk = vect.transform(docs)
                            try:
                                import scipy.sparse as sp
                                if sp.issparse(X_bulk): X_bulk = X_bulk.toarray()
                            except ImportError: pass
                        else:
                            X_bulk = vect.encode(docs, show_progress_bar=False)
                        if pca_t is not None:
                            X_bulk = pca_t.transform(X_bulk)
                        preds = clf.predict(X_bulk)
                        bulk_df["predicted_label"] = le.inverse_transform(preds)
                        st.dataframe(bulk_df.head(50), use_container_width=True)
                        download_csv_button(bulk_df, "bulk_predictions.csv")
                        save_result_widget(bulk_df, "bulk_predictions", "clf_bulk_save")
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 6 — Topic Modeling & Clustering
# ══════════════════════════════════════════════════════════════
def tab_topic_modeling():
    st.header("🗂 Topic Modeling & Clustering")

    if st.session_state.nlp_vectorized is None:
        st.warning("Run **Vectorization** first.")
        return

    topic_tabs = st.tabs(["📌 Topic Modeling (LDA / NMF)", "🔵 K-Means Text Clustering"])

    # ── Topic Modeling ──
    with topic_tabs[0]:
        st.subheader("Latent Dirichlet Allocation / Non-Negative Matrix Factorisation")

        vec_type = st.session_state.nlp_vectorizer_type
        X        = st.session_state.nlp_vectorized

        if vec_type not in ("tfidf", "count"):
            reason = ("Sentence Transformer embeddings" if vec_type == "sentence_transformer"
                      else "a PCA-reduced matrix" if vec_type == "pca_reduced"
                      else "the active matrix")
            st.info(
                f"Topic Modeling (LDA/NMF) needs an interpretable **word vocabulary** to "
                f"label topics by their top keywords, and NMF additionally requires "
                f"**non-negative** values. The active matrix right now is {reason}, which "
                "has neither a fixed vocabulary nor a non-negativity guarantee — this is "
                "exactly what was causing the `Negative values in data passed to NMF "
                "initialization` crash.\n\n"
                "**To fix:** open the **Vectorization** tab and (re-)run **TF-IDF** or "
                "**Count** vectorization — and skip PCA Reduction on top of it if you plan "
                "to use Topic Modeling.\n\n"
                "For topic-like grouping on embeddings, use the **K-Means Text Clustering** "
                "tab instead — it works on any numeric matrix, embeddings included."
            )
        else:
            try:
                import scipy.sparse as sp
                if sp.issparse(X): X_dense = X.toarray()
                else:              X_dense = X
            except ImportError:
                X_dense = X

            tm_method = st.selectbox(
                "Method", ["LDA (requires Count vectorizer)", "NMF (works with TF-IDF)"],
                key="tm_method",
            )

            lda_blocked = "LDA" in tm_method and vec_type != "count"
            if lda_blocked:
                st.warning(
                    "LDA requires **Count** (Bag of Words) vectorizer output. "
                    "Switch to Count in the Vectorization tab or use NMF instead."
                )

            n_topics = st.slider("Number of topics", 2, 20, 5, key="tm_n_topics")
            top_words = st.slider("Top words per topic", 5, 20, 10, key="tm_top_words")

            if stateful_apply_button("🚀 Run Topic Model", "status_run_tm"):
                if lda_blocked:
                    set_apply_status(
                        "status_run_tm", False,
                        "LDA requires Count vectorizer output — switch to Count in "
                        "Vectorization, or choose NMF instead."
                    )
                else:
                    try:
                        if not is_nonnegative(X_dense):
                            raise ValueError(
                                "The vectorized matrix contains negative values, which NMF/LDA "
                                "cannot factorise. This shouldn't happen with TF-IDF/Count — "
                                "try re-running Vectorization on the Vectorization tab."
                            )
                        if "LDA" in tm_method:
                            from sklearn.decomposition import LatentDirichletAllocation
                            model = LatentDirichletAllocation(
                                n_components=n_topics, random_state=42, max_iter=20
                            )
                            with st.spinner(f"Fitting LDA ({n_topics} topics)…"):
                                doc_topic = model.fit_transform(X_dense)
                            tm_type = "LDA"
                        else:
                            from sklearn.decomposition import NMF
                            model = NMF(n_components=n_topics, random_state=42, max_iter=500)
                            with st.spinner(f"Fitting NMF ({n_topics} topics)…"):
                                doc_topic = model.fit_transform(X_dense)
                            tm_type = "NMF"

                        feature_names = st.session_state.nlp_feature_names
                        if feature_names is None:
                            raise RuntimeError(
                                "Feature names unavailable — Topic Modeling requires TF-IDF or "
                                "Count vectorizer output."
                            )

                        topics = []
                        for i, comp in enumerate(model.components_):
                            top_idx  = comp.argsort()[::-1][:top_words]
                            keywords = [feature_names[j] for j in top_idx]
                            topics.append({"topic": i, "keywords": keywords, "weights": comp[top_idx]})

                        st.session_state.nlp_topic_model   = model
                        st.session_state.nlp_topic_type    = tm_type
                        st.session_state.nlp_topic_results = {
                            "topics": topics, "doc_topic": doc_topic,
                            "n_topics": n_topics, "tm_type": tm_type,
                        }
                        set_apply_status("status_run_tm", True, f"{tm_type}, {n_topics} topics")
                    except Exception as e:
                        set_apply_status("status_run_tm", False, str(e))
                st.rerun()

            res = st.session_state.nlp_topic_results
            if res:
                topics   = res["topics"]
                doc_topic = res["doc_topic"]

                # Topic keyword cards
                st.subheader("Topic Keywords")
                cols = st.columns(min(res["n_topics"], 3))
                for i, t in enumerate(topics):
                    with cols[i % min(res["n_topics"], 3)]:
                        keywords_str = " · ".join(t["keywords"])
                        st.markdown(f"""
                        <div class='stat-card stat-info'>
                            <b>Topic {t['topic']}</b><br>
                            <span style='font-size:12px'>{keywords_str}</span>
                        </div>""", unsafe_allow_html=True)

                # Document-topic distribution heatmap (sample)
                st.subheader("Document–Topic Distribution (first 50 docs)")
                sample_dt = pd.DataFrame(
                    doc_topic[:50],
                    columns=[f"Topic {i}" for i in range(res["n_topics"])]
                )
                fig_dt = px.imshow(
                    sample_dt, template=pt(),
                    title="Document–Topic Weight Heatmap",
                    color_continuous_scale="Blues",
                    labels={"x": "Topic", "y": "Document"},
                )
                st.plotly_chart(fig_dt, use_container_width=True)

                # Per-topic dominant count
                dominant = np.argmax(doc_topic, axis=1)
                dom_counts = pd.Series(dominant).value_counts().sort_index().reset_index()
                dom_counts.columns = ["topic", "count"]
                dom_counts["topic"] = dom_counts["topic"].apply(lambda x: f"Topic {x}")
                fig_dom = px.bar(
                    dom_counts, x="topic", y="count", template=pt(),
                    title="Documents per Dominant Topic",
                    color_discrete_sequence=COLOR_SEQ,
                )
                st.plotly_chart(fig_dom, use_container_width=True)

                # Topic breakdown table
                corpus_df = st.session_state.nlp_raw_corpus
                text_series, _ = get_active_text_series()
                result_df = corpus_df.copy()
                result_df["dominant_topic"] = dominant
                result_df["topic_weight"]   = doc_topic.max(axis=1).round(4)
                for i in range(res["n_topics"]):
                    result_df[f"topic_{i}_weight"] = doc_topic[:, i].round(4)

                with st.expander("📋 Document–Topic Assignments"):
                    st.dataframe(result_df[["doc_id", "source", "text", "dominant_topic", "topic_weight"]].head(100),
                                 use_container_width=True)
                download_csv_button(result_df, "topic_assignments.csv", "⬇ Download Topic Assignments")
                save_result_widget(result_df, "topic_assignments", "tm_save")


    # ── K-Means Text Clustering ──
    with topic_tabs[1]:
        st.subheader("K-Means Clustering on Text Embeddings")

        X = st.session_state.nlp_vectorized
        try:
            import scipy.sparse as sp
            if sp.issparse(X): X = X.toarray()
        except ImportError:
            pass

        k_max = min(15, X.shape[0] - 1)
        if k_max < 2:
            st.error("Not enough documents for clustering.")
            return

        # Elbow
        if stateful_apply_button("📈 Compute Elbow Curve", "status_elbow"):
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                with st.spinner("Computing elbow…"):
                    k_range = range(2, min(k_max + 1, 12))
                    inertias, sils = [], []
                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        lb = km.fit_predict(X)
                        inertias.append(km.inertia_)
                        sils.append(silhouette_score(X, lb, sample_size=min(1000, X.shape[0])))
                st.session_state["_elbow_data"] = {
                    "k_range": list(k_range), "inertias": inertias, "sils": sils,
                }
                set_apply_status("status_elbow", True, f"k = 2…{min(k_max, 11)}")
            except Exception as e:
                set_apply_status("status_elbow", False, str(e))
            st.rerun()

        elbow_data = st.session_state.get("_elbow_data")
        if elbow_data:
            ec1, ec2 = st.columns(2)
            with ec1:
                fig_elb = px.line(x=elbow_data["k_range"], y=elbow_data["inertias"], markers=True,
                                  template=pt(), title="Elbow Method",
                                  labels={"x": "k", "y": "Inertia"},
                                  color_discrete_sequence=[COLOR_SEQ[0]])
                st.plotly_chart(fig_elb, use_container_width=True)
            with ec2:
                fig_sil = px.line(x=elbow_data["k_range"], y=elbow_data["sils"], markers=True,
                                  template=pt(), title="Silhouette Score",
                                  labels={"x": "k", "y": "Silhouette"},
                                  color_discrete_sequence=[COLOR_SEQ[2]])
                st.plotly_chart(fig_sil, use_container_width=True)

        k_val = st.slider("Number of clusters (k)", 2, k_max, min(5, k_max),
                          key="clust_k_val")

        if stateful_apply_button("🚀 Run K-Means", "status_run_kmeans"):
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                with st.spinner(f"Clustering into {k_val} clusters…"):
                    km     = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                    labels = km.fit_predict(X)
                    sil    = silhouette_score(X, labels, sample_size=min(1000, X.shape[0]))

                corpus_df    = st.session_state.nlp_raw_corpus
                text_series, _ = get_active_text_series()
                result_df    = corpus_df.copy()
                result_df["cluster"] = labels.astype(str)

                # Representative docs (closest to centroid)
                reps = []
                for c in range(k_val):
                    idx_c  = np.where(labels == c)[0]
                    center = km.cluster_centers_[c]
                    dists  = np.linalg.norm(X[idx_c] - center, axis=1)
                    best   = idx_c[np.argmin(dists)]
                    reps.append({
                        "cluster": str(c),
                        "n_docs": len(idx_c),
                        "representative_text": str(text_series.iloc[best])[:200],
                    })

                st.session_state.nlp_cluster_results = {
                    "model": km, "labels": labels, "sil": sil,
                    "result_df": result_df, "reps": reps, "k": k_val,
                }
                set_apply_status("status_run_kmeans", True, f"k={k_val}, silhouette {sil:.4f}")
            except Exception as e:
                set_apply_status("status_run_kmeans", False, str(e))
            st.rerun()

        cr = st.session_state.nlp_cluster_results
        if cr:
            st.metric("Silhouette Score", f"{cr['sil']:.4f}")

            # Cluster distribution
            clust_counts = pd.Series(cr["labels"].astype(str)).value_counts().sort_index().reset_index()
            clust_counts.columns = ["cluster", "count"]
            clust_counts["cluster"] = clust_counts["cluster"].apply(lambda x: f"Cluster {x}")
            fig_cd = px.bar(
                clust_counts, x="cluster", y="count", template=pt(),
                title="Documents per Cluster",
                color_discrete_sequence=COLOR_SEQ,
            )
            st.plotly_chart(fig_cd, use_container_width=True)

            # Representative texts
            st.subheader("Representative Documents per Cluster")
            for r in cr["reps"]:
                st.markdown(f"""
                <div class='stat-card stat-info'>
                    <b>Cluster {r['cluster']}</b> — {r['n_docs']:,} docs<br>
                    <span style='font-size:12px;color:#6b7280'>{r['representative_text']}…</span>
                </div>""", unsafe_allow_html=True)

            # Use projection if available
            proj = st.session_state.nlp_vec_reduced
            if proj is not None and len(proj) == len(cr["labels"]):
                proj_plot = proj.copy()
                proj_plot["cluster"] = cr["labels"].astype(str)
                if "z" in proj_plot.columns:
                    fig_cproj = px.scatter_3d(
                        proj_plot, x="x", y="y", z="z", color="cluster",
                        template=pt(), title="Clusters in 3D Projection",
                        color_discrete_sequence=COLOR_SEQ,
                    )
                else:
                    fig_cproj = px.scatter(
                        proj_plot, x="x", y="y", color="cluster",
                        template=pt(), title="Clusters in 2D Projection",
                        color_discrete_sequence=COLOR_SEQ,
                    )
                st.plotly_chart(fig_cproj, use_container_width=True)

            with st.expander("📋 Cluster Assignments"):
                st.dataframe(cr["result_df"][["doc_id", "source", "text", "cluster"]].head(100),
                             use_container_width=True)
            download_csv_button(cr["result_df"], "cluster_assignments.csv",
                                "⬇ Download Cluster Assignments")
            save_result_widget(cr["result_df"], "cluster_assignments", "clust_save")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    st.markdown("<div class='dashboard-title'>NLP Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dashboard-subtitle'>"
        "Text &nbsp;·&nbsp; NLP &nbsp;·&nbsp; Sentiment"
        "&nbsp;&nbsp;|&nbsp;&nbsp;"
        "Corpus &nbsp;·&nbsp; Clean &nbsp;·&nbsp; Explore &nbsp;·&nbsp; Model"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tabs = st.tabs([
        "📚 Corpus Manager",
        "🧹 Text Cleaning",
        "📊 Text EDA",
        "🔢 Vectorization",
        "🎯 Text Classification",
        "🗂 Topic Modeling",
    ])

    with tabs[0]: tab_corpus_manager()
    with tabs[1]: tab_text_cleaning()
    with tabs[2]: tab_text_eda()
    with tabs[3]: tab_vectorization()
    with tabs[4]: tab_classification()
    with tabs[5]: tab_topic_modeling()


if __name__ == "__main__":
    main()
