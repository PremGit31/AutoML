"""
NLP Dashboard — nlp_app3.py
============================
Standalone text / NLP analysis companion to the Universal ML Dashboard.

Tabs:
    1. Corpus Manager — upload .txt/.log/.csv/.json/.xlsx/.pdf/.docx/.pptx, append files,
       build master corpus, persistent file log
    2. Text Cleaning — sequential spaCy-driven pipeline with per-step expanders, live banner,
       and persistent Applied/Unsuccessful status badges
    3. Text EDA — length distributions, n-gram charts, word clouds, KWIC search
    4. Vectorization — TF-IDF / CountVectorizer + Sentence Transformers (GPU-aware, explicit
       device selection, FP16), PCA dimensionality reduction (any number of components, for
       downstream tasks) and separate PCA / t-SNE 2D/3D projections (for plotting only)
    5. Text Classification — Naïve Bayes, LinearSVC, Logistic Regression, Random Forest;
       confusion matrix, classification report, bulk inference (correctly PCA-aware at inference time)
    6. Topic Modeling — LDA (CountVec) / NMF (TF-IDF) + K-Means clustering on embeddings;
       guarded against negative-value inputs (embeddings/PCA-reduced)
    7. Sentiment Analysis — NEW. Rule-based sentiment labelling on top of BERT.
       * Quick Predict — pre-trained BertTokenizer + BertForSequenceClassification
         (e.g. nlptown/bert-base-multilingual-uncased-sentiment), deterministic
         star → polarity rule (1–2★ Negative, 3★ Neutral, 4–5★ Positive)
       * Fine-Tune — tokenize your labelled corpus (token IDs, attention masks, [CLS]/[SEP]),
         configure TrainingArguments (learning rate, batch size, epochs) and fine-tune BERT
         with the 🤗 Trainer on your specific sentiment categories; full evaluation
         (accuracy, weighted F1, confusion matrix, report, loss curve) + inference.

Requires:
    pip install streamlit spacy textblob wordcloud scikit-learn gensim sentence-transformers
                plotly pandas numpy torch transformers accelerate PyMuPDF python-docx python-pptx
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
# Custom CSS (mirrors app12 palette exactly)
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
}
.stApp { background-color: #f8f7f4; color: #1a1a2e; }

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e8e4de;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }

.dashboard-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.dashboard-subtitle {
    font-size: 0.92rem;
    color: #6b7280;
    margin-bottom: 20px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #ffffff; padding: 5px;
    border-radius: 12px; border: 1px solid #e8e4de;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6b7280; border-radius: 8px;
    font-size: 13px; font-weight: 500; padding: 8px 16px;
    transition: all 0.15s ease;
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
    background: #2d2d4e;
    box-shadow: 0 4px 12px rgba(26,26,46,0.25);
    transform: translateY(-1px);
}

.active-banner {
    background: linear-gradient(135deg, #ffffff, #f0eee9);
    border: 1px solid #e8e4de; border-left: 4px solid #1a1a2e;
    border-radius: 8px; padding: 12px 18px;
    font-family: 'DM Mono', monospace; font-size: 12px; color: #374151;
    margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.stat-card {
    background: #ffffff; border: 1px solid #e8e4de; border-radius: 10px;
    padding: 14px 18px; margin: 8px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    font-family: 'DM Sans', sans-serif; font-size: 14px; color: #374151;
}
.stat-after { border-left: 4px solid #10b981; }
.stat-zero  { border-left: 4px solid #6366f1; }
.stat-warn  { border-left: 4px solid #f59e0b; }
.stat-info  { border-left: 4px solid #3b82f6; }

[data-testid="stExpander"] {
    background: #ffffff; border: 1px solid #e8e4de; border-radius: 10px;
    margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
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
[data-testid="stMetricValue"]  { color: #1a1a2e !important; font-weight: 700 !important; }

hr { border-color: #e8e4de; margin: 20px 0; }

.log-entry {
    font-family: 'DM Mono', monospace; font-size: 11px; color: #374151;
    background: #f8f7f4; border: 1px solid #e8e4de; border-radius: 6px;
    padding: 5px 10px; margin: 3px 0;
}
.save-badge {
    display:inline-block; background:#10b981; color:#fff; border-radius:6px;
    padding:2px 8px; font-size:11px; font-weight:600;
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
.kwic-hit {
    color: #1a1a2e; font-weight: 700; background: #fef3c7;
    padding: 0 3px; border-radius: 3px;
}

/* ── Sentiment Analysis chips (rule-mapped polarity) ── */
.senti-chip {
    display:inline-block; border-radius:6px; padding:3px 10px;
    font-size:12px; font-weight:600; margin-top:6px; margin-bottom:2px;
}
.senti-pos  { background:#d1fae5; color:#065f46; border:1px solid #10b981; }
.senti-vpos { background:#a7f3d0; color:#064e3b; border:1px solid #059669; }
.senti-neu  { background:#f3f4f6; color:#4b5563; border:1px solid #d1cdc7; }
.senti-neg  { background:#fee2e2; color:#991b1b; border:1px solid #ef4444; }
.senti-vneg { background:#fecaca; color:#7f1d1d; border:1px solid #dc2626; }
.token-mono {
    font-family:'DM Mono', monospace; font-size:11.5px; color:#374151;
    background:#f8f7f4; border:1px solid #e8e4de; border-radius:6px;
    padding:8px 12px; display:block; overflow-x:auto; white-space:nowrap;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        # Corpus
        "nlp_corpus_files": {},         # {filename: {"text": [str], "source": str}}
        "nlp_raw_corpus": None,         # DataFrame with columns: doc_id, text, source
        "nlp_corpus_file_log": [],      # [{"filename","n_docs","appended_at"}] — running upload history
        "nlp_last_append_msg": None,
        # Cleaning
        "nlp_cleaned_corpus": None,     # DataFrame: doc_id, text, cleaned_text, source
        "nlp_cleaning_log": [],         # [step strings]
        "nlp_clean_history": [],        # [list[str]] — snapshots for undo
        # Vectorization
        "nlp_vectorized": None,         # dense np.ndarray or sparse matrix
        "nlp_vectorizer_obj": None,     # fitted sklearn vectorizer or SentenceTransformer
        "nlp_vectorizer_type": None,    # "tfidf" | "count" | "sentence_transformer" | "pca_reduced"
        "nlp_feature_names": None,      # list[str] for TF-IDF / Count
        "nlp_vec_reduced": None,        # 2D/3D projection for scatter
        # PCA dimensionality reduction (distinct from the 2D/3D projection above)
        "nlp_vectorized_pre_pca": None,      # backup of matrix/type/features before PCA reduction
        "nlp_vectorizer_type_pre_pca": None,
        "nlp_feature_names_pre_pca": None,
        "nlp_pca_info": None,           # dict: n_components, explained_variance, cumulative
        "nlp_pca_transformer": None,    # the fitted sklearn PCA object, needed to project new text at inference time
        # Classification
        "nlp_labels": None,             # Series — target column
        "nlp_label_col": None,
        "nlp_trained_clf": None,
        "nlp_clf_features": None,       # vectorizer used for the trained model
        "nlp_clf_name": None,
        "nlp_clf_results": None,        # dict with metrics
        "nlp_label_encoder": None,
        "nlp_clf_pca_transformer": None,    # PCA object used at train time, if any — needed to project new text at inference
        "nlp_clf_vectorizer_type": None,    # base vectorizer type ("tfidf"/"count"/"sentence_transformer") at train time
        # Topic Modeling / Clustering
        "nlp_topic_model": None,
        "nlp_topic_type": None,
        "nlp_topic_results": None,
        "nlp_cluster_results": None,
        # Sentiment Analysis (BERT) — NEW
        "nlp_senta_quick_results": None,    # DataFrame from pre-trained BERT quick predict
        "nlp_senta_ft_model": None,         # fine-tuned BertForSequenceClassification (in-memory)
        "nlp_senta_ft_tokenizer": None,     # matching BertTokenizer
        "nlp_senta_ft_labels": None,        # list[str] class names (id2label order)
        "nlp_senta_results": None,          # dict with fine-tune metrics / cm / report / history
        "nlp_senta_train_info": None,       # dict: checkpoint, lr, batch, epochs, sizes, secs
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
    n_docs = len(df)
    avg_w = df["text"].apply(lambda x: len(str(x).split())).mean()
    avg_c = df["text"].apply(lambda x: len(str(x))).mean()
    sources = df["source"].nunique() if "source" in df.columns else 1
    st.markdown(f"""
    <div class="active-banner">
    📚 {label} &nbsp;·&nbsp; {n_docs:,} documents &nbsp;·&nbsp;
    avg {avg_w:.0f} words / doc &nbsp;·&nbsp; avg {avg_c:.0f} chars / doc
    &nbsp;·&nbsp; {sources} source(s)
    </div>""", unsafe_allow_html=True)

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
            f"<span class='save-badge'>✓ Step {i}: {s}</span>"
            for i, s in enumerate(steps, 1)
        )
        st.markdown(f"<div class='active-banner'>🧹 Active Cleaning Pipeline &nbsp; {lines}</div>",
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
            f"<span class='gpu-badge gpu-badge-off'>💻 CPU only</span>"
            f"<span class='status-detail'>({GPU_INFO['reason'] or 'no GPU detected'})</span>",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────
# File loading
# ──────────────────────────────────────────────────────────────
import io

def load_text_file(uploaded_file):
    """Return list[str] of documents from any supported file type."""
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower()
    if ext in ("txt", "log"):
        raw = uploaded_file.read().decode("utf-8", errors="replace")
        docs = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return docs, f"{len(docs):,} lines"
    elif ext == "pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        docs = [page.get_text().strip() for page in doc if page.get_text().strip()]
        return docs, f"{len(docs):,} pages"
    elif ext == "docx":
        import docx
        doc = docx.Document(uploaded_file)
        docs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return docs, f"{len(docs):,} paragraphs"
    elif ext == "pptx":
        import pptx
        prs = pptx.Presentation(uploaded_file)
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
            records = raw[list_keys[0]] if list_keys else [raw]
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
                        "tfidf": "TF-IDF",
                        "count": "Count (BoW)",
                        "sentence_transformer": "Sentence Transformer embeddings",
                        "pca_reduced": "PCA-reduced",
                    }.get(vec_type, vec_type)
                    pca_info = st.session_state.nlp_pca_info
                    if vec_type == "pca_reduced" and pca_info:
                        base = {
                            "tfidf": "TF-IDF",
                            "count": "Count (BoW)",
                            "sentence_transformer": "Sentence Transformer",
                        }.get(st.session_state.nlp_vectorizer_type_pre_pca, "features")
                        vec_label = f"{base} → PCA ({pca_info['n_components']} dims)"
                    st.markdown(f"**Vectorized:** ✅ {vec_label}")

            # Persistent file log (running upload history)
            file_log = st.session_state.nlp_corpus_file_log
            if file_log:
                st.markdown("---")
                st.markdown("**📥 File Log**")
                for ent in reversed(file_log[-8:]):
                    st.markdown(
                        f"<div class='log-entry'>📄 {ent['filename']} · "
                        f"{ent['n_docs']:,} docs · {ent['appended_at']}</div>",
                        unsafe_allow_html=True,
                    )

        with sb_t2:
            saved_keys = [k for k in st.session_state if k.startswith("nlp_saved_")]
            if not saved_keys:
                st.info("No results saved yet.")
            else:
                for k in saved_keys:
                    name = k.replace("nlp_saved_", "")
                    df = st.session_state[k]
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
        7. ❤️ Sentiment Analysis
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

    pending = {}  # {filename: list[str]} awaiting append
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
                st.caption(f"Label column **{label_col}** will be used in Classification & Sentiment tabs.")
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

            new_df = (pd.DataFrame(new_rows, columns=["text", "source"])
                      if new_rows else pd.DataFrame(columns=["text", "source"]))
            existing = st.session_state.nlp_raw_corpus
            if existing is not None:
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined["doc_id"] = range(len(combined))
            st.session_state.nlp_raw_corpus = combined

            # Persistent file log
            for fname, docs in pending.items():
                st.session_state.nlp_corpus_file_log.append({
                    "filename": fname,
                    "n_docs": len(docs),
                    "appended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            st.session_state.nlp_last_append_msg = (
                f"Appended {len(new_rows):,} documents → corpus now {len(combined):,} docs."
            )

            # Reset downstream state when corpus changes
            st.session_state.nlp_cleaned_corpus = None
            st.session_state.nlp_cleaning_log = []
            st.session_state.nlp_clean_history = []
            st.session_state.nlp_vectorized = None
            st.session_state.nlp_vectorizer_type = None
            st.session_state.nlp_vectorized_pre_pca = None
            st.session_state.nlp_pca_info = None
            st.session_state.nlp_pca_transformer = None
            st.session_state.nlp_vec_reduced = None
            st.rerun()

    with col_b:
        if st.button("🗑 Reset Corpus", key="btn_reset_corpus"):
            for key in ["nlp_raw_corpus", "nlp_cleaned_corpus", "nlp_vectorized",
                        "nlp_vectorizer_type", "nlp_vectorizer_obj", "nlp_vec_reduced",
                        "nlp_vectorized_pre_pca", "nlp_pca_info", "nlp_pca_transformer",
                        "nlp_topic_results", "nlp_cluster_results", "nlp_clf_results",
                        "nlp_senta_quick_results", "nlp_senta_results",
                        "nlp_senta_train_info"]:
                st.session_state[key] = None
            st.session_state.nlp_cleaning_log = []
            st.session_state.nlp_clean_history = []
            st.session_state.nlp_corpus_file_log = []
            st.session_state.nlp_last_append_msg = None
            st.rerun()

    if st.session_state.nlp_last_append_msg:
        st.success(f"✅ {st.session_state.nlp_last_append_msg}")

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
        st.dataframe(corpus[["doc_id", "source", "text"]].head(100),
                     use_container_width=True)

    if corpus["source"].nunique() > 1:
        src_counts = corpus["source"].value_counts().reset_index()
        src_counts.columns = ["source", "count"]
        fig_src = px.bar(
            src_counts, x="source", y="count", template=pt(),
            title="Documents per Source",
            color_discrete_sequence=COLOR_SEQ,
        )
        st.plotly_chart(fig_src, use_container_width=True)

    # ── 4. Upload history ──
    file_log = st.session_state.nlp_corpus_file_log
    if file_log:
        st.subheader("4. Upload History")
        st.dataframe(pd.DataFrame(file_log), use_container_width=True)

    download_csv_button(corpus, "master_corpus.csv", "⬇ Download Corpus as CSV")

# ══════════════════════════════════════════════════════════════
# TAB 2 — Text Cleaning
# ══════════════════════════════════════════════════════════════
def tab_text_cleaning():
    st.header("🧹 Text Cleaning")

    if st.session_state.nlp_raw_corpus is None:
        st.warning("Load a corpus first in **Corpus Manager**.")
        return

    # Active base: cleaned corpus if it exists, else raw
    if st.session_state.nlp_cleaned_corpus is not None:
        current = st.session_state.nlp_cleaned_corpus["cleaned_text"]
    else:
        current = st.session_state.nlp_raw_corpus["text"]

    st.caption(f"Working on **{len(current):,}** documents. "
               "Steps apply **sequentially** — each one transforms the output of the previous step.")
    cleaning_log_banner()

    stop_words = get_stopwords_spacy()

    CONTRACTIONS = {
        "n't": " not", "'re": " are", "'s": " is", "'ll": " will", "'ve": " have",
        "'d": " would", "'m": " am", "won't": "will not", "can't": "cannot",
        "cannot't": "cannot",
    }

    def current_texts():
        if st.session_state.nlp_cleaned_corpus is not None:
            return st.session_state.nlp_cleaned_corpus["cleaned_text"].tolist()
        return st.session_state.nlp_raw_corpus["text"].fillna("").astype(str).tolist()

    def commit_step(step_name, new_texts):
        # snapshot for undo
        st.session_state.nlp_clean_history.append(current_texts())
        base = st.session_state.nlp_raw_corpus
        df = pd.DataFrame({
            "doc_id": base["doc_id"].values,
            "source": base["source"].values,
            "text": base["text"].values,
            "cleaned_text": new_texts,
        })
        st.session_state.nlp_cleaned_corpus = df
        st.session_state.nlp_cleaning_log.append(step_name)
        # Invalidate downstream artifacts
        st.session_state.nlp_vectorized = None
        st.session_state.nlp_vectorizer_type = None
        st.session_state.nlp_vec_reduced = None
        st.session_state.nlp_vectorized_pre_pca = None
        st.session_state.nlp_pca_info = None
        st.session_state.nlp_pca_transformer = None

    def cleaning_step(label, status_key, description, fn, needs_spacy=False):
        """One expandable cleaning step with a stateful Apply button."""
        with st.expander(f"🧽 {label}", expanded=False):
            st.caption(description)
            sample = str(current_texts()[0])[:140] if current_texts() else ""
            st.markdown(f"**Sample before:** `{sample}`")
            if needs_spacy and load_spacy() is None:
                st.warning("spaCy model unavailable — run `python -m spacy download en_core_web_sm`.")
                return
            if stateful_apply_button(f"✨ {label}", status_key):
                try:
                    texts = [str(t) for t in current_texts()]
                    new_texts = fn(texts)
                    commit_step(label, new_texts)
                    set_apply_status(status_key, True, f"{len(new_texts):,} docs processed")
                except Exception as e:
                    set_apply_status(status_key, False, str(e))
                st.rerun()

    # ── Sequential pipeline steps ──
    cleaning_step(
        "Lowercase", "status_clean_lower",
        "Convert all text to lowercase — normalises case variations.",
        lambda texts: [t.lower() for t in texts],
    )
    cleaning_step(
        "Expand Contractions", "status_clean_contract",
        "Expand English contractions (don't → do not, I'll → I will, …).",
        lambda texts: [
            (lambda t: t if not t else
             __import__("functools").reduce(
                 lambda s, kv: s.replace(kv[0], kv[1]), CONTRACTIONS.items(), t))(t)
            for t in texts],
    )
    cleaning_step(
        "Remove URLs & Emails", "status_clean_urls",
        "Strip http(s) links, www links and email addresses.",
        lambda texts: [re.sub(r"https?://\S+|www\.\S+|(?:\S+@\S+\.\S+)", " ", t) for t in texts],
    )
    cleaning_step(
        "Remove HTML Tags & Mentions", "status_clean_html",
        "Remove <html> tags, @mentions and #hashtag symbols (keeps the word).",
        lambda texts: [re.sub(r"#(\w+)", r"\1", re.sub(r"@\w+", " ",
                       re.sub(r"<[^>]+>", " ", t))) for t in texts],
    )
    cleaning_step(
        "Remove Punctuation", "status_clean_punct",
        "Remove all punctuation characters, keeping alphanumeric words and spaces.",
        lambda texts: [re.sub(r"[^\w\s]", " ", t) for t in texts],
    )
    cleaning_step(
        "Remove Numbers", "status_clean_digits",
        "Remove standalone numeric tokens and digits.",
        lambda texts: [re.sub(r"\d+", " ", t) for t in texts],
    )
    cleaning_step(
        "Remove Stopwords", "status_clean_stop",
        f"Remove {len(stop_words)} common English stopwords (spaCy list).",
        lambda texts: [" ".join(w for w in t.split() if w.lower() not in stop_words)
                       for t in texts],
    )
    cleaning_step(
        "Lemmatization (spaCy)", "status_clean_lemma",
        "Reduce words to their dictionary form using spaCy batch processing (`nlp.pipe`).",
        lambda texts: [
            " ".join(tok.lemma_ for tok in doc if tok.lemma_.strip())
            for doc in load_spacy().pipe(texts, batch_size=500)
        ],
        needs_spacy=True,
    )
    cleaning_step(
        "Collapse Whitespace & Drop Short Tokens", "status_clean_ws",
        "Remove extra whitespace and tokens shorter than 3 characters.",
        lambda texts: [
            re.sub(r"\s+", " ", " ".join(w for w in t.split() if len(w) >= 3)).strip()
            for t in texts],
    )

    # ── Pipeline controls ──
    st.subheader("Pipeline Controls")
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        can_undo = bool(st.session_state.get("nlp_clean_history"))
        if st.button("↩ Undo Last Step", key="btn_clean_undo", disabled=not can_undo):
            prev = st.session_state.nlp_clean_history.pop()
            base = st.session_state.nlp_raw_corpus
            st.session_state.nlp_cleaned_corpus = pd.DataFrame({
                "doc_id": base["doc_id"].values,
                "source": base["source"].values,
                "text": base["text"].values,
                "cleaned_text": prev,
            })
            if st.session_state.nlp_cleaning_log:
                st.session_state.nlp_cleaning_log.pop()
            st.session_state.nlp_vectorized = None
            st.session_state.nlp_vectorizer_type = None
            st.rerun()
    with cc2:
        if st.button("🗑 Reset Cleaning", key="btn_clean_reset"):
            st.session_state.nlp_cleaned_corpus = None
            st.session_state.nlp_cleaning_log = []
            st.session_state.nlp_clean_history = []
            st.session_state.nlp_vectorized = None
            st.session_state.nlp_vectorizer_type = None
            st.session_state.nlp_vec_reduced = None
            st.session_state.nlp_vectorized_pre_pca = None
            st.session_state.nlp_pca_info = None
            st.session_state.nlp_pca_transformer = None
            st.rerun()

    # ── Preview ──
    cleaned = st.session_state.nlp_cleaned_corpus
    if cleaned is not None:
        st.subheader("Cleaning Preview")
        orig_len = cleaned["text"].astype(str).str.len().mean()
        new_len = cleaned["cleaned_text"].astype(str).str.len().mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Steps Applied", len(st.session_state.nlp_cleaning_log))
        m2.metric("Avg Chars (before)", f"{orig_len:.0f}")
        m3.metric("Avg Chars (after)", f"{new_len:.0f}",
                  delta=f"{(new_len - orig_len) / max(orig_len, 1) * 100:+.1f}%")
        preview = cleaned[["doc_id", "source", "text", "cleaned_text"]].head(20).copy()
        preview.columns = ["doc_id", "source", "before", "after"]
        with st.expander("📋 Before / After (first 20 docs)", expanded=True):
            st.dataframe(preview, use_container_width=True)
        download_csv_button(cleaned[["doc_id", "source", "text", "cleaned_text"]],
                            "cleaned_corpus.csv", "⬇ Download Cleaned Corpus")

# ══════════════════════════════════════════════════════════════
# TAB 3 — Text EDA
# ══════════════════════════════════════════════════════════════
def tab_text_eda():
    st.header("📊 Text EDA")

    text_series, text_label = get_active_text_series()
    if text_series is None:
        st.warning("Load a corpus first in **Corpus Manager**.")
        return

    st.caption(f"Analysing **{text_label}** — {len(text_series):,} documents")
    cleaning_log_banner()

    eda_tabs = st.tabs(["📏 Document Lengths", "🔤 N-grams", "☁️ Word Cloud", "🔍 KWIC Search"])

    # ── Lengths ──
    with eda_tabs[0]:
        word_counts = text_series.astype(str).apply(lambda x: len(x.split()))
        char_counts = text_series.astype(str).apply(len)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median Words / Doc", f"{word_counts.median():.0f}")
        m2.metric("Max Words", f"{word_counts.max():,}")
        m3.metric("Median Chars / Doc", f"{char_counts.median():.0f}")
        m4.metric("Vocabulary (approx.)",
                  f"{len(set(' '.join(text_series.astype(str)).lower().split())):,}")

        metric_choice = st.radio("Metric", ["Words", "Characters"],
                                 horizontal=True, key="eda_len_metric")
        series_to_plot = word_counts if metric_choice == "Words" else char_counts
        fig_len = px.histogram(
            series_to_plot, nbins=60, template=pt(),
            title=f"Document Length Distribution ({metric_choice.lower()})",
            color_discrete_sequence=[COLOR_SEQ[0]],
            labels={"value": f"{metric_choice} per document", "count": "Documents"},
        )
        st.plotly_chart(fig_len, use_container_width=True)

        fig_box = px.box(
            series_to_plot, template=pt(), points=False,
            title=f"Spread of Document Lengths ({metric_choice.lower()})",
            color_discrete_sequence=[COLOR_SEQ[3]],
            labels={"value": metric_choice},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── N-grams ──
    with eda_tabs[1]:
        n_choice = st.selectbox("N-gram", [("Unigrams", 1), ("Bigrams", 2), ("Trigrams", 3)],
                                format_func=lambda t: t[0], key="eda_ngram_n")
        top_k = st.slider("Top-k", 10, 50, 20, key="eda_topk")
        with st.spinner("Counting n-grams…"):
            ng_df = compute_ngram_freq(
                text_series.fillna("").astype(str).str.lower()
                .str.replace(r"[^\w\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True),
                n_choice[1], top_k=top_k,
            )
        if ng_df.empty:
            st.info("Not enough text to build n-grams.")
        else:
            fig_ng = px.bar(
                ng_df.sort_values("count"), x="count", y="ngram", orientation="h",
                template=pt(), title=f"Top {len(ng_df)} {n_choice[0]}",
                color_discrete_sequence=[COLOR_SEQ[2]],
            )
            fig_ng.update_layout(height=max(420, 26 * len(ng_df)))
            st.plotly_chart(fig_ng, use_container_width=True)
            download_csv_button(ng_df, f"top_{n_choice[0].lower()}.csv", "⬇ Download N-gram Counts")

    # ── Word cloud ──
    with eda_tabs[2]:
        wc_c1, wc_c2 = st.columns(2)
        max_words = wc_c1.slider("Max words", 50, 500, 200, step=50, key="wc_max_words")
        colormap = wc_c2.selectbox("Colormap",
                                   ["viridis", "magma", "cividis", "plasma", "Purples", "Blues"],
                                   key="wc_cmap")
        if stateful_apply_button("☁️ Generate Word Cloud", "status_wc"):
            try:
                from wordcloud import WordCloud
                stop_words = get_stopwords_spacy()
                text_blob = " ".join(text_series.fillna("").astype(str).tolist())
                with st.spinner("Building word cloud…"):
                    wc = WordCloud(
                        width=1600, height=800,
                        background_color="#f8f7f4",
                        colormap=colormap,
                        max_words=max_words,
                        stopwords=stop_words,
                        prefer_horizontal=0.9,
                    ).generate(text_blob)
                    st.session_state["_wc_image"] = wc.to_array()
                    st.session_state["_wc_meta"] = f"{max_words} words · colormap {colormap}"
                set_apply_status("status_wc", True, st.session_state["_wc_meta"])
            except Exception as e:
                set_apply_status("status_wc", False, str(e))
            st.rerun()

        if st.session_state.get("_wc_image") is not None:
            st.image(st.session_state["_wc_image"], use_container_width=True)

    # ── KWIC search ──
    with eda_tabs[3]:
        st.caption("**Keyword In Context** — find a term and see its surrounding words.")
        kwic_col1, kwic_col2 = st.columns([2, 1])
        query = kwic_col1.text_input("Search term", key="kwic_query",
                                     placeholder="e.g. quality, error, product")
        window = kwic_col2.slider("Context window (words)", 3, 15, 7, key="kwic_window")

        if query.strip():
            hits = []
            pattern = re.compile(rf"\b{re.escape(query.strip())}\b", re.IGNORECASE)
            for idx, doc in text_series.fillna("").astype(str).items():
                words = doc.split()
                for i, w in enumerate(words):
                    if pattern.search(w):
                        left = " ".join(words[max(0, i - window):i])
                        right = " ".join(words[i + 1:i + 1 + window])
                        hits.append({"doc_id": idx, "left": left, "hit": w, "right": right})
                        break
                if len(hits) >= 100:
                    break
            st.markdown(f"**{len(hits):,}** document(s) contain «{query.strip()}» "
                        f"(showing first {min(30, len(hits))})")
            for h in hits[:30]:
                st.markdown(
                    f"<div class='log-entry'>"
                    f"<span class='kwic-context'>{h['left']}</span> "
                    f"<span class='kwic-hit'>{h['hit']}</span> "
                    f"<span class='kwic-context'>{h['right']}</span>"
                    f" <span class='status-detail'>(doc {h['doc_id']})</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════
# TAB 4 — Vectorization
# ══════════════════════════════════════════════════════════════
def tab_vectorization():
    st.header("🔢 Vectorization")

    if st.session_state.nlp_raw_corpus is None:
        st.warning("Load a corpus first in **Corpus Manager**.")
        return

    text_series, text_label = get_active_text_series()
    st.caption(f"Vectorizing **{text_label}** — {len(text_series):,} documents")
    corpus_info_banner(st.session_state.nlp_raw_corpus, "Corpus")
    gpu_status_badge()

    vec_tabs = st.tabs([
        "🧱 TF-IDF / Count",
        "🧠 Sentence Transformers",
        "📉 PCA Reduction",
        "🗺 Projection (2D/3D)",
    ])

    # ── TF-IDF / Count ──
    with vec_tabs[0]:
        st.subheader("TF-IDF / Count Vectorizer (Classic)")
        vec_choice = st.radio("Vectorizer", ["TF-IDF", "Count"],
                              horizontal=True, key="vec_classic_type")
        vc1, vc2 = st.columns(2)
        max_feat = vc1.slider("Max features", 1000, 50000, 10000,
                              step=1000, key="vec_max_feat")
        ng = vc2.selectbox("N-gram range",
                           [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
                           index=1, key="vec_ng")

        if stateful_apply_button("🚀 Vectorize", "status_vec_classic"):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
                docs = text_series.fillna("").astype(str).tolist()
                with st.spinner(f"Vectorizing {len(docs):,} docs…"):
                    if vec_choice == "TF-IDF":
                        vec = TfidfVectorizer(max_features=max_feat,
                                              ngram_range=ng, sublinear_tf=True)
                    else:
                        vec = CountVectorizer(max_features=max_feat, ngram_range=ng)
                    X = vec.fit_transform(docs)
                    st.session_state.nlp_vectorized = X
                    st.session_state.nlp_vectorizer_obj = vec
                    st.session_state.nlp_vectorizer_type = ("tfidf"
                                                            if vec_choice == "TF-IDF" else "count")
                    st.session_state.nlp_feature_names = vec.get_feature_names_out().tolist()
                    st.session_state.nlp_vec_reduced = None
                    # clear any previous PCA reduction
                    st.session_state.nlp_vectorized_pre_pca = None
                    st.session_state.nlp_vectorizer_type_pre_pca = None
                    st.session_state.nlp_feature_names_pre_pca = None
                    st.session_state.nlp_pca_info = None
                    st.session_state.nlp_pca_transformer = None
                set_apply_status("status_vec_classic", True,
                                 f"matrix {X.shape[0]:,} × {X.shape[1]:,}")
            except Exception as e:
                set_apply_status("status_vec_classic", False, str(e))
            st.rerun()

        if (st.session_state.nlp_vectorizer_type in ("tfidf", "count")
                and st.session_state.nlp_vectorized is not None):
            X = st.session_state.nlp_vectorized
            feats = st.session_state.nlp_feature_names
            import scipy.sparse as sp
            st.markdown(
                f"**Matrix:** {X.shape[0]:,} docs × {X.shape[1]:,} features "
                f"(sparsity: {100*(1 - X.nnz/(X.shape[0]*X.shape[1])):.1f}%)"
            )
            # Top features by mean weight
            mean_w = np.asarray(X.mean(axis=0)).flatten()
            top_idx = np.argsort(mean_w)[::-1][:30]
            top_df = pd.DataFrame({"feature": [feats[i] for i in top_idx],
                                   "mean_weight": mean_w[top_idx]})
            fig_feat = px.bar(
                top_df.sort_values("mean_weight"),
                x="mean_weight", y="feature", orientation="h",
                template=pt(), title="Top 30 Features by Mean Weight",
                color_discrete_sequence=[COLOR_SEQ[0]],
            )
            fig_feat.update_layout(height=600)
            st.plotly_chart(fig_feat, use_container_width=True)

            # Downloadable dense sample
            dense_sample = pd.DataFrame(X[:500].toarray(), columns=feats)
            download_csv_button(dense_sample, "vectorized_sample.csv",
                                "⬇ Download Vectorized Matrix (first 500 docs)")

    # ── Sentence Transformers ──
    with vec_tabs[1]:
        st.subheader("Sentence Transformers (Semantic Embeddings)")
        st.caption(
            "Uses a pre-trained transformer to create dense 384-dim embeddings. "
            "GPU will be used automatically if available."
        )
        gpu_status_badge()
        st.info("ℹ️ First run downloads the model (~90 MB for MiniLM). Subsequent runs use cache.")

        dev_options = ["auto (GPU if available)", "cpu"] + (["cuda"] if GPU_INFO["available"] else [])
        device_sel = st.selectbox("Device", dev_options, key="st_device",
                                  help="Explicit device override for encoding.")
        model_name = st.selectbox(
            "Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"],
            key="st_model_name",
        )
        batch_size = st.slider("Batch size", 16, 256, 64, key="st_batch")
        fp16 = st.checkbox("FP16 (half precision — GPU only)", value=False, key="st_fp16",
                           disabled=not GPU_INFO["available"])

        if stateful_apply_button("🚀 Generate Embeddings", "status_vec_st"):
            try:
                device = None
                if device_sel == "cpu":
                    device = "cpu"
                elif device_sel == "cuda":
                    device = "cuda"
                model = load_sentence_transformer(model_name, device=device)
                if model is None:
                    raise RuntimeError("Could not load model.")
                docs = text_series.fillna("").astype(str).tolist()
                if fp16 and GPU_INFO["available"]:
                    import torch
                    model = model.half()
                with st.spinner(f"Encoding {len(docs):,} docs with {model_name}…"):
                    embeddings = model.encode(
                        docs, batch_size=batch_size, show_progress_bar=False
                    )
                st.session_state.nlp_vectorized = embeddings
                st.session_state.nlp_vectorizer_obj = model
                st.session_state.nlp_vectorizer_type = "sentence_transformer"
                st.session_state.nlp_feature_names = None
                st.session_state.nlp_vec_reduced = None
                st.session_state.nlp_vectorized_pre_pca = None
                st.session_state.nlp_vectorizer_type_pre_pca = None
                st.session_state.nlp_feature_names_pre_pca = None
                st.session_state.nlp_pca_info = None
                st.session_state.nlp_pca_transformer = None
                set_apply_status("status_vec_st", True,
                                 f"embeddings {embeddings.shape[0]:,} × {embeddings.shape[1]}")
            except Exception as e:
                set_apply_status("status_vec_st", False, str(e))
            st.rerun()

        if (st.session_state.nlp_vectorizer_type == "sentence_transformer"
                and st.session_state.nlp_vectorized is not None):
            emb = st.session_state.nlp_vectorized
            st.markdown(f"**Embeddings matrix:** {emb.shape[0]:,} docs × {emb.shape[1]} dims")

    # ── PCA Reduction ──
    with vec_tabs[2]:
        st.subheader("PCA Dimensionality Reduction")
        st.caption(
            "Reduces the **active feature matrix itself** to any number of components — "
            "Classification / Topic Modeling / Clustering will then use the reduced matrix. "
            "This is different from the 2D/3D projection, which is only used for plotting."
        )
        if st.session_state.nlp_vectorized is None:
            st.info("Run vectorization first (TF-IDF or Sentence Transformers).")
        else:
            vec_type_p = st.session_state.nlp_vectorizer_type
            X_cur = st.session_state.nlp_vectorized
            already_reduced = vec_type_p == "pca_reduced"

            if already_reduced and st.session_state.nlp_pca_info:
                info = st.session_state.nlp_pca_info
                p1, p2 = st.columns(2)
                p1.metric("Components", info["n_components"])
                p2.metric("Variance retained", f"{info['cumulative']*100:.1f}%")
                ev = np.asarray(info["explained_variance"])
                fig_ev = px.line(
                    x=np.arange(1, len(ev) + 1), y=np.cumsum(ev),
                    template=pt(), title="Cumulative Explained Variance",
                    labels={"x": "Component", "y": "Cumulative variance"},
                    color_discrete_sequence=[COLOR_SEQ[2]],
                )
                st.plotly_chart(fig_ev, use_container_width=True)

            if not already_reduced:
                max_comp = int(min(500, X_cur.shape[1] - 1, X_cur.shape[0] - 1))
                max_comp = max(2, max_comp)
                n_comp = st.slider("Number of components", 2, max_comp,
                                   min(50, max_comp), key="pca_ncomp")
                if stateful_apply_button("📉 Apply PCA Reduction", "status_pca_reduce"):
                    try:
                        from sklearn.decomposition import PCA as skPCA
                        X = st.session_state.nlp_vectorized
                        try:
                            import scipy.sparse as sp
                            if sp.issparse(X):
                                X = X.toarray()
                        except ImportError:
                            pass
                        with st.spinner(f"Fitting PCA ({n_comp} components)…"):
                            pca = skPCA(n_components=n_comp, random_state=42)
                            Xr = pca.fit_transform(X)
                        # Backup the original matrix for restore / topic modeling
                        st.session_state.nlp_vectorized_pre_pca = st.session_state.nlp_vectorized
                        st.session_state.nlp_vectorizer_type_pre_pca = st.session_state.nlp_vectorizer_type
                        st.session_state.nlp_feature_names_pre_pca = st.session_state.nlp_feature_names
                        # Replace active matrix
                        st.session_state.nlp_vectorized = Xr
                        st.session_state.nlp_vectorizer_type = "pca_reduced"
                        st.session_state.nlp_feature_names = None
                        st.session_state.nlp_pca_transformer = pca
                        ev = pca.explained_variance_ratio_
                        st.session_state.nlp_pca_info = {
                            "n_components": int(n_comp),
                            "explained_variance": ev.tolist(),
                            "cumulative": float(ev.sum()),
                        }
                        st.session_state.nlp_vec_reduced = None
                        set_apply_status("status_pca_reduce", True,
                                         f"{n_comp} components · {ev.sum()*100:.1f}% variance kept")
                    except Exception as e:
                        set_apply_status("status_pca_reduce", False, str(e))
                    st.rerun()

            if st.session_state.nlp_vectorized_pre_pca is not None:
                if st.button("↩ Restore Original Features", key="btn_pca_restore"):
                    st.session_state.nlp_vectorized = st.session_state.nlp_vectorized_pre_pca
                    st.session_state.nlp_vectorizer_type = st.session_state.nlp_vectorizer_type_pre_pca
                    st.session_state.nlp_feature_names = st.session_state.nlp_feature_names_pre_pca
                    st.session_state.nlp_vectorized_pre_pca = None
                    st.session_state.nlp_vectorizer_type_pre_pca = None
                    st.session_state.nlp_feature_names_pre_pca = None
                    st.session_state.nlp_pca_info = None
                    st.session_state.nlp_pca_transformer = None
                    st.session_state.nlp_vec_reduced = None
                    set_apply_status("status_pca_reduce", True, "original features restored")
                    st.rerun()

    # ── Projection (2D/3D, plotting only) ──
    with vec_tabs[3]:
        st.subheader("Projection for Visualisation (Plotting Only)")
        st.caption("This 2D/3D projection is only used for the scatter plot — the full feature "
                   "matrix stays untouched and downstream tasks continue to use it unreduced. "
                   "For reducing the *feature matrix* itself "
                   "(any number of components), use the **PCA Reduction** tab instead.")

        if st.session_state.nlp_vectorized is None:
            st.info("Run vectorization first (TF-IDF or Sentence Transformers).")
            return

        proj_method = st.selectbox("Method", ["PCA", "t-SNE"], key="proj_method")
        proj_dims = st.selectbox("Dimensions", [2, 3], key="proj_dims")
        proj_color = "source"
        corpus_df = st.session_state.nlp_raw_corpus

        if proj_method == "t-SNE":
            max_tsne = 5000
            n_docs = (st.session_state.nlp_vectorized.shape[0]
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
                        coords = reducer.fit_transform(X)
                    else:
                        from sklearn.manifold import TSNE
                        sample_idx = (np.random.choice(n_samples, min(max_tsne, n_samples), replace=False)
                                      if n_samples > max_tsne else np.arange(n_samples))
                        X_sub = X[sample_idx]
                        pre_n = min(50, X_sub.shape[1])
                        pre_pca = skPCA(n_components=pre_n, random_state=42)
                        X_pre = pre_pca.fit_transform(X_sub)
                        eff_perplexity = min(perplexity, max(5, len(sample_idx) // 4))
                        tsne = TSNE(n_components=proj_dims, perplexity=eff_perplexity,
                                    random_state=42, n_iter=300)
                        coords = tsne.fit_transform(X_pre)

                if proj_method == "t-SNE":
                    plot_idx = sample_idx
                else:
                    plot_idx = np.arange(n_samples)

                src_vals = (corpus_df["source"].iloc[plot_idx].values
                            if len(corpus_df) >= len(plot_idx) else ["unknown"] * len(plot_idx))
                text_vals = (text_series.iloc[plot_idx].astype(str).str[:80].values
                             if len(text_series) >= len(plot_idx) else [""] * len(plot_idx))

                if proj_dims == 2:
                    plot_df = pd.DataFrame({
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "source": src_vals,
                        "text_preview": text_vals,
                    })
                else:
                    plot_df = pd.DataFrame({
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "z": coords[:, 2],
                        "source": src_vals,
                        "text_preview": text_vals,
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
                    title=f"{meta['method']} 2D Projection",
                    color_discrete_sequence=COLOR_SEQ,
                )
            else:
                fig_proj = px.scatter_3d(
                    plot_df, x="x", y="y", z="z", color="source",
                    hover_data={"text_preview": True}, template=pt(),
                    title=f"{meta['method']} 3D Projection",
                    color_discrete_sequence=COLOR_SEQ,
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

    corpus_df = st.session_state.nlp_raw_corpus
    text_series, _ = get_active_text_series()

    # ── Label column selection ──
    st.subheader("1. Select Label Column")
    label_col_default = st.session_state.get("nlp_label_col", None)

    if corpus_df is not None and "source" in corpus_df.columns:
        available_label_cols = ["source"] + [
            c for c in corpus_df.columns
            if c not in ("doc_id", "text", "source")
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
        "Label column",
        available_label_cols,
        index=0 if label_col_default not in available_label_cols
        else available_label_cols.index(label_col_default),
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
            y = le.fit_transform(y_raw.fillna("unknown").astype(str))

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if pd.Series(y).value_counts().min() >= 2 else None
            )

            clf_map = {
                "Multinomial Naïve Bayes": MultinomialNB(),
                "Linear SVC": LinearSVC(max_iter=2000, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            }
            clf = clf_map[clf_name]

            with st.spinner(f"Training {clf_name}…"):
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            cm = confusion_matrix(y_te, y_pred)
            report = classification_report(y_te, y_pred, target_names=le.classes_,
                                           output_dict=True)

            st.session_state.nlp_trained_clf = clf
            st.session_state.nlp_clf_features = st.session_state.nlp_vectorizer_obj
            st.session_state.nlp_clf_name = clf_name
            st.session_state.nlp_label_encoder = le
            st.session_state.nlp_clf_pca_transformer = (
                st.session_state.nlp_pca_transformer if vec_type == "pca_reduced" else None
            )
            st.session_state.nlp_clf_vectorizer_type = (
                st.session_state.nlp_vectorizer_type_pre_pca
                if vec_type == "pca_reduced" else vec_type
            )
            st.session_state.nlp_clf_results = {
                "acc": acc,
                "cm": cm,
                "report": report,
                "y_test": y_te,
                "y_pred": y_pred,
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
        m1.metric("Accuracy", f"{res['acc']:.4f}")
        m2.metric("Classifier", st.session_state.nlp_clf_name)
        m3.metric("Classes", len(res["classes"]))

        # Confusion matrix
        cm_df = pd.DataFrame(
            res["cm"],
            index=res["classes"],
            columns=res["classes"]
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
            manual_text = st.text_area("Enter text to classify", key="clf_manual_input", height=120)
            if st.button("🔮 Predict", key="btn_clf_manual_predict"):
                clf = st.session_state.nlp_trained_clf
                vect = st.session_state.nlp_clf_features
                le = st.session_state.nlp_label_encoder
                vec_t = st.session_state.nlp_clf_vectorizer_type
                pca_t = st.session_state.nlp_clf_pca_transformer
                if manual_text.strip():
                    try:
                        if vec_t in ("tfidf", "count"):
                            X_new = vect.transform([manual_text])
                            try:
                                import scipy.sparse as sp
                                if sp.issparse(X_new):
                                    X_new = X_new.toarray()
                            except ImportError:
                                pass
                        else:
                            X_new = vect.encode([manual_text])
                        if pca_t is not None:
                            X_new = pca_t.transform(X_new)
                        pred = clf.predict(X_new)
                        st.markdown(f"""
                        <div class="stat-card stat-after">
                        <b>Predicted label:</b> {le.inverse_transform(pred)[0]}
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
                        clf = st.session_state.nlp_trained_clf
                        vect = st.session_state.nlp_clf_features
                        le = st.session_state.nlp_label_encoder
                        vec_t = st.session_state.nlp_clf_vectorizer_type
                        pca_t = st.session_state.nlp_clf_pca_transformer

                        docs = bulk_df[bulk_text_col].fillna("").astype(str).tolist()
                        if vec_t in ("tfidf", "count"):
                            X_bulk = vect.transform(docs)
                            try:
                                import scipy.sparse as sp
                                if sp.issparse(X_bulk):
                                    X_bulk = X_bulk.toarray()
                            except ImportError:
                                pass
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
        X = st.session_state.nlp_vectorized

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
                if sp.issparse(X):
                    X_dense = X.toarray()
                else:
                    X_dense = X
            except ImportError:
                X_dense = X

            tm_method = st.selectbox(
                "Method",
                ["LDA (requires Count vectorizer)", "NMF (works with TF-IDF)"],
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
                            top_idx = comp.argsort()[::-1][:top_words]
                            keywords = [feature_names[j] for j in top_idx]
                            topics.append({"topic": i, "keywords": keywords,
                                           "weights": comp[top_idx]})

                        st.session_state.nlp_topic_model = model
                        st.session_state.nlp_topic_type = tm_type
                        st.session_state.nlp_topic_results = {
                            "topics": topics,
                            "doc_topic": doc_topic,
                            "n_topics": n_topics,
                            "tm_type": tm_type,
                        }
                        set_apply_status("status_run_tm", True, f"{tm_type}, {n_topics} topics")
                    except Exception as e:
                        set_apply_status("status_run_tm", False, str(e))
                st.rerun()

            res = st.session_state.nlp_topic_results
            if res:
                topics = res["topics"]
                doc_topic = res["doc_topic"]

                # Topic keyword cards
                st.subheader("Topic Keywords")
                cols = st.columns(min(res["n_topics"], 3))
                for i, t in enumerate(topics):
                    with cols[i % min(res["n_topics"], 3)]:
                        keywords_str = " · ".join(t["keywords"])
                        st.markdown(f"""
                        <div class="stat-card stat-zero">
                        <b>Topic {t['topic']}</b><br>{keywords_str}
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
                result_df["topic_weight"] = doc_topic.max(axis=1).round(4)
                for i in range(res["n_topics"]):
                    result_df[f"topic_{i}_weight"] = doc_topic[:, i].round(4)

                with st.expander("📋 Document–Topic Assignments"):
                    st.dataframe(result_df[["doc_id", "source", "text",
                                            "dominant_topic", "topic_weight"]].head(100),
                                 use_container_width=True)
                download_csv_button(result_df, "topic_assignments.csv",
                                    "⬇ Download Topic Assignments")
                save_result_widget(result_df, "topic_assignments", "tm_save")

    # ── K-Means Text Clustering ──
    with topic_tabs[1]:
        st.subheader("K-Means Clustering on Text Embeddings")

        X = st.session_state.nlp_vectorized
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                X = X.toarray()
        except ImportError:
            pass

        k_max = min(15, X.shape[0] - 1)
        if k_max < 2:
            st.warning("Not enough documents to cluster.")
            return

        n_clusters = st.slider("Number of clusters (k)", 2, k_max,
                               min(5, k_max), key="km_k")

        if stateful_apply_button("🚀 Run K-Means", "status_run_km"):
            try:
                from sklearn.cluster import KMeans
                with st.spinner(f"Clustering into {n_clusters} groups…"):
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = km.fit_predict(X)

                # Representative doc per cluster (closest to centroid)
                reps = []
                for c in range(n_clusters):
                    idx_c = np.where(labels == c)[0]
                    if len(idx_c) < 1:
                        continue
                    centroid = km.cluster_centers_[c]
                    dists = np.linalg.norm(X[idx_c] - centroid, axis=1)
                    rep_doc = idx_c[np.argmin(dists)]
                    text_series_c, _ = get_active_text_series()
                    reps.append({
                        "cluster": c,
                        "n_docs": int(len(idx_c)),
                        "representative_text": str(text_series_c.iloc[rep_doc])[:300],
                    })

                corpus_df = st.session_state.nlp_raw_corpus
                result_df = corpus_df.copy()
                result_df["cluster"] = labels

                st.session_state.nlp_cluster_results = {
                    "labels": labels,
                    "reps": reps,
                    "n_clusters": n_clusters,
                    "result_df": result_df,
                }
                set_apply_status("status_run_km", True, f"k = {n_clusters}")
            except Exception as e:
                set_apply_status("status_run_km", False, str(e))
            st.rerun()

        cr = st.session_state.nlp_cluster_results
        if cr:
            # Cluster size bar
            size_df = pd.Series(cr["labels"]).value_counts().sort_index().reset_index()
            size_df.columns = ["cluster", "count"]
            size_df["cluster"] = size_df["cluster"].apply(lambda x: f"Cluster {x}")
            fig_sizes = px.bar(
                size_df, x="cluster", y="count", template=pt(),
                title="Documents per Cluster",
                color_discrete_sequence=COLOR_SEQ,
            )
            st.plotly_chart(fig_sizes, use_container_width=True)

            st.markdown("#### Representative Documents")
            for r in cr["reps"]:
                st.markdown(f"""
                <div class="stat-card stat-info">
                <b>Cluster {r['cluster']}</b> — {r['n_docs']:,} docs<br>
                <span class="kwic-context">{r['representative_text']}…</span>
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
# TAB 7 — Sentiment Analysis (BERT) — NEW
#
# Rule-based sentiment labelling built on top of BERT:
#   • Load tokenizer and model  — BertTokenizer + BertForSequenceClassification
#   • Tokenize your text        — token IDs, attention masks, [CLS]/[SEP]
#   • Set training arguments    — learning rate, batch size, epochs
#   • Train and evaluate        — 🤗 Trainer fine-tuned on your sentiment categories
# ══════════════════════════════════════════════════════════════

# Rule map: BERT star-rating logits → human sentiment polarity (deterministic rules)
STAR_SENTIMENT_RULES = {
    1: ("Very Negative", "negative", "senti-vneg"),
    2: ("Negative",      "negative", "senti-neg"),
    3: ("Neutral",       "neutral",  "senti-neu"),
    4: ("Positive",      "positive", "senti-pos"),
    5: ("Very Positive", "positive", "senti-vpos"),
}

def senti_rule_from_label(raw_label):
    """
    RULE-BASED mapping from a raw model label to (sentiment, polarity, css_class).
      - nlptown-style star labels ('1 star'…'5 stars') → star rules above
      - SST-2 style labels (LABEL_0/LABEL_1, NEGATIVE/POSITIVE/NEUTRAL) passthrough
    Returns (sentiment_str, polarity_str, css_class).
    """
    lab = str(raw_label).strip().lower()
    m = re.match(r"^([1-5])\s*star\s*s?$", lab)
    if m:
        return STAR_SENTIMENT_RULES[int(m.group(1))]
    if lab in ("negative", "label_0", "0", "neg"):
        return ("Negative", "negative", "senti-neg")
    if lab in ("neutral", "1", "neu"):
        return ("Neutral", "neutral", "senti-neu")
    if lab in ("positive", "label_2", "2", "label_1", "pos"):
        return ("Positive", "positive", "senti-pos")
    return (str(raw_label), "neutral", "senti-neu")

def sentiment_chip_html(sentiment, css):
    return f"<span class='senti-chip {css}'>{sentiment}</span>"

@st.cache_resource
def load_bert_sentiment(checkpoint="nlptown/bert-base-multilingual-uncased-sentiment"):
    """Load a pre-trained BERT tokenizer + sequence-classification head (cached)."""
    try:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        tok = BertTokenizer.from_pretrained(checkpoint)
        model = BertForSequenceClassification.from_pretrained(checkpoint)
        device = "cuda" if GPU_INFO["available"] else "cpu"
        model.to(device).eval()
        return tok, model, device
    except Exception as e:
        st.error(f"Could not load BERT sentiment model: {e}")
        return None, None, None

@st.cache_resource
def load_bert_tokenizer(checkpoint):
    """Lightweight tokenizer-only loader used for the tokenization preview."""
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(checkpoint)

def bert_predict(texts, tokenizer, model, device, max_length=256, batch_size=32):
    """Run batched BERT inference → rule-mapped sentiment rows."""
    import torch
    id2label = getattr(model.config, "id2label", None) or {}
    out = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        for row in probs:
            top = int(row.argmax())
            raw = id2label.get(top, str(top))
            sentiment, polarity, css = senti_rule_from_label(raw)
            out.append({
                "raw_label": raw,
                "sentiment": sentiment,
                "polarity": polarity,
                "css": css,
                "confidence": float(row[top]),
                "probs": {id2label.get(j, str(j)): float(row[j]) for j in range(len(row))},
            })
    return out

class _SentimentDataset:
    """Minimal torch Dataset wrapping BERT encodings + integer labels for the Trainer."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        import torch
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

def tab_sentiment_analysis():
    st.header("❤️ Sentiment Analysis")
    st.markdown(
        "Rule-based sentiment labelling built on top of **BERT** "
        "(`BertTokenizer` + `BertForSequenceClassification`). "
        "**Quick Predict** runs a pre-trained sentiment checkpoint and applies a deterministic "
        "star → polarity rule (1–2★ = Negative, 3★ = Neutral, 4–5★ = Positive). "
        "**Fine-Tune** trains BERT on your *own* labelled data with the 🤗 `Trainer`."
    )
    gpu_status_badge()

    senta_tabs = st.tabs(["⚡ Quick Predict (Pre-trained BERT)",
                          "🏋 Fine-Tune BERT (Trainer)"])

    # ═════════════════ SUB-TAB 1 · QUICK PREDICT ═════════════════
    with senta_tabs[0]:
        st.subheader("Pre-trained BERT → Rule-based Sentiment")
        st.caption(
            "Zero extra training: the checkpoint’s logits are turned into probabilities, the "
            "top star rating is selected, and a rule maps it to sentiment polarity. "
            "First run downloads the model (~700 MB). Subsequent runs are cached."
        )

        checkpoint = st.text_input(
            "HF checkpoint",
            value="nlptown/bert-base-multilingual-uncased-sentiment",
            key="senta_checkpoint",
            help="Any BertForSequenceClassification-compatible checkpoint that outputs star ratings or sentiment labels.",
        )
        max_len_q = st.slider("Max sequence length", 64, 512, 256, step=64, key="senta_maxlen_q")

        corpus_ready = (st.session_state.nlp_raw_corpus is not None
                        or st.session_state.nlp_cleaned_corpus is not None)
        src_idx = 1 if corpus_ready else 0
        src = st.radio(
            "Text source",
            ["✏️ Manual entry", "📚 Active corpus"],
            index=src_idx if corpus_ready else 0,
            horizontal=True, key="senta_src",
        )

        texts_to_score = []
        if src == "✏️ Manual entry":
            manual = st.text_area(
                "Enter text to score (one document per line)",
                value=("I absolutely loved this product, it works flawlessly!\n"
                       "Terrible experience, the support never replied.\n"
                       "It is okay for the price, nothing special."),
                height=140, key="senta_manual",
            )
            texts_to_score = [ln.strip() for ln in manual.splitlines() if ln.strip()]
        else:
            if not corpus_ready:
                st.warning("No corpus loaded — switch to manual entry or load a corpus first.")
                return
            active_series, active_label = get_active_text_series()
            n_cap = st.slider("Max documents to score", 10, 3000,
                              min(500, len(active_series)), step=10, key="senta_cap")
            texts_to_score = active_series.fillna("").astype(str).tolist()[:n_cap]
            st.caption(f"Scoring the first **{len(texts_to_score):,}** docs of the "
                       f"**{active_label}** column.")

        if stateful_apply_button("🔮 Predict Sentiment", "status_senta_quick"):
            try:
                if not texts_to_score:
                    raise ValueError("Nothing to score — provide at least one line of text.")
                tok, model, device = load_bert_sentiment(checkpoint)
                if tok is None or model is None:
                    raise RuntimeError("Model failed to load — check the checkpoint name.")
                with st.spinner(f"Running BERT inference on {len(texts_to_score):,} docs "
                                f"({device})…"):
                    preds = bert_predict(texts_to_score, tok, model, device, max_len_q)
                df_out = pd.DataFrame({
                    "text": texts_to_score,
                    "bert_label": [p["raw_label"] for p in preds],
                    "sentiment": [p["sentiment"] for p in preds],
                    "polarity": [p["polarity"] for p in preds],
                    "confidence": [round(p["confidence"], 4) for p in preds],
                })
                st.session_state.nlp_senta_quick_results = df_out
                set_apply_status("status_senta_quick", True,
                                 f"{len(df_out):,} docs scored")
            except Exception as e:
                set_apply_status("status_senta_quick", False, str(e))
            st.rerun()

        qres = st.session_state.nlp_senta_quick_results
        if qres is not None:
            pol = qres["polarity"].value_counts()
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Positive", int(pol.get("positive", 0)))
            q2.metric("Neutral", int(pol.get("neutral", 0)))
            q3.metric("Negative", int(pol.get("negative", 0)))
            q4.metric("Avg Confidence", f"{qres['confidence'].mean():.2f}")

            pie_df = pol.reset_index()
            pie_df.columns = ["polarity", "count"]
            fig_pol = px.pie(
                pie_df, names="polarity", values="count", hole=0.55, template=pt(),
                title="Polarity Distribution",
                color="polarity",
                color_discrete_map={"positive": "#10b981", "neutral": "#6b7280",
                                    "negative": "#ef4444"},
            )
            lbl_df = qres["sentiment"].value_counts().reset_index()
            lbl_df.columns = ["sentiment", "count"]
            order = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
            lbl_df["sentiment"] = pd.Categorical(lbl_df["sentiment"], order, ordered=True)
            lbl_df = lbl_df.sort_values("sentiment")
            fig_lbl = px.bar(
                lbl_df, x="sentiment", y="count", template=pt(),
                title="Rule-mapped Sentiment (from BERT scores)",
                color_discrete_sequence=[COLOR_SEQ[2]],
            )

            pc1, pc2 = st.columns(2)
            pc1.plotly_chart(fig_pol, use_container_width=True)
            pc2.plotly_chart(fig_lbl, use_container_width=True)

            with st.expander("📋 Scored Documents", expanded=True):
                st.dataframe(qres.head(100), use_container_width=True)
            download_csv_button(qres, "sentiment_predictions.csv",
                                "⬇ Download Sentiment Predictions")
            save_result_widget(qres, "sentiment_predictions", "senta_quick_save")

    # ═════════════════ SUB-TAB 2 · FINE-TUNE BERT ═════════════════
    with senta_tabs[1]:
        st.subheader("Fine-tune BERT on Your Own Sentiment Categories")

        corpus_df = st.session_state.nlp_raw_corpus
        if corpus_df is None:
            st.warning(
                "Load a **labelled** corpus first in Corpus Manager — e.g. a CSV with a "
                "text column and a sentiment label column (`positive / negative / neutral`, "
                "stars, thumbs…)."
            )
            return

        text_series, _ = get_active_text_series()

        # ── 1. Label column ──
        st.markdown("##### 1. Select Label Column")
        available_label_cols = ["source"] + [
            c for c in corpus_df.columns if c not in ("doc_id", "text", "source")
        ]
        label_col_default = st.session_state.get("nlp_label_col", None)
        label_col = st.selectbox(
            "Sentiment label column",
            available_label_cols,
            index=0 if label_col_default not in available_label_cols
            else available_label_cols.index(label_col_default),
            key="senta_label_col",
        )
        y_raw = corpus_df[label_col].fillna("unknown").astype(str)
        class_counts = y_raw.value_counts()
        n_classes = len(class_counts)
        if n_classes < 2:
            st.warning("Need at least 2 distinct sentiment categories to fine-tune.")
            return
        with st.expander(f"Class balance — {n_classes} categories", expanded=False):
            cc_df = class_counts.reset_index()
            cc_df.columns = ["label", "count"]
            fig_cc = px.bar(cc_df, x="label", y="count", template=pt(),
                            color_discrete_sequence=COLOR_SEQ)
            st.plotly_chart(fig_cc, use_container_width=True)

        # ── 2. Tokenize ──
        st.markdown("##### 2. Tokenize (BertTokenizer)")
        base_ckpt = st.selectbox(
            "Base checkpoint",
            ["bert-base-uncased", "bert-base-multilingual-uncased",
             "prajjwal1/bert-mini", "prajjwal1/bert-tiny"],
            key="senta_base_ckpt",
            help="Tiny/mini variants train much faster on CPU.",
        )
        max_len = st.slider("Max sequence length (truncation)", 32, 512, 128, step=32,
                            key="senta_max_len")

        with st.expander("🔬 Tokenization preview — first document ([CLS] / [SEP], IDs, masks)",
                         expanded=False):
            try:
                _prev_tok = load_bert_tokenizer(base_ckpt)
                _sample = str(text_series.iloc[0])
                _enc = _prev_tok(_sample, truncation=True, max_length=max_len)
                _tokens = _prev_tok.convert_ids_to_tokens(_enc["input_ids"])
                st.markdown(f"**Raw text:** `{_sample[:200]}`")
                st.caption(f"Tokens ({len(_tokens)}) — note the special [CLS] / [SEP] markers:")
                st.markdown(f"<span class='token-mono'>{_tokens[:64]}</span>",
                            unsafe_allow_html=True)
                st.caption(f"Token IDs (input_ids, first 32 of {len(_enc['input_ids'])}):")
                st.markdown(f"<span class='token-mono'>{_enc['input_ids'][:32]}</span>",
                            unsafe_allow_html=True)
                st.caption(f"Attention mask — {sum(_enc['attention_mask'])} real tokens, "
                           f"{len(_enc['attention_mask']) - sum(_enc['attention_mask'])} masked/pad positions:")
                st.markdown(f"<span class='token-mono'>{_enc['attention_mask'][:32]}</span>",
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to load tokenizer: {e}")

        # ── 3. Training arguments ──
        st.markdown("##### 3. Training Arguments")
        ta1, ta2, ta3 = st.columns(3)
        lr = ta1.selectbox("Learning rate", ["5e-5", "3e-5", "2e-5", "1e-5"],
                           index=1, key="senta_lr")
        batch_size = ta2.slider("Batch size (train/eval)", 4, 64,
                                16 if GPU_INFO["available"] else 8, step=4, key="senta_bs")
        epochs = ta3.slider("Epochs", 1, 8, 3, key="senta_epochs")
        tb1, tb2, tb3 = st.columns(3)
        test_size = tb1.slider("Validation split", 0.1, 0.4, 0.2, step=0.05,
                               key="senta_test_size")
        weight_decay = tb2.number_input("Weight decay", 0.0, 0.3, 0.01, step=0.01,
                                        key="senta_wd")
        warmup_ratio = tb3.number_input("Warmup ratio", 0.0, 0.5, 0.1, step=0.05,
                                        key="senta_warmup")
        fp16_note = "FP16 (mixed precision) will be **on** automatically on GPU." \
            if GPU_INFO["available"] else "Running on CPU — batch size and epochs are kept conservative; consider `prajjwal1/bert-tiny`."
        st.caption(fp16_note)

        # ── 4. Train & evaluate with the Trainer ──
        st.markdown("##### 4. Train and Evaluate")
        if stateful_apply_button("🚀 Fine-tune BERT", "status_senta_train"):
            try:
                import inspect
                import torch
                from transformers import (
                    BertTokenizer, BertForSequenceClassification,
                    Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding,
                )
                from sklearn.preprocessing import LabelEncoder
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

                t0 = time.time()
                docs = text_series.fillna("").astype(str).tolist()
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
                classes = [str(c) for c in le.classes_]

                strat = y if pd.Series(y).value_counts().min() >= 2 else None
                X_tr, X_te, y_tr, y_te = train_test_split(
                    docs, y, test_size=test_size, random_state=42, stratify=strat,
                )

                # Tokenize → token IDs, attention masks, [CLS]/[SEP]
                tok = BertTokenizer.from_pretrained(base_ckpt)
                with st.spinner("Tokenizing corpus…"):
                    enc_tr = tok(X_tr, truncation=True, max_length=max_len)
                    enc_te = tok(X_te, truncation=True, max_length=max_len)
                train_ds = _SentimentDataset(enc_tr, y_tr)
                test_ds = _SentimentDataset(enc_te, y_te)

                id2label = {i: c for i, c in enumerate(classes)}
                label2id = {c: i for i, c in enumerate(classes)}
                model = BertForSequenceClassification.from_pretrained(
                    base_ckpt, num_labels=len(classes),
                    id2label=id2label, label2id=label2id,
                )

                # TrainingArguments — version-compatible eval strategy kwarg
                ta_kwargs = dict(
                    output_dir="./senta_training_out",
                    learning_rate=float(lr),
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=float(epochs),
                    weight_decay=float(weight_decay),
                    warmup_ratio=float(warmup_ratio),
                    logging_steps=10,
                    save_strategy="no",
                    fp16=bool(GPU_INFO["available"]),
                    seed=42,
                    report_to=[],
                    disable_tqdm=True,
                )
                ta_params = inspect.signature(TrainingArguments.__init__).parameters
                if "eval_strategy" in ta_params:
                    ta_kwargs["eval_strategy"] = "epoch"
                elif "evaluation_strategy" in ta_params:
                    ta_kwargs["evaluation_strategy"] = "epoch"
                training_args = TrainingArguments(**ta_kwargs)

                # Streamlit progress hook
                prog = st.progress(0.0, text="Preparing…")
                logged_losses = []

                class _ProgressCB(TrainerCallback):
                    def on_step_end(self, args, state, control, **kwargs):
                        total = max(1, state.max_steps)
                        frac = min(0.99, state.global_step / total)
                        prog.progress(frac,
                                      text=f"Training — step {state.global_step}/{state.max_steps}")

                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if logs and "loss" in logs:
                            logged_losses.append(
                                {"step": state.global_step, "loss": logs["loss"],
                                 "epoch": logs.get("epoch")})

                def compute_metrics(eval_pred):
                    logits, labels_ = eval_pred
                    preds = np.argmax(logits, axis=-1)
                    return {
                        "accuracy": accuracy_score(labels_, preds),
                        "f1_weighted": f1_score(labels_, preds, average="weighted"),
                    }

                trainer_kwargs = dict(
                    model=model, args=training_args,
                    train_dataset=train_ds, eval_dataset=test_ds,
                    data_collator=DataCollatorWithPadding(tok),
                    compute_metrics=compute_metrics,
                    callbacks=[_ProgressCB()],
                )
                tr_params = inspect.signature(Trainer.__init__).parameters
                if "processing_class" in tr_params:
                    trainer_kwargs["processing_class"] = tok
                elif "tokenizer" in tr_params:
                    trainer_kwargs["tokenizer"] = tok
                trainer = Trainer(**trainer_kwargs)

                with st.spinner(f"Fine-tuning {base_ckpt} — {epochs} epoch(s), "
                                f"LR {lr}, batch {batch_size}…"):
                    trainer.train()
                    eval_out = trainer.evaluate()
                prog.progress(1.0, text="Done ✓")

                pred_out = trainer.predict(test_ds)
                y_pred = np.argmax(pred_out.predictions, axis=-1)
                acc = float(eval_out.get("eval_accuracy", accuracy_score(y_te, y_pred)))
                f1 = float(eval_out.get("eval_f1_weighted",
                                        f1_score(y_te, y_pred, average="weighted")))
                cm = confusion_matrix(y_te, y_pred)
                report = classification_report(y_te, y_pred, target_names=classes,
                                               output_dict=True)

                st.session_state.nlp_senta_ft_model = model
                st.session_state.nlp_senta_ft_tokenizer = tok
                st.session_state.nlp_senta_ft_labels = classes
                st.session_state.nlp_senta_results = {
                    "acc": acc, "f1": f1, "cm": cm, "report": report,
                    "classes": classes,
                    "log_history": [h for h in trainer.state.log_history
                                    if "loss" in h or "eval_loss" in h],
                    "train_losses": logged_losses,
                }
                st.session_state.nlp_senta_train_info = {
                    "checkpoint": base_ckpt, "lr": lr, "batch": batch_size,
                    "epochs": epochs, "n_train": len(X_tr), "n_val": len(X_te),
                    "secs": round(time.time() - t0, 1),
                }
                set_apply_status("status_senta_train", True,
                                 f"accuracy {acc:.4f} · F1 {f1:.4f}")
            except Exception as e:
                set_apply_status("status_senta_train", False, str(e))
            st.rerun()

        # ── Results ──
        sres = st.session_state.nlp_senta_results
        if sres:
            info = st.session_state.nlp_senta_train_info or {}
            st.markdown("##### 5. Evaluation Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{sres['acc']:.4f}")
            m2.metric("Weighted F1", f"{sres['f1']:.4f}")
            m3.metric("Classes", len(sres["classes"]))
            m4.metric("Trained", f"{info.get('secs', '?')}s · {info.get('checkpoint', '')}")

            # Loss curve
            hist = sres.get("log_history", [])
            train_points = [h for h in hist if "loss" in h]
            eval_points = [h for h in hist if "eval_loss" in h]
            if train_points:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=[h.get("epoch") for h in train_points],
                    y=[h["loss"] for h in train_points],
                    mode="lines+markers", name="train loss",
                    line=dict(color="#1a1a2e"),
                ))
                if eval_points:
                    fig_loss.add_trace(go.Scatter(
                        x=[h.get("epoch") for h in eval_points],
                        y=[h["eval_loss"] for h in eval_points],
                        mode="lines+markers", name="eval loss",
                        line=dict(color="#6366f1", dash="dash"),
                    ))
                fig_loss.update_layout(template=pt(), title="Training / Eval Loss",
                                       xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig_loss, use_container_width=True)

            cm_df = pd.DataFrame(sres["cm"], index=sres["classes"],
                                 columns=sres["classes"])
            fig_cm2 = px.imshow(cm_df, text_auto=True, template=pt(),
                                title="Confusion Matrix — Fine-tuned BERT",
                                color_continuous_scale="Blues")
            st.plotly_chart(fig_cm2, use_container_width=True)

            report_df = pd.DataFrame(sres["report"]).T.round(4)
            with st.expander("📋 Classification Report", expanded=True):
                st.dataframe(report_df, use_container_width=True)
            download_csv_button(report_df.reset_index(), "senta_classification_report.csv")

        # ── Inference with the fine-tuned model ──
        if st.session_state.nlp_senta_ft_model is not None:
            st.markdown("##### 6. Predict with the Fine-tuned Model")
            f_infer = st.tabs(["✏️ Manual Entry", "📂 Upload CSV"])

            with f_infer[0]:
                ft_text = st.text_area("Enter text to score", key="senta_ft_manual",
                                       height=110)
                if st.button("🔮 Predict Sentiment", key="btn_senta_ft_predict"):
                    if ft_text.strip():
                        try:
                            import torch
                            model = st.session_state.nlp_senta_ft_model
                            tok = st.session_state.nlp_senta_ft_tokenizer
                            device = "cuda" if GPU_INFO["available"] else "cpu"
                            model.to(device).eval()
                            enc = tok(ft_text, return_tensors="pt", truncation=True,
                                      max_length=max_len, padding=True)
                            enc = {k: v.to(device) for k, v in enc.items()}
                            with torch.no_grad():
                                logits = model(**enc).logits
                                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                            top = int(probs.argmax())
                            classes = st.session_state.nlp_senta_ft_labels
                            raw_label = model.config.id2label.get(top, classes[top])
                            senti, polarity, css = senti_rule_from_label(raw_label)
                            st.markdown(
                                f"**Predicted:** {raw_label} &nbsp; "
                                f"{sentiment_chip_html(senti, css)} &nbsp; "
                                f"<span class='status-detail'>confidence {probs[top]:.2%}</span>",
                                unsafe_allow_html=True,
                            )
                            prob_df = pd.DataFrame({
                                "class": classes,
                                "probability": probs.round(4),
                            }).sort_values("probability")
                            fig_pb = px.bar(prob_df, x="probability", y="class",
                                            orientation="h", template=pt(),
                                            color_discrete_sequence=[COLOR_SEQ[2]])
                            st.plotly_chart(fig_pb, use_container_width=True)
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")

            with f_infer[1]:
                ft_file = st.file_uploader("Upload CSV with a text column", type=["csv"],
                                           key="senta_ft_bulk")
                if ft_file:
                    ft_df = pd.read_csv(ft_file)
                    ft_col = st.selectbox("Text column", ft_df.columns.tolist(),
                                          key="senta_ft_col")
                    if st.button("🔮 Predict Batch", key="btn_senta_ft_bulk"):
                        try:
                            model = st.session_state.nlp_senta_ft_model
                            tok = st.session_state.nlp_senta_ft_tokenizer
                            device = "cuda" if GPU_INFO["available"] else "cpu"
                            model.to(device).eval()
                            docs_bulk = ft_df[ft_col].fillna("").astype(str).tolist()
                            with st.spinner(f"Scoring {len(docs_bulk):,} docs…"):
                                preds = bert_predict(docs_bulk, tok, model, device,
                                                     max_length=max_len)
                            ft_df["predicted_label"] = [p["raw_label"] for p in preds]
                            ft_df["sentiment"] = [p["sentiment"] for p in preds]
                            ft_df["polarity"] = [p["polarity"] for p in preds]
                            ft_df["confidence"] = [round(p["confidence"], 4) for p in preds]
                            st.dataframe(ft_df.head(50), use_container_width=True)
                            download_csv_button(ft_df, "senta_bulk_predictions.csv")
                            save_result_widget(ft_df, "senta_bulk_predictions",
                                               "senta_bulk_save")
                        except Exception as e:
                            st.error(f"Batch prediction failed: {e}")

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    st.markdown("<div class='dashboard-title'>🧠 NLP Dashboard</div>",
                unsafe_allow_html=True)
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
        "❤️ Sentiment Analysis",
    ])

    with tabs[0]:
        tab_corpus_manager()
    with tabs[1]:
        tab_text_cleaning()
    with tabs[2]:
        tab_text_eda()
    with tabs[3]:
        tab_vectorization()
    with tabs[4]:
        tab_classification()
    with tabs[5]:
        tab_topic_modeling()
    with tabs[6]:
        tab_sentiment_analysis()

if __name__ == "__main__":
    main()
