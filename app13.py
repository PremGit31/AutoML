"""
app13.py — ML and EDA Dashboard
================================
Same application as app12.py, with ONE addition requested by the guide:
the Data Manager (and every "upload future dataset" picker) now also
accepts XML files.

XML files are not inherently tabular, so a dedicated, hardened
XML → DataFrame pipeline was added. It handles every known difficulty:

  1. XML is hierarchical, not tabular
       → the XML tree is parsed into a row-based structure, nested nodes
         are flattened into column names, repeated elements become records.
  2. XML structures vary wildly (attributes, nesting, namespaces)
       → pandas.read_xml is tried first (lxml → etree parser fallback),
         then a custom ElementTree parser takes over; namespaces are
         stripped, attributes are surfaced as "tag__attr" columns.
  3. Records have missing / inconsistent fields
       → the union of all keys across records is collected, a uniform
         row dictionary is built, missing values become None / NaN.
  4. Type ambiguity (everything is text in XML)
       → post-import clean-up converts columns via pd.to_numeric(),
         boolean mapping, and guarded datetime parsing.
  5. File size / performance
       → a hard size cap, a "large file" warning, a record cap, and the
         UI only ever previews the first few rows after import.
  6. Namespace & attribute handling
       → both tag text AND attributes are extracted; attributes are named
         clearly, e.g. item__id, customer__name.
  7. UI integration
       → XML keeps the exact same load_file() contract as CSV/JSON:
         it returns (df, info_message) every time.
  8. Error handling
       → every parse step is wrapped in try/except; malformed or empty
         files raise one friendly ValueError that surfaces via st.error()
         without ever crashing the upload flow, and the custom parser is
         used whenever the default one fails.

Nothing else in the UI was changed.
"""

import io
import json
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xml.etree.ElementTree as ET

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix, silhouette_score
)


# ============================================================
# Page Config
# ============================================================
try:
    st.set_page_config(
        page_title="ML & EDA Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass


# ============================================================
# Global Styling (matches the home hub)
# ============================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background-color: #FAF6EF;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-weight: 800;
        font-size: 3rem;
        line-height: 1.15;
        color: #1A1A1A;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.05rem;
        color: #6B6B6B;
        margin-bottom: 0.25rem;
    }
    .hero-tags {
        font-size: 0.95rem;
        color: #9A9A9A;
        margin-bottom: 2.2rem;
    }
    .hero-divider {
        border: none;
        border-top: 1px solid #E5DFD3;
        margin: 1rem 0 1.4rem 0;
    }
    .df-card {
        background: #FFFFFF;
        border: 1px solid #ECE6D8;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    .df-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 1.02rem;
        color: #1A1A1A;
    }
    .df-card p {
        margin: 0;
        font-size: 0.85rem;
        color: #6B6B6B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session State
# ============================================================
def _init_state():
    defaults = {
        "dfs": {},                       # original uploads: label -> DataFrame
        "pending_excel": {},             # label -> pd.ExcelFile awaiting sheet pick
        "preprocessed_dfs": {},          # source label -> preprocessed DataFrame
        "reduced_dfs": {},               # source label -> PCA DataFrame
        "reduced_info": {},              # source label -> metadata dict
        "active_label": None,            # dataset highlighted in selectors
        "data_version": 0,               # bumped whenever the data pool changes
        "version_log": [],               # [(n, note), ...] newest last
        # regression
        "trained_reg_model": None,
        "trained_reg_features": None,
        "trained_reg_target": None,
        "trained_reg_name": None,
        "reg_train_results": None,
        # classification
        "trained_cls_model": None,
        "trained_cls_features": None,
        "trained_cls_target": None,
        "trained_cls_name": None,
        "trained_cls_label_encoder": None,
        "trained_cls_class_names": None,
        "cls_train_results": None,
        # clustering
        "trained_clust_model": None,
        "trained_clust_features": None,
        "trained_clust_scaler": None,
        "trained_clust_name": None,
        "trained_clust_k": None,
        "clust_train_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ============================================================
# Small Helpers
# ============================================================
def pt():
    """Shared plotly template for every chart."""
    return "plotly_white"


def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_cols(df):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def df_info_card(df, label):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicated Rows", f"{int(df.duplicated().sum()):,}")
    st.caption(f"Dataset: **{label}**")


def bump_version(note):
    """Data pool changed — record it so the UI can acknowledge freshness."""
    st.session_state.data_version += 1
    st.session_state.version_log.append((st.session_state.data_version, note))
    st.session_state.version_log = st.session_state.version_log[-15:]


def interactive_result_table(df, key_prefix, title, state_key):
    """Searchable, row-capped result table used inside result expanders."""
    st.markdown(f"**{title}** — {len(df):,} rows")
    search = st.text_input("Filter rows", key=f"{key_prefix}_filter",
                           placeholder="Type to filter… (matches any column)")
    out = df
    if search:
        try:
            mask = df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False, regex=False)
            ).any(axis=1)
            out = df[mask]
        except Exception:
            out = df
    st.caption(f"Showing {min(len(out), 500):,} of {len(out):,} matching rows (preview capped at 500).")
    st.dataframe(out.head(500), use_container_width=True)
    return out


def download_csv_button(df, filename):
    st.download_button(
        label="⬇️ Download as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"dl_{filename}",
    )


# ============================================================
# Dataset Registry Helpers
# ============================================================
def get_all_dataset_options():
    """One flat pool of selectable dataframes, clearly labelled.

    Originals keep their file label, preprocessed appear as
    "Preprocessed: <label>", PCA results as "PCA: <label>".
    """
    options = {}
    for label, df in st.session_state.dfs.items():
        options[label] = df
    for label, df in st.session_state.preprocessed_dfs.items():
        options[f"Preprocessed: {label}"] = df
    for label, df in st.session_state.reduced_dfs.items():
        options[f"PCA: {label}"] = df
    return options


def get_pca_input_options():
    """PCA can be run on originals and preprocessed datasets (not on
    already-reduced ones)."""
    options = {}
    for label, df in st.session_state.dfs.items():
        options[label] = df
    for label, df in st.session_state.preprocessed_dfs.items():
        options[f"Preprocessed: {label}"] = df
    return options


def smart_default_label(labels):
    """Pick the most sensible default for a dataset dropdown."""
    if not labels:
        return None
    if st.session_state.active_label in labels:
        return st.session_state.active_label
    for candidate in reversed(labels):
        return candidate          # newest label (dicts keep insertion order)
    return labels[0]


def get_model_df(tab_key):
    """Dataset picker shared by the modelling tabs.

    Returns (df, label, pool) where pool is "upload" / "preprocessed" / "pca".
    """
    options = get_all_dataset_options()
    if not options:
        return None, None, None
    labels = list(options.keys())
    default_label = smart_default_label(labels)
    idx = labels.index(default_label) if default_label in labels else 0
    label = st.selectbox("📁 Dataset to use", labels, index=idx,
                         key=f"{tab_key}_dataset_select")
    if label.startswith("PCA:"):
        pool = "pca"
    elif label.startswith("Preprocessed:"):
        pool = "preprocessed"
    else:
        pool = "upload"
    return options[label], label, pool


# ============================================================
# XML → DataFrame Pipeline
# ============================================================
# Problem 5 (size / performance): refuse absurd files, warn on large ones,
# and cap the number of parsed records so the UI stays responsive.
XML_SIZE_HARD_LIMIT_MB = 100.0
XML_SIZE_WARN_MB = 25.0
XML_MAX_RECORDS = 100_000

_XML_TRUE = {"true", "1", "yes"}
_XML_FALSE = {"false", "0", "no"}


def _xml_strip_namespace(tag):
    """Problem 2 & 6: normalise tags by stripping namespaces.

    '{http://ex.com}row' → 'row'   and   'ns:row' → 'row'
    """
    if tag is None:
        return ""
    tag = str(tag)
    if "}" in tag:
        tag = tag.rsplit("}", 1)[-1]
    if ":" in tag:
        tag = tag.rsplit(":", 1)[-1]
    return tag.strip()


def _xml_clean_text(text):
    if text is None:
        return None
    text = text.strip()
    return text if text else None


def _xml_merge(out, key, value):
    """Insert a value; when a repeated element produces the same key twice,
    join the values with '; ' instead of silently dropping data."""
    if key is None or value is None:
        return
    if key in out and out[key] is not None:
        existing = str(out[key])
        if str(value) not in existing.split("; "):
            out[key] = f"{existing}; {value}"
    else:
        out[key] = value


def _xml_flatten_node(elem, base, out):
    """Problem 1 (hierarchical → tabular) and Problem 6 (attributes).

    Recursively flatten a child element into `out`:
      - attributes of this node            →  <base>__<attr>
      - leaf text                          →  <base>
      - nested children                    →  <base>__<child>, recursively
      - repeated siblings                  →  values joined with '; '
    """
    for attr, val in elem.attrib.items():
        _xml_merge(out, f"{base}__{_xml_strip_namespace(attr)}", val)

    children = list(elem)
    text = _xml_clean_text(elem.text)

    if not children:
        if text is not None:
            _xml_merge(out, base, text)
        return

    # Element mixes its own text with children — keep the text as
    # "<base>__text" so no information is lost.
    if text is not None:
        _xml_merge(out, f"{base}__text", text)

    for child in children:
        child_base = f"{base}__{_xml_strip_namespace(child.tag)}"
        _xml_flatten_node(child, child_base, out)


def _xml_record_to_row(record_elem):
    """Flatten ONE record element into a row dict.

    Column naming (Problem 6):
      <item id="1"><name>Pen</name></item>  →  {'item__id': '1', 'name': 'Pen'}
      <customer><name>John</name></customer> (as a nested child)
                                          →  {'customer__name': 'John'}
      - the record's OWN attributes are namespaced with its tag (item__id)
      - child tags become plain columns, nested children use '__' paths
    """
    out = {}
    tag = _xml_strip_namespace(record_elem.tag) or "record"

    for attr, val in record_elem.attrib.items():
        _xml_merge(out, f"{tag}__{_xml_strip_namespace(attr)}", val)

    children = list(record_elem)
    text = _xml_clean_text(record_elem.text)

    if not children:
        # A record that is only a text leaf, e.g. <value>42</value>
        if text is not None:
            _xml_merge(out, tag, text)
        return out

    if text is not None:
        _xml_merge(out, f"{tag}__text", text)

    for child in children:
        _xml_flatten_node(child, _xml_strip_namespace(child.tag), out)

    return out


def _xml_find_records(root):
    """Problem 2: decide which elements are the records (rows).

    Returns (records, strategy_description). Handles the common shapes:
      <rows><row/><row/></rows>                 → repeated children of root
      <data><items><item/><item/></items></data> → single wrapper element
      <data><order/><order/><note/></data>      → dominant repeated tag
      <order><id/><total/></order>              → root itself is one record
    """
    children = list(root)
    if not children:
        return [root], "single root record"

    tags = [_xml_strip_namespace(c.tag) for c in children]
    counts = Counter(tags)

    if len(counts) == 1:
        only_tag = next(iter(counts))
        if counts[only_tag] >= 2:
            return children, f"repeated <{only_tag}> elements under <{_xml_strip_namespace(root.tag)}>"

    # Single wrapper element whose children repeat (…<items><item/><item/>…)
    if len(children) == 1:
        wrapper = children[0]
        grandchildren = list(wrapper)
        if grandchildren:
            gtags = Counter(_xml_strip_namespace(g.tag) for g in grandchildren)
            g_tag, g_n = gtags.most_common(1)[0]
            if g_n >= 2:
                records = [g for g in grandchildren if _xml_strip_namespace(g.tag) == g_tag]
                return records, (f"nested <{_xml_strip_namespace(wrapper.tag)}> wrapper with "
                                 f"repeated <{g_tag}> records")

    # Mixed children but one tag clearly repeats → use those as records
    top_tag, top_n = counts.most_common(1)[0]
    if top_n >= 2:
        records = [c for c, t in zip(children, tags) if t == top_tag]
        return records, f"dominant repeated <{top_tag}> blocks"

    # All children are distinct tags → the root itself is the single record
    return [root], "single root record"


def _xml_smart_convert(df):
    """Problem 4: XML is all text — infer real dtypes after parsing.

    numeric → boolean → guarded-datetime, in that priority order.
    Anything that fails stays exactly as it was (never destructive).
    """
    converted = {"numeric": [], "boolean": [], "datetime": []}
    for col in df.columns:
        s = df[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        as_str = non_null.astype(str).str.strip()

        # numeric?
        try:
            coerced = pd.to_numeric(as_str, errors="coerce")
            if coerced.notna().all():
                df[col] = pd.to_numeric(s, errors="coerce")
                converted["numeric"].append(col)
                continue
        except Exception:
            pass

        # boolean? (only when the whole column lives in the yes/no vocab
        # AND contains both kinds — avoids mangling 0/1 numeric flags,
        # which were already converted above anyway)
        try:
            low = as_str.str.lower()
            uniq = set(low.unique())
            if uniq and uniq <= (_XML_TRUE | _XML_FALSE) and (uniq & _XML_TRUE) and (uniq & _XML_FALSE):
                mapping = {**{t: True for t in _XML_TRUE}, **{f: False for f in _XML_FALSE}}
                df[col] = s.astype(str).str.strip().str.lower().map(mapping).astype("boolean")
                converted["boolean"].append(col)
                continue
        except Exception:
            pass

        # datetime? (guarded: only when the strings actually LOOK like dates)
        try:
            sample = low.head(25)
            looks_like_date = sample.str.match(
                r"^\d{4}-\d{2}-\d{2}([ t]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(z|[+-]\d{2}:?\d{2})?)?$"
            ).mean()
            if looks_like_date >= 0.8:
                dt = pd.to_datetime(s, errors="coerce")
                if dt.notna().mean() >= 0.8:
                    df[col] = dt
                    converted["datetime"].append(col)
        except Exception:
            pass

    return df, converted


def load_xml_file(uploaded_file):
    """Parse an uploaded XML file into a (DataFrame, info_message) tuple —
    the exact same contract load_file() uses for CSV / JSON / Excel.

    Strategy (Problem 2 & 8):
      1. validate / size-guard the raw bytes (Problem 5)
      2. try pandas.read_xml with the lxml parser, then the built-in etree
         parser (covers lxml not installed)
      3. fall back to the custom ElementTree parser for irregular XML
      4. build uniform rows over the union of keys (Problem 3)
      5. post-import dtype clean-up (Problem 4)
      6. anything unexpected becomes ONE friendly ValueError (Problem 8)
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    data = uploaded_file.read()
    if isinstance(data, str):
        data = data.encode("utf-8", errors="replace")

    if not data or not data.strip():
        raise ValueError("The XML file is empty — nothing to parse.")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > XML_SIZE_HARD_LIMIT_MB:
        raise ValueError(
            f"XML file is {size_mb:.1f} MB, above the {XML_SIZE_HARD_LIMIT_MB:.0f} MB "
            "safety limit. Please convert it to CSV or split it first — very large "
            "XML trees can exhaust memory while flattening."
        )

    notes = []
    if size_mb > XML_SIZE_WARN_MB:
        notes.append(f"large file ({size_mb:.1f} MB) — parsed and previewed on the first rows only")

    df = None
    engine = None

    # ── Path 1: pandas.read_xml (fast, handles many regular shapes) ──────
    for parser in ("lxml", "etree"):
        try:
            candidate = pd.read_xml(io.BytesIO(data), parser=parser)
            if candidate is not None and not candidate.empty:
                df = candidate
                engine = f"pandas.read_xml ({parser})"
                break
        except ImportError:
            # lxml isn't installed — try the stdlib etree parser next
            continue
        except Exception:
            # irregular XML — stop trying pandas, the custom parser takes over
            break

    # ── Path 2: custom ElementTree parser (Problem 2 fallback) ───────────
    if df is None:
        try:
            text = data.decode("utf-8", errors="replace")
            root = ET.fromstring(text)
            records, strategy = _xml_find_records(root)

            truncated = False
            if len(records) > XML_MAX_RECORDS:
                records = records[:XML_MAX_RECORDS]
                truncated = True

            rows = []
            for rec in records:
                try:
                    row = _xml_record_to_row(rec)
                except Exception:
                    continue                    # one bad record never kills the file
                if row:
                    rows.append(row)

            if not rows:
                raise ValueError(
                    "The XML is well-formed but contains no tabular records "
                    "(no repeated elements, attributes, or text fields could "
                    "be extracted). Please check the file's structure."
                )

            # Problem 3: pd.DataFrame over a list of dicts automatically
            # collects the union of keys and fills missing ones with NaN —
            # exactly the uniform-row behaviour we need.
            df = pd.DataFrame(rows)
            df = df.dropna(axis=1, how="all")   # drop columns that are empty everywhere
            if df.empty or df.shape[1] == 0:
                raise ValueError(
                    "Records were found, but every field was empty after flattening."
                )
            df = df.reset_index(drop=True)
            engine = f"custom parser ({strategy})"
            if truncated:
                notes.append(f"parsing capped at the first {XML_MAX_RECORDS:,} records")

        except ET.ParseError as pe:
            raise ValueError(
                f"Malformed XML: {pe}. The file could not be parsed — "
                "please make sure it is well-formed (matching open/close tags)."
            )

    # ── Problem 4: post-import dtype clean-up (never fatal) ──────────────
    converted = {"numeric": [], "boolean": [], "datetime": []}
    try:
        df, converted = _xml_smart_convert(df)
    except Exception:
        pass

    n_numeric = len(converted["numeric"])
    n_bool = len(converted["boolean"])
    n_dt = len(converted["datetime"])
    typed_bits = []
    if n_numeric:
        typed_bits.append(f"{n_numeric} numeric")
    if n_bool:
        typed_bits.append(f"{n_bool} boolean")
    if n_dt:
        typed_bits.append(f"{n_dt} datetime")

    msg = (f"✅ XML loaded via {engine} — "
           f"{len(df):,} rows × {df.shape[1]} columns")
    if typed_bits:
        msg += f" (auto-typed: {', '.join(typed_bits)})"
    if notes:
        msg += " · " + "; ".join(notes)
    return df, msg


# ============================================================
# Universal File Loader  → (df_or_ExcelFile, info_message)
# ============================================================
def _json_to_df(raw):
    """Problem: nested JSON. Flatten nested objects via json_normalize;
    when the top level is a dict, prefer an embedded list-of-records."""
    if isinstance(raw, list):
        return pd.json_normalize(raw)
    if isinstance(raw, dict):
        for val in raw.values():
            if isinstance(val, list) and val and isinstance(val[0], (dict, list)):
                return pd.json_normalize(val)
        return pd.json_normalize(raw)
    raise ValueError("JSON content does not look like a table (expected a list of records or an object).")


def load_file(uploaded_file):
    """Load any supported upload and return (df_or_ExcelFile, info_message).

    Keeping ONE return contract for every format (Problem 7):
      - single-sheet CSV/Excel/JSON/XML → (DataFrame, message)
      - multi-sheet Excel               → (pd.ExcelFile, message) so the
                                          caller can offer a sheet picker
      - any failure                     → raises ValueError with a friendly
                                          message; callers show st.error()
    """
    name = (uploaded_file.name or "").lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df, f"✅ CSV loaded — {len(df):,} rows × {df.shape[1]} columns"

        if name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(uploaded_file)
            if len(xls.sheet_names) > 1:
                return xls, (f"📑 Excel file has {len(xls.sheet_names)} sheets — "
                             "pick one below to load it.")
            df = pd.read_excel(xls, sheet_name=0)
            return df, f"✅ Excel loaded — {len(df):,} rows × {df.shape[1]} columns"

        if name.endswith(".json"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            text = uploaded_file.read()
            text = text.decode("utf-8", errors="replace").strip() if isinstance(text, bytes) else text.strip()
            if not text:
                raise ValueError("The JSON file is empty — nothing to parse.")
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                # maybe JSON-lines (one object per line)
                try:
                    raw = [json.loads(line) for line in text.splitlines() if line.strip()]
                except json.JSONDecodeError as je:
                    raise ValueError(f"Malformed JSON: {je}")
            df = _json_to_df(raw)
            df = df.reset_index(drop=True)
            return df, (f"✅ JSON loaded & flattened — {len(df):,} rows × "
                        f"{df.shape[1]} columns")

        if name.endswith(".xml"):
            return load_xml_file(uploaded_file)

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not load “{uploaded_file.name}”: {e}")

    raise ValueError("Unsupported file type. Please upload CSV, Excel, JSON, or XML.")


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧪 AutoML Workspace")
        st.caption("CSV · Excel · JSON · **XML**")
        st.markdown("---")

        st.markdown("**📚 Datasets in memory**")
        options = get_all_dataset_options()
        if not options:
            st.info("No datasets yet. Upload one in the Data Manager tab.")
        else:
            for label, df in options.items():
                with st.container():
                    st.markdown(f"• **{label}**  \n"
                                f"  `{df.shape[0]:,} × {df.shape[1]}`")

        with st.expander("🕘 Data version log"):
            if not st.session_state.version_log:
                st.caption("No changes logged yet.")
            else:
                for n, note in reversed(st.session_state.version_log):
                    st.caption(f"v{n} — {note}")

        st.markdown("---")
        with st.expander("ℹ️ About file support"):
            st.markdown(
                "- **CSV** — plain rows & columns\n"
                "- **Excel** — single sheet auto-loads, multi-sheet asks\n"
                "- **JSON** — nested objects are flattened\n"
                "- **XML** — trees are flattened: repeated elements become "
                "rows, attributes become `tag__attr` columns, namespaces are "
                "stripped, types are inferred after import\n\n"
                "Every format returns the same `(DataFrame, message)` pair, "
                "so downstream tabs never care where the data came from."
            )


# ============================================================
# TAB 0 — Data Manager
# ============================================================
def tab_data_manager():
    st.header("📁 Data Manager")
    st.caption(
        "Upload **CSV**, **Excel**, **JSON** (nested gets flattened), or **XML** "
        "(trees get flattened into rows & columns). Every upload becomes a "
        "labelled dataset available everywhere in the app."
    )

    uploaded_files = st.file_uploader(
        "Upload dataset(s)",
        type=["csv", "xlsx", "xls", "json", "xml"],
        accept_multiple_files=True,
        key="dm_upload",
        help="CSV, Excel (.xlsx/.xls), JSON, XML — multiple files welcome.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            label = uf.name
            already_have = (label in st.session_state.dfs
                            or label in st.session_state.pending_excel)
            if already_have:
                continue                    # don't re-parse on every rerun
            try:
                result = load_file(uf)
                if isinstance(result[0], pd.ExcelFile):
                    st.session_state.pending_excel[label] = result[0]
                    st.info(f"**{label}** — {result[1]}")
                else:
                    df, info = result
                    st.session_state.dfs[label] = df
                    st.session_state.active_label = label
                    bump_version(f"Uploaded {label}")
                    st.success(f"**{label}** — {info}")
            except Exception as e:
                # Problem 8: a bad file never crashes the upload flow
                st.error(f"**{label}** — {e}")

    # ── Pending multi-sheet Excel files: sheet pickers ────────────────────
    for label, xls in list(st.session_state.pending_excel.items()):
        st.markdown(f"**📑 {label}**")
        pick_col, btn_col, skip_col = st.columns([3, 1, 1])
        with pick_col:
            sheet = st.selectbox("Sheet to load", xls.sheet_names,
                                 key=f"sheet_pick_{label}")
        with btn_col:
            st.write("")
            if st.button("Load sheet", key=f"sheet_load_{label}"):
                try:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    st.session_state.dfs[label] = df
                    st.session_state.active_label = label
                    del st.session_state.pending_excel[label]
                    bump_version(f"Uploaded {label} (sheet: {sheet})")
                    st.success(f"Loaded sheet **{sheet}** — {len(df):,} rows × {df.shape[1]} columns")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not read sheet: {e}")
        with skip_col:
            st.write("")
            if st.button("Dismiss", key=f"sheet_skip_{label}"):
                del st.session_state.pending_excel[label]
                st.rerun()

    # ── Registry ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Uploaded Datasets")

    if not st.session_state.dfs:
        st.info("No datasets yet — upload a file above to get started.")
        return

    for label in list(st.session_state.dfs.keys()):
        df = st.session_state.dfs[label]
        num_c = numeric_cols(df)
        cat_c = categorical_cols(df)
        missing = int(df.isna().sum().sum())
        mem_kb = df.memory_usage(deep=True).sum() / 1024

        st.markdown(
            f"""<div class="df-card">
                <h4>📄 {label}</h4>
                <p>{df.shape[0]:,} rows × {df.shape[1]} columns ·
                   {len(num_c)} numeric · {len(cat_c)} categorical ·
                   {missing:,} missing · {mem_kb:,.0f} KB</p>
                </div>""",
            unsafe_allow_html=True,
        )

        with st.expander(f"👀 Preview — {label}", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)
            if df.shape[0] > 100:
                st.caption(f"Preview limited to the first 100 of {df.shape[0]:,} rows.")
            st.markdown(f"**Columns:** {', '.join(map(str, df.columns[:30]))}"
                        + (" …" if df.shape[1] > 30 else ""))

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("⭐ Set active", key=f"activate_{label}"):
                st.session_state.active_label = label
                st.toast(f"Active dataset: {label}")
        with b2:
            download_csv_button(df, f"{label.rsplit('.', 1)[0]}.csv")
        with b3:
            if st.button("🗑️ Delete", key=f"delete_{label}"):
                del st.session_state.dfs[label]
                st.session_state.preprocessed_dfs.pop(label, None)
                st.session_state.reduced_dfs.pop(label, None)
                st.session_state.reduced_info.pop(label, None)
                if st.session_state.active_label == label:
                    st.session_state.active_label = None
                bump_version(f"Deleted {label}")
                st.rerun()

    if st.session_state.active_label:
        st.caption(f"⭐ Active dataset: **{st.session_state.active_label}** "
                   "(used as the default pick in other tabs)")


# ============================================================
# TAB 1 — Preprocessing
# ============================================================
def tab_preprocessing():
    st.header("🔧 Preprocessing")

    options = get_all_dataset_options()
    if not options:
        st.warning("No dataset available. Upload a file first.")
        return

    labels = list(options.keys())
    default_label = smart_default_label(labels)
    idx = labels.index(default_label) if default_label in labels else 0
    source_label = st.selectbox("📁 Dataset to preprocess", labels, index=idx,
                                key="prep_source_select")
    df = options[source_label]
    base_label = (source_label.replace("PCA: ", "")
                              .replace("Preprocessed: ", ""))

    df_info_card(df, source_label)
    with st.expander("📋 View Selected Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    out = df.copy()

    # ── 1. Column selection ───────────────────────────────────────────────
    st.subheader("1️⃣ Columns")
    drop_cols = st.multiselect("Drop columns (they will be removed from the copy)",
                               list(out.columns), key="prep_drop_cols")
    if drop_cols:
        out = out.drop(columns=drop_cols)
    if out.shape[1] == 0:
        st.error("All columns were dropped — nothing left to preprocess.")
        return

    # ── 2. Missing values ─────────────────────────────────────────────────
    st.subheader("2️⃣ Missing Values")
    n_missing = int(out.isna().sum().sum())
    if n_missing == 0:
        st.success("No missing values — nothing to do here.")
        miss_strategy = "Do nothing"
    else:
        st.caption(f"{n_missing:,} missing cells detected.")
        miss_strategy = st.selectbox(
            "Strategy",
            ["Do nothing", "Drop rows with any missing",
             "Numeric → Mean, Categorical → Mode",
             "Numeric → Median, Categorical → Mode",
             "Fill everything with 0 / 'Unknown'"],
            key="prep_missing",
        )
        if miss_strategy == "Drop rows with any missing":
            out = out.dropna().reset_index(drop=True)
        elif miss_strategy in ("Numeric → Mean, Categorical → Mode",
                               "Numeric → Median, Categorical → Mode"):
            for col in out.columns:
                if out[col].isna().any():
                    if pd.api.types.is_numeric_dtype(out[col]):
                        filler = out[col].mean() if "Mean" in miss_strategy else out[col].median()
                        out[col] = out[col].fillna(filler)
                    else:
                        mode = out[col].mode(dropna=True)
                        out[col] = out[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        elif miss_strategy == "Fill everything with 0 / 'Unknown'":
            for col in out.columns:
                if out[col].isna().any():
                    out[col] = out[col].fillna(0 if pd.api.types.is_numeric_dtype(out[col]) else "Unknown")

    if out.empty:
        st.error("No rows left after handling missing values.")
        return

    # ── 3. Duplicates & outliers ──────────────────────────────────────────
    st.subheader("3️⃣ Duplicates & Outliers")
    c_dup, c_out = st.columns(2)
    with c_dup:
        n_dup = int(out.duplicated().sum())
        drop_dups = st.checkbox(f"Remove duplicated rows ({n_dup:,} found)",
                                value=False, key="prep_drop_dups")
        if drop_dups and n_dup:
            out = out.drop_duplicates().reset_index(drop=True)
    with c_out:
        outlier_mode = st.selectbox("Outlier handling (numeric columns)",
                                    ["Keep as-is", "Flag in new column",
                                     "Remove rows (IQR rule)"], key="prep_outliers")
        if outlier_mode != "Keep as-is":
            num_c_out = numeric_cols(out)
            if not num_c_out:
                st.info("No numeric columns to check for outliers.")
            else:
                q1 = out[num_c_out].quantile(0.25)
                q3 = out[num_c_out].quantile(0.75)
                iqr = q3 - q1
                mask = ((out[num_c_out] < (q1 - 1.5 * iqr))
                        | (out[num_c_out] > (q3 + 1.5 * iqr))).any(axis=1)
                if outlier_mode == "Flag in new column":
                    out["outlier_flag"] = mask.astype(bool)
                    st.caption(f"{int(mask.sum()):,} rows flagged via the IQR rule.")
                else:
                    out = out[~mask].reset_index(drop=True)
                    st.caption(f"Removed {int(mask.sum()):,} outlier rows (IQR rule).")

    # ── 4. Encoding ───────────────────────────────────────────────────────
    st.subheader("4️⃣ Encode Categorical Columns")
    cat_cols_now = categorical_cols(out)
    if not cat_cols_now:
        st.success("No categorical columns — nothing to encode.")
    else:
        enc_mode = st.selectbox("Encoding", ["None", "One-Hot Encode", "Label Encode"],
                                key="prep_encoding")
        if enc_mode == "One-Hot Encode":
            enc_cols = st.multiselect("Columns to one-hot encode", cat_cols_now,
                                      default=cat_cols_now, key="prep_ohe_cols")
            if enc_cols:
                out = pd.get_dummies(out, columns=enc_cols, dummy_na=False)
        elif enc_mode == "Label Encode":
            enc_cols = st.multiselect("Columns to label encode", cat_cols_now,
                                      default=cat_cols_now, key="prep_le_cols")
            for col in enc_cols:
                out[col] = LabelEncoder().fit_transform(out[col].astype(str))

    # ── 5. Scaling ────────────────────────────────────────────────────────
    st.subheader("5️⃣ Scale Numeric Columns")
    scale_mode = st.selectbox("Scaling", ["None", "Standard (z-score)",
                                          "Min-Max (0–1)"], key="prep_scaling")
    if scale_mode != "None":
        num_cols_now = numeric_cols(out)
        scale_cols = st.multiselect("Columns to scale", num_cols_now,
                                    default=num_cols_now, key="prep_scale_cols")
        if scale_cols:
            scaler = StandardScaler() if "Standard" in scale_mode else MinMaxScaler()
            out[scale_cols] = scaler.fit_transform(out[scale_cols].astype(float))

    # ── Preview & save ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("👀 Preview of Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{out.shape[0]:,}", delta=f"{out.shape[0] - df.shape[0]:+,}")
    c2.metric("Columns", f"{out.shape[1]:,}", delta=f"{out.shape[1] - df.shape[1]:+,}")
    c3.metric("Missing Cells", f"{int(out.isna().sum().sum()):,}")
    st.dataframe(out.head(50), use_container_width=True)

    st.caption("Nothing is saved until you click below — the original stays untouched.")
    if st.button("💾 Save Preprocessed Dataset", key="btn_save_prep"):
        st.session_state.preprocessed_dfs[base_label] = out
        st.session_state.active_label = f"Preprocessed: {base_label}"
        bump_version(f"Preprocessed {base_label}")
        st.success(f"✅ Saved as **Preprocessed: {base_label}** — available in EDA, "
                   "PCA, Regression, Classification, and Clustering tabs.")


# ============================================================
# TAB 2 — EDA & Visualization
# ============================================================
def tab_eda():
    st.header("📊 EDA & Visualization")

    options = get_all_dataset_options()
    if not options:
        st.warning("No dataset available. Upload a file first.")
        return

    labels = list(options.keys())
    default_label = smart_default_label(labels)
    idx = labels.index(default_label) if default_label in labels else 0
    label = st.selectbox("📁 Dataset to explore", labels, index=idx, key="eda_source_select")
    df = options[label]

    df_info_card(df, label)
    with st.expander("📋 View Selected Dataset", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    num_c = numeric_cols(df)
    cat_c = categorical_cols(df)

    eda_t1, eda_t2, eda_t3, eda_t4, eda_t5 = st.tabs([
        "📋 Overview", "📈 Distributions", "🔗 Correlations",
        "🛠️ Custom Chart", "🌐 Pair Plot"
    ])

    # ── Overview ──────────────────────────────────────────────────────────
    with eda_t1:
        st.subheader("Dataset Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Column Types**")
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Dtype": [str(t) for t in df.dtypes],
                "Non-Null": df.notna().sum().values,
                "Missing": df.isna().sum().values,
                "Unique": df.nunique().values,
            })
            st.dataframe(dtype_df, use_container_width=True, height=320)
        with c2:
            st.markdown("**Summary Statistics**")
            if num_c:
                st.dataframe(df[num_c].describe().T.round(3),
                             use_container_width=True, height=320)
            else:
                st.info("No numeric columns to summarise.")

        if cat_c:
            st.markdown("**Categorical Snapshot**")
            cat_summary = pd.DataFrame({
                "Column": cat_c,
                "Unique Values": [df[c].nunique() for c in cat_c],
                "Top Value": [(df[c].mode(dropna=True).iloc[0]
                               if not df[c].mode(dropna=True).empty else "—") for c in cat_c],
                "Top Frequency": [(int(df[c].value_counts(dropna=True).iloc[0])
                                   if not df[c].value_counts(dropna=True).empty else 0) for c in cat_c],
            })
            st.dataframe(cat_summary, use_container_width=True)

    # ── Distributions ─────────────────────────────────────────────────────
    with eda_t2:
        st.subheader("Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numeric**")
            if num_c:
                dist_col = st.selectbox("Column", num_c, key="eda_dist_col")
                nbins = st.slider("Bins", 10, 120, 40, key="eda_dist_bins")
                fig_d = px.histogram(df, x=dist_col, nbins=nbins, template=pt(),
                                     marginal="box", color_discrete_sequence=["#1a1a2e"],
                                     title=f"Distribution of {dist_col}")
                st.plotly_chart(fig_d, use_container_width=True)
            else:
                st.info("No numeric columns in this dataset.")
        with c2:
            st.markdown("**Categorical**")
            if cat_c:
                cat_col = st.selectbox("Column", cat_c, key="eda_cat_col")
                top_n = st.slider("Top N", 5, 30, 15, key="eda_topn")
                vc = df[cat_col].astype(str).value_counts().head(top_n).reset_index()
                vc.columns = [cat_col, "Count"]
                fig_v = px.bar(vc, x=cat_col, y="Count", template=pt(),
                               color_discrete_sequence=["#374151"],
                               title=f"Top {top_n} values of {cat_col}")
                fig_v.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.info("No categorical columns in this dataset.")

    # ── Correlations ──────────────────────────────────────────────────────
    with eda_t3:
        st.subheader("Correlation Analysis")
        selected_num = st.multiselect("Numeric columns", num_c,
                                      default=num_c[:min(8, len(num_c))],
                                      key="corr_cols")
        corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"],
                                   key="corr_method")
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
        else:
            st.info("Select at least 2 numeric columns.")

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
    st.markdown(f"**Model:** `{saved_model_name}` · predicting `{saved_target}`", unsafe_allow_html=True)

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
                    manual_vals[feat] = st.number_input(feat, value=0.0, key=f"reg_manual_{feat}")
        if st.button("🔮 Predict", key="reg_manual_predict"):
            try:
                X_manual = np.array([[manual_vals[f] for f in saved_features]])
                pred = st.session_state.trained_reg_model.predict(X_manual)
                st.success(f"**Predicted {saved_target}:** {pred[0]:,.4f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with pred_tab_file:
        future_file = st.file_uploader("Upload future dataset", type=["csv","xlsx","xls","json","xml"], key="reg_future_upload")
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
                        filtered_future_reg = interactive_result_table(future_df, "reg_future", "Predictions", "reg_future_preds")
                        download_csv_button(filtered_future_reg, "regression_predictions.csv")
                        fig_pred = px.histogram(future_df, x=f"Predicted_{saved_target}", template=pt(),
                            title="Prediction Distribution", color_discrete_sequence=["#1a1a2e"])
                        st.plotly_chart(fig_pred, use_container_width=True)
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
    if len(num_c) < 1:
        st.error("Need at least 1 numeric feature column.")
        return
    c1,c2 = st.columns(2)
    with c1: target = st.selectbox("Target (Y)", df.columns.tolist(), key="cls_target")
    with c2: features = st.multiselect("Feature columns (X)", [c for c in num_c if c!=target],
        default=[c for c in num_c if c!=target][:8], key="cls_features")
    if not features:
        st.info("Select at least one feature.")
        return
    model_df = df[features+[target]].dropna().copy()
    if model_df.empty or model_df[target].nunique() < 2:
        st.error("Target needs at least 2 distinct classes after dropping NaN rows.")
        return
    le = LabelEncoder()
    y_raw = model_df[target].astype(str).values
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    X = model_df[features].values
    st.markdown(f"**{len(model_df):,} samples** · **{len(class_names)} classes** after dropping NaN rows.")
    c3,c4,c5 = st.columns(3)
    with c3: model_choice = st.selectbox("Model", ["Logistic Regression","K-Nearest Neighbors","SVM (RBF)","Decision Tree","Random Forest"], key="cls_model")
    with c4: test_size = st.slider("Test Size (%)", 10, 40, 20, key="cls_test_size") / 100
    with c5: eval_mode = st.selectbox("Evaluation", ["Train/Test Split","Cross-Validation","Both"], key="cls_eval")
    k_val, svm_c, dt_depth, rf_n, rf_d = 5, 1.0, 5, 100, 5
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
    st.markdown(f"**Model:** `{saved_model_name}` · {len(saved_features)} features · {len(saved_class_names)} classes", unsafe_allow_html=True)
    future_file = st.file_uploader("Upload future dataset", type=["csv","xlsx","xls","json","xml"], key="cls_future_upload")
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
    st.markdown(f"**Model:** `{saved_clust_name}` · {len(saved_features)} features", unsafe_allow_html=True)
    future_file = st.file_uploader("Upload future dataset", type=["csv","xlsx","xls","json","xml"], key="clust_future_upload")
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

    st.markdown('<div class="hero-title">ML and EDA Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-tags">'
        'CSV  ·  Excel  ·  JSON  ·  XML'
        '   |   '
        'Preprocess  ·  Explore  ·  Model  ·  Cluster'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<hr class="hero-divider"/>', unsafe_allow_html=True)

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
