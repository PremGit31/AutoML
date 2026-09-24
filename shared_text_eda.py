"""
shared_text_eda.py
===================

Plain, reusable TEXT EDA functions — no Streamlit page setup, no
session_state, no widgets. Mirrors shared_eda.py's role (which handles
structured/numeric EDA) but for a free-text column: length distributions,
n-gram frequency, a word cloud, and KWIC (Key Word In Context) search.
Functions take a pandas Series of text and return either data (DataFrames)
or Plotly `Figure` objects / numpy image arrays a caller can render
however it likes.

WHY THIS FILE EXISTS:
Same reason as shared_eda.py / shared_features.py / shared_preprocessing.py
/ shared_training.py before it: nlp_app2.py is a full Streamlit PAGE, and
importing it directly would re-run its unconditional `st.set_page_config()`
and session-state setup at import time — which crashes fusion.py, which
already calls its own st.set_page_config(). This file holds copies of the
side-effect-free chart/data logic from nlp_app2.py's tab_text_eda(), with
nothing else in it. Safe to import from anywhere.

nlp_app2.py's own tab_text_eda() is left completely untouched — these are
copies of the underlying computation (compute_ngram_freq, the length/n-gram/
word-cloud/KWIC mechanics), not moves.

WHY IT EXISTS SEPARATELY FROM shared_eda.py:
shared_eda.py's column-utility/relevance-ranking machinery
(classify_column_utility, rank_columns_by_target_relevance, ...) is built
around dataframes of many typed columns. Text EDA operates on a single
free-text Series with entirely different mechanics (tokenizing, n-grams,
word clouds, KWIC) — bolting that onto shared_eda.py would blur what each
file is for. Keeping it separate also matches how nlp_app2.py itself
keeps Text EDA as its own tab, distinct from any per-column chart-picking
logic.

WHAT'S NEW vs. nlp_app2.py:
Only auto_text_eda() is new — a bundling function (not present in
nlp_app2.py) that runs a sensible default pass (length charts + top
unigrams/bigrams + a word cloud) in one call, so a caller can show
something immediately on a single "Analyze Text" trigger, the same
"automatic first, then user adjusts" pattern shared_eda.auto_eda() already
established for structured EDA. KWIC search is NOT included in
auto_text_eda() — it needs a search term, so there is nothing sensible to
auto-run; callers should always render the KWIC control and call
kwic_search() only once the user types something.

NOTHING in this file talks to st.session_state or renders any UI.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px

TEMPLATE = "plotly_white"
COLOR_SEQ = ["#1a1a2e", "#374151", "#6366f1", "#10b981", "#f59e0b", "#3b82f6", "#ec4899"]


# ---------------------------------------------------------------------------
# 1. LENGTH DISTRIBUTIONS
# (mechanic copied from nlp_app2.py tab_text_eda "eda_tabs[0] Length
#  Distribution")
# ---------------------------------------------------------------------------

def length_stats(text_series: pd.Series) -> pd.DataFrame:
    """Word-count / char-count per document. Mirrors nlp_app2.py's
    length-distribution computation exactly."""
    return pd.DataFrame({
        "word_count": text_series.apply(lambda x: len(str(x).split())),
        "char_count": text_series.apply(lambda x: len(str(x))),
    })


def build_word_count_histogram(text_series: pd.Series):
    lengths = length_stats(text_series)
    return px.histogram(
        lengths, x="word_count", nbins=50, template=TEMPLATE,
        title="Word Count Distribution", color_discrete_sequence=[COLOR_SEQ[0]],
        labels={"word_count": "Words per Document"},
    )


def build_char_count_histogram(text_series: pd.Series):
    lengths = length_stats(text_series)
    return px.histogram(
        lengths, x="char_count", nbins=50, template=TEMPLATE,
        title="Character Count Distribution", color_discrete_sequence=[COLOR_SEQ[2]],
        labels={"char_count": "Characters per Document"},
    )


# ---------------------------------------------------------------------------
# 2. N-GRAM FREQUENCY
# (mechanic copied from nlp_app2.py's get_ngrams()/compute_ngram_freq()
#  helpers and "eda_tabs[1] N-gram Frequency")
# ---------------------------------------------------------------------------

def get_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_ngram_freq(series: pd.Series, n: int, top_k: int = 20) -> pd.DataFrame:
    """Mirrors nlp_app2.py's compute_ngram_freq() exactly (lowercase,
    whitespace-split tokenization — same as the rest of that tab)."""
    all_ngrams: list[str] = []
    for doc in series.dropna():
        tokens = str(doc).lower().split()
        all_ngrams.extend(get_ngrams(tokens, n))
    counter = Counter(all_ngrams)
    return pd.DataFrame(counter.most_common(top_k), columns=["ngram", "count"])


def build_ngram_chart(series: pd.Series, n: int, top_k: int = 20):
    """Mirrors nlp_app2.py's n-gram bar chart. Returns None (rather than
    an empty figure) when there isn't enough text to produce any n-grams,
    so the caller can show an info message instead of a blank chart."""
    df = compute_ngram_freq(series, n, top_k)
    if df.empty:
        return None
    label = {1: "Uni", 2: "Bi", 3: "Tri"}.get(n, "N")
    fig = px.bar(
        df.sort_values("count"), x="count", y="ngram", orientation="h", template=TEMPLATE,
        title=f"Top {top_k} {label}grams", color_discrete_sequence=[COLOR_SEQ[0]],
        labels={"count": "Frequency", "ngram": ""},
    )
    fig.update_layout(height=max(400, top_k * 22))
    return fig


# ---------------------------------------------------------------------------
# 3. WORD CLOUD
# (mechanic copied from nlp_app2.py's "eda_tabs[2] Word Cloud" — with one
#  deliberate change: returns a numpy image array via WordCloud.to_array()
#  instead of a matplotlib Figure/Axes. nlp_app2.py builds a matplotlib
#  fig and calls st.pyplot(); that requires holding a stateful pyplot
#  figure object. Since this module renders nothing itself, returning a
#  plain array lets any caller st.image() it directly with no matplotlib
#  figure lifecycle to manage.)
# ---------------------------------------------------------------------------

def build_wordcloud_array(
    text_series: pd.Series, max_words: int = 150, bg_color: str = "#f8f7f4"
) -> Optional[np.ndarray]:
    """Returns an RGB numpy array ready for st.image(), or None if there
    isn't enough text, or the `wordcloud` package isn't installed."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    full_text = " ".join(text_series.dropna().astype(str).tolist())
    if len(full_text.strip()) < 10:
        return None
    wc = WordCloud(
        width=1200, height=500, max_words=max_words,
        background_color=bg_color, colormap="cividis",
    ).generate(full_text)
    return wc.to_array()


# ---------------------------------------------------------------------------
# 4. KWIC (Key Word In Context) SEARCH
# (mechanic copied from nlp_app2.py's "eda_tabs[3] KWIC Search" exactly —
#  same tokenization, same window logic, same 500-result cap.)
# ---------------------------------------------------------------------------

def kwic_search(
    text_series: pd.Series,
    query: str,
    window: int = 7,
    case_sensitive: bool = False,
    max_results: int = 500,
) -> pd.DataFrame:
    """Returns columns [doc_id, left_context, hit, right_context]. Empty
    DataFrame (not an error) for a blank query or no matches."""
    if not query or not query.strip():
        return pd.DataFrame(columns=["doc_id", "left_context", "hit", "right_context"])

    pattern = re.compile(re.escape(query.strip()), flags=0 if case_sensitive else re.IGNORECASE)
    results = []
    for idx, doc in text_series.items():
        tokens = str(doc).split()
        for i, tok in enumerate(tokens):
            if pattern.search(tok):
                left = " ".join(tokens[max(0, i - window):i])
                right = " ".join(tokens[i + 1:i + 1 + window])
                results.append({"doc_id": idx, "left_context": left, "hit": tok, "right_context": right})
        if len(results) >= max_results:
            break
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# NEW: bundled "automatic by default" pass — not present in nlp_app2.py.
# ---------------------------------------------------------------------------

def auto_text_eda(text_series: pd.Series, top_k: int = 20, max_words: int = 150) -> dict:
    """
    Single-call, "show something immediately" text-EDA pass: length
    distributions + top unigrams/bigrams + a word cloud, all pre-rendered.
    KWIC is intentionally excluded (see module docstring) — callers should
    always render the search box themselves and call kwic_search() once
    the user types a term.

    Returns:
      {
        "n_docs": int,
        "word_count_fig": Figure, "char_count_fig": Figure,
        "unigram_fig": Figure | None, "bigram_fig": Figure | None,
        "wordcloud_array": np.ndarray | None,
      }
    """
    return {
        "n_docs": int(text_series.notna().sum()),
        "word_count_fig": build_word_count_histogram(text_series),
        "char_count_fig": build_char_count_histogram(text_series),
        "unigram_fig": build_ngram_chart(text_series, 1, top_k),
        "bigram_fig": build_ngram_chart(text_series, 2, top_k),
        "wordcloud_array": build_wordcloud_array(text_series, max_words),
    }
