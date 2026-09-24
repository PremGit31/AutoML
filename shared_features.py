"""
shared_features.py
===================

Plain, reusable feature-extraction functions — no Streamlit page setup,
no session_state, no widgets. Just "data in, numbers out" functions.

WHY THIS FILE EXISTS:
nlp_app2.py and image_app.py are each a full Streamlit PAGE. When Python
imports a file, it runs everything in that file top to bottom — including
things like st.set_page_config(...), the CSS block, and the session-state
setup near the top of image_app.py. That's fine when the page is opened
directly, but it means fusion.py can't safely write
`from image_app import extract_full_image_features` — doing so would
re-run all of image_app.py's page-setup code as a side effect and could
break things.

The fix is this file: it holds copies of the two genuinely reusable,
side-effect-free pieces of logic (image feature extraction from
image_app.py, and text vectorization pulled out of nlp_app2.py's
vectorization tab), with nothing else in it. It's safe to import from
anywhere.

NOTHING in this file talks to st.session_state or renders any UI.
If you're tempted to add a widget here, it belongs in the page instead.

RELATIONSHIP TO THE ORIGINAL FILES:
- image_app.py's own copy of extract_full_image_features/calc_glcm_features
  is left completely untouched (it's your teammate's file) — the versions
  below are copies, not moves.
- nlp_app2.py's vectorization tab is also left untouched. vectorize_text()
  below is a new function that does the same computation as that tab, in a
  reusable form. The tab and this file will compute the same numbers, but
  currently are two separate pieces of code — worth eventually pointing
  the tab at this function too, so there's only one place to fix bugs, but
  that's optional cleanup, not required for fusion to work.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import cv2


# ---------------------------------------------------------------------------
# GPU detection (copied from nlp_app2.py's detect_gpu — identical logic)
# ---------------------------------------------------------------------------

def detect_gpu() -> dict:
    """Detect CUDA availability once. Returns dict with status info."""
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


# ---------------------------------------------------------------------------
# TEXT -> NUMBERS
# ---------------------------------------------------------------------------

def vectorize_text(
    text_series: pd.Series,
    method: str = "sentence_transformer",
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_fp16: bool = False,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 1),
) -> pd.DataFrame:
    """
    Turns a column of text into a table of numbers, one row per input text.

    method: "sentence_transformer" (semantic, GPU-aware — the fusion default)
            "tfidf" or "count" (fast, keyword-based, no GPU needed)

    Returns a DataFrame with as many rows as the input Series, in the same
    order, so it can be safely glued onto other columns by position.
    """
    docs = text_series.fillna("").astype(str).tolist()

    if method == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        gpu_info = detect_gpu()
        resolved_device = device or ("cuda" if gpu_info["available"] else "cpu")
        model = SentenceTransformer(model_name, device=resolved_device)
        if use_fp16 and resolved_device == "cuda":
            model = model.half()
        resolved_batch = batch_size or (128 if gpu_info["available"] else 32)

        embeddings = model.encode(docs, batch_size=resolved_batch, show_progress_bar=False)
        col_names = [f"embed_{i}" for i in range(embeddings.shape[1])]
        return pd.DataFrame(embeddings, columns=col_names)

    elif method in ("tfidf", "count"):
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

        if method == "tfidf":
            vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
        else:
            vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        matrix = vec.fit_transform(docs)
        return pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out())

    else:
        raise ValueError(f"Unknown text vectorization method: {method!r}")


# ---------------------------------------------------------------------------
# IMAGE -> NUMBERS
# (calc_glcm_features + extract_full_image_features copied verbatim from
#  image_app.py, lines 228-338 — logic unchanged, just relocated so it can
#  be imported safely. If image_app.py's version ever changes, re-sync here.)
# ---------------------------------------------------------------------------

def calc_glcm_features(gray_img: np.ndarray) -> dict:
    """Compute GLCM texture properties across 0, 45, 90, 135 degrees."""
    scaled = (gray_img / 16).astype(np.uint8)
    glcm = np.zeros((16, 16), dtype=np.float64)
    h, w = scaled.shape
    for r in range(h):
        for c in range(w - 1):
            i_val, j_val = scaled[r, c], scaled[r, c + 1]
            glcm[i_val, j_val] += 1
            glcm[j_val, i_val] += 1
    sum_g = np.sum(glcm)
    if sum_g > 0:
        glcm /= sum_g

    contrast = 0.0
    dissimilarity = 0.0
    homogeneity = 0.0
    energy = 0.0
    mean_i = 0.0
    for i in range(16):
        for j in range(16):
            p = glcm[i, j]
            contrast += p * ((i - j) ** 2)
            dissimilarity += p * abs(i - j)
            homogeneity += p / (1.0 + (i - j) ** 2)
            energy += p ** 2
            mean_i += i * p

    correlation = 0.0
    std_i = np.sqrt(np.sum([((i - mean_i) ** 2) * np.sum(glcm[i, :]) for i in range(16)]))
    if std_i > 0:
        for i in range(16):
            for j in range(16):
                correlation += glcm[i, j] * (i - mean_i) * (j - mean_i) / (std_i ** 2)

    return {
        "GLCM_Contrast": round(float(contrast), 4),
        "GLCM_Dissimilarity": round(float(dissimilarity), 4),
        "GLCM_Homogeneity": round(float(homogeneity), 4),
        "GLCM_Energy_ASM": round(float(energy), 4),
        "GLCM_Correlation": round(float(correlation), 4),
    }


def extract_full_image_features(rgb_img: np.ndarray, filename: str = "image") -> dict:
    """Extract a comprehensive structured feature vector from an RGB image."""
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY) if len(rgb_img.shape) == 3 else rgb_img

    if len(rgb_img.shape) == 3:
        r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    else:
        r, g, b = gray, gray, gray

    features = {
        "Filename": filename,
        "Width": w,
        "Height": h,
        "Aspect_Ratio": round(w / h, 3),
        "Red_Mean": round(float(np.mean(r)), 2),
        "Red_Std": round(float(np.std(r)), 2),
        "Green_Mean": round(float(np.mean(g)), 2),
        "Green_Std": round(float(np.std(g)), 2),
        "Blue_Mean": round(float(np.mean(b)), 2),
        "Blue_Std": round(float(np.std(b)), 2),
        "Gray_Mean": round(float(np.mean(gray)), 2),
        "Gray_Std": round(float(np.std(gray)), 2),
        "Gray_Skewness": round(float(stats.skew(gray.ravel())), 3),
    }

    glcm_feats = calc_glcm_features(gray)
    features.update(glcm_feats)

    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        val = -1.0 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i])) if hu[i] != 0 else 0.0
        features[f"Hu_Moment_{i+1}"] = round(float(val), 4)

    edges = cv2.Canny(gray, 100, 200)
    features["Edge_Pixel_Ratio"] = round(float(np.count_nonzero(edges) / (h * w)), 4)

    return features


def extract_image_features_batch(image_bytes_list: list[bytes]) -> pd.DataFrame:
    """
    Convenience wrapper: takes raw image file bytes (e.g. straight out of a
    ZIP) and returns one row of numeric features per image, in the same
    order as the input list. This is what fusion.py calls.
    """
    rows = []
    for i, raw_bytes in enumerate(image_bytes_list):
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        rgb_array = np.array(pil_img)
        rows.append(extract_full_image_features(rgb_array, filename=f"image_{i}"))
    df = pd.DataFrame(rows)
    return df.drop(columns=["Filename"])  # not numeric, and fusion already tracks the id separately
