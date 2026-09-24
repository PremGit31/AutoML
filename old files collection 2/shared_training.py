"""
shared_training.py
===================

Plain, reusable model training/scoring functions — no Streamlit page
setup, no session_state, no widgets. Functions take arrays/dataframes and
return fitted models + result dicts a caller can render or feed into
fusion_core.py however it likes.

WHY THIS FILE EXISTS:
Same reason as shared_preprocessing.py / shared_eda.py before it: app12.py
is a full Streamlit PAGE, and importing it directly would re-run its
unconditional `st.set_page_config()` at import time — which crashes
because fusion.py already calls its own. This file holds copies of the
side-effect-free training/scoring logic from app12.py's `tab_regression`,
`tab_classification`, and `tab_clustering`, with nothing else in it.

app12.py's own tabs are left completely untouched — these are copies of
the underlying computation, not moves.

THIS FILE IS CLOSER TO A PURE EXTRACTION than shared_preprocessing.py or
shared_eda.py were: app12.py's regression/classification/clustering
mechanics (which model, how to split, how to score) are already solid,
well-tested computation — there's comparatively little "invent new
behavior" needed. The auto-decide layer added here is intentionally
narrow: which algorithm/hyperparameters to default to, and what train/
test split or CV setting to use, all via small, explainable, rule-based
heuristics — not a replacement for the extracted mechanics, which remain
available for manual/full-control use exactly as app12.py implements
them.

RULE-BASED, DETERMINISTIC, NO AI/LLM INVOLVED ANYWHERE. Every auto-decide
function returns a plain-English `description` string ready for an
inline "change this" UI control later — nothing here renders one.

NOTHING in this file talks to st.session_state or renders any UI.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix, classification_report, silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN


# ===========================================================================
# 1. REGRESSION
# (mechanics copied from app12.py tab_regression — same model constructors,
#  same train_test_split/metrics calls, same CV setup.)
# ===========================================================================

_REGRESSION_MODEL_NAMES = (
    "Linear Regression", "Ridge Regression", "Lasso Regression", "Polynomial Regression",
)


def build_regression_model(model_choice: str, alpha: float = 1.0, poly_degree: int = 2):
    """Mirrors app12.py's regression model_choice -> estimator mapping
    exactly, including the Polynomial Pipeline construction."""
    if model_choice == "Linear Regression":
        return LinearRegression()
    elif model_choice == "Ridge Regression":
        return Ridge(alpha=alpha)
    elif model_choice == "Lasso Regression":
        return Lasso(alpha=alpha)
    elif model_choice == "Polynomial Regression":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("lin", LinearRegression()),
        ])
    else:
        raise ValueError(f"Unknown regression model_choice: {model_choice!r}")


def train_regression_model(
    X: np.ndarray,
    y: np.ndarray,
    model_choice: str,
    alpha: float = 1.0,
    poly_degree: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train/test-split mechanic, mirrors app12.py's tab_regression training
    block exactly: same train_test_split call, same r2/rmse/mae metrics.
    Returns a result dict (model, split data, predictions, metrics).
    """
    model = build_regression_model(model_choice, alpha, poly_degree)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = mean_absolute_error(y_test, y_pred)

    return {
        "model": model,
        "model_choice": model_choice,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test, "y_pred": y_pred,
        "r2": r2, "rmse": rmse, "mae": mae,
    }


def cross_validate_regression(
    model, X: np.ndarray, y: np.ndarray, cv_folds: int = 5, random_state: int = 42
) -> dict:
    """Mirrors app12.py's KFold cross-validation block exactly (same
    KFold(shuffle=True, random_state), same r2 / neg_mean_squared_error
    scoring)."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
    return {
        "cv_r2_scores": cv_r2,
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "cv_rmse_scores": cv_rmse,
        "cv_rmse_mean": float(cv_rmse.mean()),
        "cv_folds": cv_folds,
    }


def extract_regression_coefficients(model, model_choice: str, features: list[str]) -> Optional[dict]:
    """Mirrors app12.py: coefficients are only reported for Linear/Ridge/
    Lasso — Polynomial Regression is a Pipeline without a direct `.coef_`
    matching the original feature names, so it's skipped, exactly as in
    app12.py."""
    if model_choice not in ("Linear Regression", "Ridge Regression", "Lasso Regression"):
        return None
    return {"features": features, "coefficients": model.coef_}


def predict_regression(model, X_new: np.ndarray) -> np.ndarray:
    return model.predict(X_new)


def score_regression_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Reusable metrics block — mirrors app12.py's future-data evaluation
    (used both for the main test split and for scoring predictions on a
    future/held-out file that happens to include the true target)."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# ===========================================================================
# 2. CLASSIFICATION
# (mechanics copied from app12.py tab_classification — same model
#  constructors, same stratify-eligibility check, same present-classes-
#  only classification_report, same CV setup.)
# ===========================================================================

_CLASSIFICATION_MODEL_NAMES = (
    "Logistic Regression", "K-Nearest Neighbors", "SVM (RBF)",
    "Decision Tree", "Random Forest",
)


def build_classification_model(
    model_choice: str,
    k: int = 5,
    svm_c: float = 1.0,
    dt_depth: int = 5,
    rf_n: int = 100,
    rf_d: int = 5,
):
    """Mirrors app12.py's classification model_choice -> estimator
    mapping exactly, including every hyperparameter default."""
    if model_choice == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_choice == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=k)
    elif model_choice == "SVM (RBF)":
        return SVC(C=svm_c, kernel="rbf", random_state=42, probability=True)
    elif model_choice == "Decision Tree":
        return DecisionTreeClassifier(max_depth=dt_depth, random_state=42)
    elif model_choice == "Random Forest":
        return RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d, random_state=42)
    else:
        raise ValueError(f"Unknown classification model_choice: {model_choice!r}")


def encode_classification_target(y_raw: pd.Series) -> tuple[np.ndarray, np.ndarray, Optional[LabelEncoder]]:
    """
    Mirrors app12.py's target-encoding behavior, with a more robust dtype
    check. app12.py's original test is
    `if y_raw.dtype == object or str(y_raw.dtype) == "category": ...`,
    which misses pandas' newer dedicated "string" dtype (the default for
    string columns since pandas 2.x/3.x) — a string-dtype target would
    fall through to `.astype(int)` and crash. This copy uses
    `pd.api.types.is_numeric_dtype()` instead: numeric/bool dtypes are
    treated as "already numeric" (same as app12.py), and everything else
    (object, category, the newer string dtype, ...) goes through
    LabelEncoder — identical behavior to app12.py for the dtypes it was
    actually exercised against, but robust to the newer string dtype too.
    Returns (y_encoded, class_names, label_encoder_or_None).
    """
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.values.astype(int)
        class_names = np.unique(y).astype(str)
        return y, class_names, None
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        class_names = le.classes_
        return y, class_names, le


def train_classification_model(
    X: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    model_choice: str,
    k: int = 5,
    svm_c: float = 1.0,
    dt_depth: int = 5,
    rf_n: int = 100,
    rf_d: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train/test-split mechanic, mirrors app12.py's tab_classification
    training block exactly: same stratify-eligibility check, same
    present-classes-only classification_report/confusion_matrix.
    """
    model = build_classification_model(model_choice, k, svm_c, dt_depth, rf_n, rf_d)

    min_class_count = pd.Series(y).value_counts().min()
    can_stratify = bool(len(np.unique(y)) > 1 and min_class_count >= 2)
    strat = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    present_idx = np.unique(np.concatenate([y_test, y_pred]))
    present_class_names = [class_names[i] for i in present_idx]
    report = classification_report(
        y_test, y_pred, target_names=present_class_names, labels=present_idx, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "model_choice": model_choice,
        "stratified": can_stratify,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test, "y_pred": y_pred,
        "acc": acc, "report": report, "cm": cm,
        "class_names": class_names, "present_class_names": present_class_names,
    }


def cross_validate_classification(
    model, X: np.ndarray, y: np.ndarray, cv_folds: int = 5, random_state: int = 42
) -> dict:
    """Mirrors app12.py's StratifiedKFold cross-validation block exactly."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return {
        "cv_accuracy_scores": cv_scores,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_folds": cv_folds,
    }


def extract_classification_feature_importance(model, model_choice: str) -> Optional[np.ndarray]:
    """Mirrors app12.py: feature importances are only reported for
    Decision Tree / Random Forest."""
    if model_choice in ("Decision Tree", "Random Forest"):
        return model.feature_importances_
    return None


def predict_classification(
    model, X_new: np.ndarray, label_encoder: Optional[LabelEncoder] = None
) -> np.ndarray:
    """Mirrors app12.py's future-data classification: predict, then
    inverse_transform back to original labels if a LabelEncoder was used."""
    preds = model.predict(X_new)
    if label_encoder is not None:
        return label_encoder.inverse_transform(preds)
    return preds.astype(str)


# ===========================================================================
# 3. CLUSTERING
# (mechanics copied from app12.py tab_clustering — same missing-value
#  mapping, same StandardScaler, same elbow/silhouette loop, same K-Means
#  and DBSCAN setup, same future-data assignment logic.)
# ===========================================================================

def scale_for_clustering(
    df: pd.DataFrame, feature_cols: list[str], missing_strategy: str = "mean"
) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    missing_strategy: "mean" | "drop" | "zero" — mirrors app12.py's
    "Fill with Mean" / "Drop Rows" / "Fill with 0" options exactly (note:
    mean, not median — app12.py's clustering tab has always used mean
    here, unlike shared_preprocessing's median-fill default; kept as-is
    to faithfully mirror the manual mechanic).
    Returns (X_scaled, fitted_scaler, cleaned_feature_df).
    """
    cluster_df = df[feature_cols].copy()
    if missing_strategy == "drop":
        cluster_df = cluster_df.dropna()
    elif missing_strategy == "mean":
        cluster_df = cluster_df.fillna(cluster_df.mean())
    elif missing_strategy == "zero":
        cluster_df = cluster_df.fillna(0)
    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy!r}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)
    return X_scaled, scaler, cluster_df


def compute_kmeans_elbow_curve(
    X_scaled: np.ndarray, k_range: Optional[range] = None, random_state: int = 42
) -> dict:
    """Mirrors app12.py's elbow/silhouette loop exactly (same
    KMeans(n_init=10, random_state=42) per k, same silhouette_score)."""
    n_samples = X_scaled.shape[0]
    if k_range is None:
        max_k = min(11, n_samples - 1)
        k_range = range(2, max_k + 1)
    k_values = list(k_range)

    inertias, silhouettes = [], []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    return {"k_values": k_values, "inertias": inertias, "silhouettes": silhouettes}


def train_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42) -> dict:
    """Mirrors app12.py's K-Means run block exactly."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return {"model": km, "algorithm": "K-Means", "k": k, "labels": labels, "silhouette": sil}


def train_dbscan(X_scaled: np.ndarray, eps: float, min_samples: int) -> dict:
    """Mirrors app12.py's DBSCAN run block exactly, including the
    silhouette-only-if->1-cluster-excluding-noise guard."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    sil = None
    if n_clusters > 1:
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 0 and len(np.unique(labels[non_noise_mask])) > 1:
            sil = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])

    return {
        "model": db, "algorithm": "DBSCAN", "eps": eps, "min_samples": min_samples,
        "labels": labels, "n_clusters": n_clusters, "n_noise": n_noise, "silhouette": sil,
    }


def assign_clusters_to_new_data(
    algorithm: str,
    scaler: StandardScaler,
    X_future_raw: np.ndarray,
    model=None,
    train_X_scaled: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Mirrors app12.py's future-data cluster assignment exactly:
      - K-Means: model.predict() on the scaled future data.
      - DBSCAN (no native out-of-sample predict): 1-nearest-neighbor
        lookup against the scaled training data, assigning the nearest
        training point's cluster label.
    """
    X_future_scaled = scaler.transform(X_future_raw)
    if algorithm == "K-Means":
        if model is None:
            raise ValueError("K-Means assignment requires the fitted model.")
        return model.predict(X_future_scaled)
    elif algorithm == "DBSCAN":
        if train_X_scaled is None or train_labels is None:
            raise ValueError("DBSCAN assignment requires train_X_scaled and train_labels.")
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(train_X_scaled)
        _, indices = nn.kneighbors(X_future_scaled)
        return train_labels[indices.flatten()]
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")


# ===========================================================================
# NEW: AUTO-DECIDE LOGIC — none of this exists in app12.py today.
# Deliberately narrow: which algorithm/hyperparameters and which split/CV
# setting to default to. The extracted mechanics above remain the full
# manual-control surface; auto mode is a thin, explainable layer on top.
# ===========================================================================

def _auto_decide_split_and_cv(n_samples: int) -> dict:
    """
    Shared train/test split + cross-validation default, used by both
    regression and classification auto-training:
      - n < 100   -> test_size 0.3, WITH 3-fold CV (small data needs a
        larger test fraction and CV to get a trustworthy estimate at all)
      - n < 500   -> test_size 0.2, WITH 5-fold CV (standard split, CV as
        a sanity check against an unlucky single split)
      - n >= 500  -> test_size 0.15, WITHOUT CV (a single held-out split
        is already stable at this size; skip CV to save compute)
    """
    if n_samples < 100:
        test_size, use_cv, cv_folds = 0.3, True, 3
        reason = (
            "very small dataset — a larger test fraction plus cross-validation gives "
            "a more trustworthy estimate than a single small split"
        )
    elif n_samples < 500:
        test_size, use_cv, cv_folds = 0.2, True, 5
        reason = (
            "moderate dataset size — a standard 80/20 split, with 5-fold CV as a "
            "sanity check against an unlucky single split"
        )
    else:
        test_size, use_cv, cv_folds = 0.15, False, 5
        reason = (
            "large dataset — a single held-out split is already stable at this size, "
            "so cross-validation is skipped by default to save compute"
        )
    return {
        "test_size": test_size,
        "use_cv": use_cv,
        "cv_folds": cv_folds,
        "description": (
            f"n={n_samples}: test_size={test_size}, "
            f"{'with' if use_cv else 'without'} cross-validation "
            f"({cv_folds}-fold) — {reason}."
        ),
    }


def auto_decide_regression_model(
    df: pd.DataFrame, features: list[str], multicollinearity_threshold: float = 0.8
) -> dict:
    """
    Rule-based default (not present in app12.py):
      - if features show strong pairwise multicollinearity (max |corr| >
        threshold) -> Ridge Regression (L2 penalty stabilizes coefficients
        under correlated inputs)
      - otherwise -> Linear Regression (most interpretable default when
        there's no strong collinearity to guard against)
    Polynomial Regression is deliberately never auto-picked — the right
    degree is a bias/variance trade-off that needs human judgment, so it
    stays a manual-only option via build_regression_model().
    """
    if len(features) > 1:
        corr = df[features].corr().abs()
        corr_values = corr.values.copy()  # pandas 3.0 Copy-on-Write can make
        # `.values` read-only; fill_diagonal needs a writable array.
        np.fill_diagonal(corr_values, 0.0)
        max_corr = float(np.nanmax(corr_values)) if corr_values.size else 0.0
    else:
        max_corr = 0.0

    if max_corr > multicollinearity_threshold:
        return {
            "model_choice": "Ridge Regression",
            "alpha": 1.0,
            "description": (
                f"Chose Ridge Regression — features show multicollinearity (max "
                f"pairwise correlation {max_corr:.2f} > {multicollinearity_threshold}), "
                f"and Ridge's L2 penalty stabilizes coefficients under correlated inputs."
            ),
        }
    return {
        "model_choice": "Linear Regression",
        "alpha": 1.0,
        "description": (
            f"Chose Linear Regression — features show low multicollinearity (max "
            f"pairwise correlation {max_corr:.2f}), so ordinary least squares is a "
            f"fine, maximally-interpretable default."
        ),
    }


def auto_decide_classification_model(
    n_samples: int, small_sample_threshold: int = 150
) -> dict:
    """
    Rule-based default (not present in app12.py):
      - n_samples < small_sample_threshold -> Logistic Regression (simple,
        low-variance — safer than a flexible model when there's little
        data to learn from)
      - otherwise -> Random Forest (robust general-purpose default for
        tabular classification: handles non-linearity, needs no scaling)
    """
    if n_samples < small_sample_threshold:
        return {
            "model_choice": "Logistic Regression",
            "description": (
                f"Chose Logistic Regression — only {n_samples} sample(s) available "
                f"(below the {small_sample_threshold}-sample threshold), so a simple, "
                f"low-variance model is safer than a complex one that could overfit."
            ),
        }
    return {
        "model_choice": "Random Forest",
        "description": (
            f"Chose Random Forest — {n_samples} samples is enough to support a more "
            f"flexible model; Random Forest is a robust general-purpose default for "
            f"tabular classification (handles non-linearity, needs no feature scaling)."
        ),
    }


def auto_decide_k_for_kmeans(
    X_scaled: np.ndarray, k_range: Optional[range] = None, random_state: int = 42
) -> dict:
    """
    Rule-based default (not present in app12.py — app12.py shows the
    elbow/silhouette curve and leaves picking k entirely to the user):
    runs the same elbow/silhouette curve and picks the k with the highest
    silhouette score. Simple, explainable, and reuses the exact extracted
    compute_kmeans_elbow_curve() mechanic — no new curve-fitting logic.
    """
    n_samples = X_scaled.shape[0]
    if k_range is None:
        max_k = min(11, n_samples - 1)
        k_range = range(2, max_k + 1)
    k_values = list(k_range)
    if not k_values:
        return {
            "k": None, "silhouette": None,
            "description": "Not enough samples to cluster (need at least 3 rows).",
        }

    curve = compute_kmeans_elbow_curve(X_scaled, k_range, random_state=random_state)
    best_idx = int(np.argmax(curve["silhouettes"]))
    best_k = curve["k_values"][best_idx]
    best_sil = curve["silhouettes"][best_idx]

    return {
        "k": best_k,
        "silhouette": round(float(best_sil), 4),
        "k_range_tested": k_values,
        "curve": curve,
        "description": (
            f"Auto-selected k={best_k} — highest silhouette score ({best_sil:.4f}) "
            f"among k={k_values[0]}..{k_values[-1]} tested via the elbow/silhouette curve."
        ),
    }


def auto_decide_dbscan_params(X_scaled: np.ndarray, min_samples: Optional[int] = None) -> dict:
    """
    Rule-based default (not present in app12.py — app12.py requires the
    user to pick both eps and min_samples with sliders and no guidance):
      - min_samples: 2 x number of features (a well-known DBSCAN rule of
        thumb, floor of 5), if not given.
      - eps: the median distance to each point's min_samples-th nearest
        neighbor — a rough, data-driven starting point standing in for
        eyeballing the k-distance "elbow" plot.
    This is explicitly a starting point, not a substitute for inspecting
    the k-distance elbow plot on unusual data.
    """
    n_samples, n_features = X_scaled.shape
    if min_samples is None:
        min_samples = max(2 * n_features, 5)
    min_samples = min(min_samples, max(n_samples - 1, 1))

    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    eps = float(np.median(k_distances))

    return {
        "eps": round(eps, 4),
        "min_samples": min_samples,
        "description": (
            f"Auto-selected DBSCAN eps={eps:.4f}, min_samples={min_samples} — "
            f"min_samples follows the common '2 x number of features' rule of thumb "
            f"(floor of 5); eps is the median {min_samples}-th nearest-neighbor "
            f"distance across all points, a rough data-driven starting point rather "
            f"than a substitute for inspecting the k-distance elbow plot."
        ),
    }


def auto_train_regression(df: pd.DataFrame, features: list[str], target: str) -> dict:
    """
    Full "automatic by default" regression pass: picks the model (Linear
    vs Ridge, by multicollinearity), the train/test split, and whether to
    run cross-validation — all via the rule-based functions above — then
    trains and scores. Returns the same result shape as
    train_regression_model(), plus "log" (list of plain-English decision
    strings) and "cv" (present only if cross-validation ran).
    """
    log = []
    model_decision = auto_decide_regression_model(df, features)
    log.append(model_decision["description"])

    model_df = df[features + [target]].dropna()
    X = model_df[features].values
    y = model_df[target].values

    split_decision = _auto_decide_split_and_cv(len(model_df))
    log.append(split_decision["description"])

    results = train_regression_model(
        X, y, model_decision["model_choice"],
        alpha=model_decision.get("alpha", 1.0),
        test_size=split_decision["test_size"],
    )

    if split_decision["use_cv"]:
        cv_model = build_regression_model(model_decision["model_choice"], model_decision.get("alpha", 1.0))
        cv_results = cross_validate_regression(cv_model, X, y, cv_folds=split_decision["cv_folds"])
        results["cv"] = cv_results
        log.append(
            f"Cross-validation ({cv_results['cv_folds']}-fold): mean R² "
            f"{cv_results['cv_r2_mean']:.4f} (±{cv_results['cv_r2_std']:.4f})"
        )

    results["coefficients"] = extract_regression_coefficients(
        results["model"], model_decision["model_choice"], features
    )
    results["features"] = features
    results["target"] = target
    results["log"] = log
    return results


def auto_train_classification(df: pd.DataFrame, features: list[str], target: str) -> dict:
    """
    Full "automatic by default" classification pass: encodes the target,
    picks the model (Logistic Regression vs Random Forest, by sample
    size) and the split/CV setting, then trains and scores. Returns the
    same result shape as train_classification_model(), plus "log" and
    "cv" (present only if cross-validation ran).
    """
    log = []
    model_df = df[features + [target]].dropna()
    X = model_df[features].values
    y, class_names, label_encoder = encode_classification_target(model_df[target])

    model_decision = auto_decide_classification_model(len(model_df))
    log.append(model_decision["description"])

    split_decision = _auto_decide_split_and_cv(len(model_df))
    log.append(split_decision["description"])

    results = train_classification_model(
        X, y, class_names, model_decision["model_choice"],
        test_size=split_decision["test_size"],
    )
    results["label_encoder"] = label_encoder

    if split_decision["use_cv"]:
        cv_model = build_classification_model(model_decision["model_choice"])
        cv_results = cross_validate_classification(cv_model, X, y, cv_folds=split_decision["cv_folds"])
        results["cv"] = cv_results
        log.append(
            f"Cross-validation ({cv_results['cv_folds']}-fold): mean accuracy "
            f"{cv_results['cv_accuracy_mean']:.4f} (±{cv_results['cv_accuracy_std']:.4f})"
        )

    results["feature_importances"] = extract_classification_feature_importance(
        results["model"], model_decision["model_choice"]
    )
    results["features"] = features
    results["target"] = target
    results["log"] = log
    return results


def auto_train_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    algorithm: str = "kmeans",
    missing_strategy: str = "mean",
) -> dict:
    """
    Full "automatic by default" clustering pass:
      - algorithm="kmeans" (default): auto-picks k via silhouette score,
        then trains.
      - algorithm="dbscan": auto-picks eps/min_samples via the rule of
        thumb above, then trains.
    Returns the train_kmeans()/train_dbscan() result shape, plus "log"
    and "scaler"/"feature_cols" so the caller can assign future data via
    assign_clusters_to_new_data() afterward.
    """
    log = []
    X_scaled, scaler, cleaned_df = scale_for_clustering(df, feature_cols, missing_strategy)
    log.append(
        f"Scaled {len(feature_cols)} feature column(s) with StandardScaler "
        f"(missing values handled via '{missing_strategy}')."
    )

    if algorithm == "kmeans":
        k_decision = auto_decide_k_for_kmeans(X_scaled)
        log.append(k_decision["description"])
        if k_decision["k"] is None:
            return {"error": k_decision["description"], "log": log}
        results = train_kmeans(X_scaled, k_decision["k"])
    elif algorithm == "dbscan":
        param_decision = auto_decide_dbscan_params(X_scaled)
        log.append(param_decision["description"])
        results = train_dbscan(X_scaled, param_decision["eps"], param_decision["min_samples"])
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")

    results["scaler"] = scaler
    results["feature_cols"] = feature_cols
    results["cleaned_df"] = cleaned_df
    results["log"] = log
    return results
