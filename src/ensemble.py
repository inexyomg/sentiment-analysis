import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, log_loss
from scipy.special import softmax
from scipy.optimize import minimize_scalar


# ──────────────────────────────────────────────
# Basic ensembles
# ──────────────────────────────────────────────

def hard_voting(predictions: List[np.ndarray]) -> np.ndarray:
    """Majority vote across model hard predictions."""
    stacked = np.stack(predictions, axis=1)
    n_classes = int(max(p.max() for p in predictions)) + 1
    return np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_classes).argmax(),
        axis=1, arr=stacked,
    )


def soft_voting(probabilities: List[np.ndarray],
                weights: Optional[List[float]] = None) -> np.ndarray:
    """Average probability distributions, optionally weighted."""
    return np.argmax(soft_voting_proba(probabilities, weights), axis=-1)


def soft_voting_proba(probabilities: List[np.ndarray],
                      weights: Optional[List[float]] = None) -> np.ndarray:
    """Weighted-average probability matrix (no argmax) — reused by inference."""
    if weights is None:
        weights = [1.0] * len(probabilities)
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return sum(wi * p for wi, p in zip(w, probabilities))


def weighted_averaging(probabilities: List[np.ndarray],
                       val_f1_scores: List[float]) -> np.ndarray:
    """Weight each model by its validation F1-macro score."""
    return soft_voting(probabilities, weights=val_f1_scores)


# ──────────────────────────────────────────────
# Confidence calibration
# ──────────────────────────────────────────────

def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """
    Fit a single temperature T by minimizing NLL on validation logits.
    Returns T; apply with softmax(logits / T).
    """
    def nll(t):
        if t <= 0:
            return 1e9
        probs = softmax(val_logits / t, axis=-1)
        return log_loss(val_labels, probs, labels=list(range(val_logits.shape[1])))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def temperature_scaling(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling. Use fit_temperature() to choose T per model."""
    return softmax(logits / temperature, axis=-1)


# ──────────────────────────────────────────────
# Stacking (leak-free)
# ──────────────────────────────────────────────

def stacking_ensemble(
    meta_train_probs: List[np.ndarray],
    meta_train_labels: np.ndarray,
    test_probs: List[np.ndarray],
    meta_learner: str = "logistic",
    cv: int = 5,
) -> Tuple[np.ndarray, object]:
    """
    Stacking with a meta-learner over concatenated base-model probabilities.

    IMPORTANT — avoid leakage: meta_train_probs MUST come from a split the base
    models did NOT train on (use the saved validation probabilities, val_probs.npy,
    not the training set). The meta-learner is then applied to test_probs.

    Args:
        meta_train_probs: list of (n_meta, n_classes) arrays — base model probs on a
                          held-out split (validation), per model.
        meta_train_labels: labels for that held-out split.
        test_probs: list of (n_test, n_classes) arrays — base model probs on test.
        meta_learner: 'logistic' | 'svm' | 'xgboost' | 'gradient_boosting'
        cv: folds for the cross-validation diagnostic on the meta-train set.

    Returns:
        (test_predictions, fitted_meta_learner)
    """
    X_meta = np.hstack(meta_train_probs)
    X_test = np.hstack(test_probs)

    if meta_learner == "logistic":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    elif meta_learner == "svm":
        clf = SVC(probability=True, kernel="rbf", random_state=42)
    elif meta_learner == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        n_classes = len(np.unique(meta_train_labels))
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
    elif meta_learner == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown meta_learner: {meta_learner}")

    n_splits = min(cv, np.min(np.bincount(meta_train_labels)))
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_meta, meta_train_labels, cv=skf, scoring="f1_macro")
        print(f"  [{meta_learner}] meta-CV F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")

    clf.fit(X_meta, meta_train_labels)
    return clf.predict(X_test), clf


# ──────────────────────────────────────────────
# Comparison helper
# ──────────────────────────────────────────────

def evaluate_ensemble(y_true: np.ndarray,
                      ensemble_results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """DataFrame comparing accuracy, F1-macro, F1-weighted per ensemble method."""
    rows = []
    for method, y_pred in ensemble_results.items():
        rows.append({
            "method": method,
            "accuracy":    accuracy_score(y_true, y_pred),
            "f1_macro":    f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        })
    return pd.DataFrame(rows).set_index("method").sort_values("f1_macro", ascending=False)


# ──────────────────────────────────────────────
# Save / load ensemble configuration
# ──────────────────────────────────────────────

def save_ensemble(
    save_dir: str,
    method: str,
    model_dirs: List[str],
    weights: Optional[List[float]] = None,
    meta_clf: Optional[Any] = None,
    label_names: Optional[List[str]] = None,
    metrics: Optional[Dict] = None,
) -> None:
    """
    Persist an ensemble so it can be reloaded for inference.

    Saves:
      - ensemble_config.json  — method, model paths, weights, label names, metrics
      - meta_learner.pkl      — fitted sklearn meta-learner (stacking only)
    """
    os.makedirs(save_dir, exist_ok=True)
    config: Dict = {
        "method": method,
        "model_dirs": list(model_dirs),
        "weights": list(weights) if weights is not None else None,
        "label_names": label_names,
        "metrics": metrics or {},
    }
    with open(os.path.join(save_dir, "ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    if meta_clf is not None:
        with open(os.path.join(save_dir, "meta_learner.pkl"), "wb") as f:
            pickle.dump(meta_clf, f)
    print(f"Ансамбль сохранён в {save_dir}")


def load_ensemble_config(save_dir: str) -> Tuple[Dict, Optional[Any]]:
    """
    Load a previously saved ensemble.

    Returns:
        (config_dict, meta_clf_or_None)
    """
    with open(os.path.join(save_dir, "ensemble_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    meta_clf = None
    clf_path = os.path.join(save_dir, "meta_learner.pkl")
    if os.path.exists(clf_path):
        with open(clf_path, "rb") as f:
            meta_clf = pickle.load(f)
    return config, meta_clf
