import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
        meta_learner: 'logistic' | 'svm'
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
