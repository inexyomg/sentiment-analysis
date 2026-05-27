import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax


# ──────────────────────────────────────────────
# Basic ensembles
# ──────────────────────────────────────────────

def hard_voting(predictions: List[np.ndarray]) -> np.ndarray:
    """Majority vote across model hard predictions."""
    stacked = np.stack(predictions, axis=1)   # (n_samples, n_models)
    n_classes = max(p.max() for p in predictions) + 1
    return np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_classes).argmax(),
        axis=1, arr=stacked,
    )


def soft_voting(
    probabilities: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Average probability distributions, optionally weighted."""
    if weights is None:
        weights = [1.0] * len(probabilities)
    w = np.array(weights, dtype=float)
    w /= w.sum()
    avg = sum(wi * p for wi, p in zip(w, probabilities))
    return np.argmax(avg, axis=-1)


def weighted_averaging(
    probabilities: List[np.ndarray],
    val_f1_scores: List[float],
) -> np.ndarray:
    """Weight each model by its validation F1-macro score."""
    return soft_voting(probabilities, weights=val_f1_scores)


def temperature_scaling(logits: np.ndarray, temperature: float = 1.5) -> np.ndarray:
    """Calibrate confidence via temperature before softmax."""
    return softmax(logits / temperature, axis=-1)


# ──────────────────────────────────────────────
# Stacking
# ──────────────────────────────────────────────

def stacking_ensemble(
    train_probs: List[np.ndarray],
    train_labels: np.ndarray,
    test_probs: List[np.ndarray],
    meta_learner: str = "logistic",
    cv: int = 5,
) -> Tuple[np.ndarray, object]:
    """
    Stacking: concatenate model probability outputs as features,
    then fit a meta-learner.

    Args:
        train_probs: list of (n_train, n_classes) arrays from each model
        train_labels: ground-truth labels for training set
        test_probs: list of (n_test, n_classes) arrays
        meta_learner: 'logistic' | 'svm'
        cv: folds for cross-validation diagnostic

    Returns:
        (test_predictions, fitted_meta_learner)
    """
    X_train = np.hstack(train_probs)
    X_test  = np.hstack(test_probs)

    if meta_learner == "logistic":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    elif meta_learner == "svm":
        clf = SVC(probability=True, kernel="rbf", random_state=42)
    else:
        raise ValueError(f"Unknown meta_learner: {meta_learner}")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, train_labels, cv=skf, scoring="f1_macro")
    print(f"  [{meta_learner}] CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    clf.fit(X_train, train_labels)
    return clf.predict(X_test), clf


# ──────────────────────────────────────────────
# Comparison helper
# ──────────────────────────────────────────────

def evaluate_ensemble(
    y_true: np.ndarray,
    ensemble_results: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return a DataFrame comparing accuracy, F1-macro, F1-weighted for each method."""
    rows = []
    for method, y_pred in ensemble_results.items():
        rows.append({
            "method": method,
            "accuracy":     accuracy_score(y_true, y_pred),
            "f1_macro":     f1_score(y_true, y_pred, average="macro"),
            "f1_weighted":  f1_score(y_true, y_pred, average="weighted"),
        })
    return pd.DataFrame(rows).set_index("method").sort_values("f1_macro", ascending=False)
