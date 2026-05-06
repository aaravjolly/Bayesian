"""
Data utilities: synthetic generators, train/val/test split, standardization.

Kept dependency-free (just NumPy) so both the from-scratch and Bayesian
training pipelines share the same data layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


@dataclass
class SyntheticData:
    X: np.ndarray
    y: np.ndarray
    true_w: np.ndarray
    true_b: float
    feature_names: list


def make_logistic_data(
    n_samples: int = 1000,
    n_features: int = 4,
    class_sep: float = 1.5,
    noise: float = 0.0,
    seed: int = 0,
) -> SyntheticData:
    """
    Generate data drawn from a true logistic model.

    For each sample:
        x ~ N(0, I)
        z = w.x + b + noise
        p = sigmoid(z)
        y ~ Bernoulli(p)

    Parameters
    ----------
    class_sep : float
        Multiplier on the true weight magnitude. Larger -> easier task.
    noise : float
        Optional Gaussian noise added to the logits before sampling. Lets us
        simulate label noise / model misspecification.
    """
    rng = np.random.default_rng(seed)

    # True weights drawn from a heavy-tailed-ish distribution so feature
    # importance varies.
    true_w = rng.normal(scale=class_sep, size=n_features)
    true_b = float(rng.normal(scale=0.5))

    X = rng.normal(size=(n_samples, n_features))
    logits = X @ true_w + true_b
    if noise > 0:
        logits = logits + rng.normal(scale=noise, size=n_samples)
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n_samples) < p).astype(int)

    feature_names = [f"x{i}" for i in range(n_features)]
    return SyntheticData(X=X, y=y, true_w=true_w, true_b=true_b,
                         feature_names=feature_names)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, ...]:
    """Three-way random split. Returns X_tr, X_val, X_te, y_tr, y_val, y_te."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_te = int(round(n * test_frac))
    n_va = int(round(n * val_frac))
    te_idx = idx[:n_te]
    va_idx = idx[n_te : n_te + n_va]
    tr_idx = idx[n_te + n_va :]
    return (
        X[tr_idx], X[va_idx], X[te_idx],
        y[tr_idx], y[va_idx], y[te_idx],
    )


# ---------------------------------------------------------------------------
# Standardization (fit on train, apply to all splits)
# ---------------------------------------------------------------------------


class Standardizer:
    """Subtract train mean, divide by train std. No sklearn dependency."""

    def __init__(self) -> None:
        self.mean_: np.ndarray = None
        self.scale_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Avoid division by zero for constant columns.
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer not fitted")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()))


def log_loss(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Mean binary cross-entropy from probabilities."""
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=float).ravel()
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]] - matches sklearn's default ordering."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def roc_auc(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """
    AUC via the Mann-Whitney U formulation.

    AUC = P(score(positive) > score(negative)).
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(p_pred, dtype=float).ravel()
    pos = p[y_true == 1]
    neg = p[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Compute U statistic via ranks.
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, p.size + 1)
    # Tie correction with average rank.
    # (For our continuous probabilities ties are unlikely; this still handles them.)
    sum_ranks_pos = ranks[y_true == 1].sum()
    n_pos = pos.size
    n_neg = neg.size
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))
