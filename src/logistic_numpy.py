"""
Logistic regression implemented from first principles with NumPy.

Includes:
- Numerically stable sigmoid and log-loss.
- Closed-form gradient.
- Two optimizers built from scratch: vanilla gradient descent and Adam.
- L2 regularization (weight decay) on the weights, not the bias.
- Mini-batch and full-batch training.

No scikit-learn, no autograd. Just NumPy + math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Numerically stable primitives
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid: 1 / (1 + exp(-z)).

    For large |z|, naive ``exp`` overflows. We branch on the sign of z so
    that ``exp`` only sees non-positive arguments.
    """
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    # For z >= 0: 1 / (1 + exp(-z))
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    # For z < 0:  exp(z) / (1 + exp(z))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    """log(sigmoid(z)) computed without ever evaluating sigmoid directly.

    Uses the identity log(sigmoid(z)) = -softplus(-z), with a stable softplus.
    """
    return -np.logaddexp(0.0, -z)


def binary_cross_entropy(
    y: np.ndarray, z: np.ndarray, eps: float = 1e-12
) -> float:
    """
    Mean binary cross-entropy from logits ``z`` directly.

    Equivalent to ``-mean(y*log(p) + (1-y)*log(1-p))`` where ``p = sigmoid(z)``,
    but computed via ``logaddexp`` to avoid log(0). This is the canonical
    "binary cross entropy with logits" form.
    """
    # log(1 + exp(z)) - y * z   (per-sample loss)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


# ---------------------------------------------------------------------------
# Optimizers - implemented from scratch, no torch / no autograd.
# ---------------------------------------------------------------------------


class Optimizer:
    """Base class. Subclasses just implement ``step``."""

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GradientDescent(Optimizer):
    """Vanilla (full or mini-batch) gradient descent."""

    def __init__(self, lr: float = 0.1) -> None:
        self.lr = lr

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return params - self.lr * grad


class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2015) implemented from scratch.

    Maintains exponential moving averages of the gradient and squared
    gradient (1st and 2nd moments) with bias correction.
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        # Update biased moment estimates.
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad ** 2
        # Bias-corrected estimates.
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(name: str, **kwargs) -> Optimizer:
    """Factory used by the training script and CLI."""
    name = name.lower()
    if name in {"sgd", "gd", "gradient_descent"}:
        return GradientDescent(**kwargs)
    if name == "adam":
        return Adam(**kwargs)
    raise ValueError(f"Unknown optimizer: {name!r}")


# ---------------------------------------------------------------------------
# Logistic regression model
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    """Loss curves and optional metric histories across training."""

    loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)


class LogisticRegression:
    """
    Binary logistic regression with L2 regularization, built from NumPy.

    Parameters
    ----------
    l2 : float
        L2 penalty applied to the *weights* only (bias is not regularized).
    fit_intercept : bool
        If True, learn an intercept term. (We pack the bias into ``params``
        as the last entry to keep the optimizer interface simple.)
    """

    def __init__(self, l2: float = 0.0, fit_intercept: bool = True) -> None:
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.l2 = l2
        self.fit_intercept = fit_intercept

        # Filled in by ``fit``.
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.history_: Optional[TrainHistory] = None
        self.n_features_: Optional[int] = None

    # ------------------------------------------------------------------
    # Forward / loss / gradient
    # ------------------------------------------------------------------
    @staticmethod
    def _logits(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
        return X @ w + b

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        """Mean negative log-likelihood + L2 penalty on weights."""
        z = self._logits(X, w, b)
        nll = binary_cross_entropy(y, z)
        if self.l2 > 0.0:
            nll = nll + 0.5 * self.l2 * float(np.dot(w, w))
        return nll

    def gradient(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
    ) -> Tuple[np.ndarray, float]:
        """
        Closed-form gradient of the per-sample mean loss.

        d/dw [ -1/n * sum( y log p + (1-y) log(1-p) ) ] = X.T @ (p - y) / n
        d/db  same but summed
        """
        z = self._logits(X, w, b)
        p = sigmoid(z)
        n = X.shape[0]
        err = (p - y) / n
        grad_w = X.T @ err
        if self.l2 > 0.0:
            grad_w = grad_w + self.l2 * w
        grad_b = float(err.sum()) if self.fit_intercept else 0.0
        return grad_w, grad_b

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: Optimizer | str = "adam",
        epochs: int = 200,
        batch_size: Optional[int] = None,
        validation: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        tol: float = 1e-7,
        verbose: bool = False,
        seed: int = 0,
        on_epoch_end: Optional[Callable[[int, float], None]] = None,
        **opt_kwargs,
    ) -> "LogisticRegression":
        """
        Fit the model.

        Parameters
        ----------
        optimizer : {"sgd", "adam"} or Optimizer instance
        epochs : int
            Number of full passes through the data.
        batch_size : int or None
            None = full-batch gradient descent. Otherwise mini-batch SGD.
        validation : tuple of arrays or None
            (X_val, y_val) for tracking validation loss each epoch.
        tol : float
            Early stop when |loss change| < tol for two consecutive epochs.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._check_xy(X, y)

        n, d = X.shape
        self.n_features_ = d
        rng = np.random.default_rng(seed)

        # Pack params as a single vector so optimizers don't need to know
        # about bias separately. params = [w; b] when fit_intercept=True.
        if self.fit_intercept:
            params = np.zeros(d + 1)
        else:
            params = np.zeros(d)

        if isinstance(optimizer, str):
            optimizer = make_optimizer(optimizer, **opt_kwargs)

        history = TrainHistory()
        prev_loss = np.inf
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # ---- one epoch ---------------------------------------------
            if batch_size is None or batch_size >= n:
                grad = self._pack_grad(X, y, params)
                params = optimizer.step(params, grad)
            else:
                idx = rng.permutation(n)
                for start in range(0, n, batch_size):
                    sel = idx[start : start + batch_size]
                    grad = self._pack_grad(X[sel], y[sel], params)
                    params = optimizer.step(params, grad)

            # ---- diagnostics -------------------------------------------
            w, b = self._unpack(params)
            train_loss = self.loss(X, y, w, b)
            history.loss.append(train_loss)

            grad_full = self._pack_grad(X, y, params)
            history.grad_norm.append(float(np.linalg.norm(grad_full)))

            if validation is not None:
                Xv, yv = validation
                history.val_loss.append(self.loss(np.asarray(Xv, dtype=float),
                                                  np.asarray(yv, dtype=float).ravel(),
                                                  w, b))
            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                msg = f"epoch {epoch:4d}  loss={train_loss:.6f}"
                if validation is not None:
                    msg += f"  val_loss={history.val_loss[-1]:.6f}"
                msg += f"  |grad|={history.grad_norm[-1]:.2e}"
                print(msg)

            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss)

            # ---- convergence check -------------------------------------
            if abs(prev_loss - train_loss) < tol:
                no_improve += 1
                if no_improve >= 2:
                    if verbose:
                        print(f"converged at epoch {epoch} (delta < {tol})")
                    break
            else:
                no_improve = 0
            prev_loss = train_loss

        # Save final params.
        self.w, self.b = self._unpack(params)
        self.history_ = history
        return self

    # ------------------------------------------------------------------
    # Predict / score
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return sigmoid(self._logits(X, self.w, self.b))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy."""
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pack_grad(self, X, y, params) -> np.ndarray:
        w, b = self._unpack(params)
        gw, gb = self.gradient(X, y, w, b)
        if self.fit_intercept:
            return np.concatenate([gw, [gb]])
        return gw

    def _unpack(self, params) -> Tuple[np.ndarray, float]:
        if self.fit_intercept:
            return params[:-1], float(params[-1])
        return params, 0.0

    def _check_xy(self, X, y):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D and match X length")
        unique = np.unique(y)
        if not set(unique.tolist()).issubset({0.0, 1.0}):
            raise ValueError(f"y must contain only 0/1, got {unique}")

    def _check_fitted(self):
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")


# ---------------------------------------------------------------------------
# Gradient check utility - used by tests to verify the analytic gradient
# matches a numerical gradient.
# ---------------------------------------------------------------------------


def numerical_gradient(
    loss_fn: Callable[[np.ndarray], float], params: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """Two-sided finite-difference gradient. Slow but accurate."""
    grad = np.zeros_like(params)
    for i in range(params.size):
        orig = params[i]
        params[i] = orig + h
        f_plus = loss_fn(params)
        params[i] = orig - h
        f_minus = loss_fn(params)
        params[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad
