"""
Pytest-free smoke test that validates the entire from-scratch pipeline.

Run with:  python scripts/smoke_test.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    Standardizer,
    accuracy,
    confusion_matrix,
    log_loss,
    make_logistic_data,
    roc_auc,
    train_val_test_split,
)
from src.logistic_numpy import (
    Adam,
    GradientDescent,
    LogisticRegression,
    binary_cross_entropy,
    log_sigmoid,
    numerical_gradient,
    sigmoid,
)


def section(name):
    print(f"\n=== {name} " + "=" * (60 - len(name)))


def main():
    fail = []

    # ------------------------------------------------------------------
    section("numerically stable primitives")
    z = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s)), "sigmoid produced non-finite values"
    assert s[0] == 0.0 and s[-1] == 1.0
    print("  sigmoid stable across [-1000, 1000] OK")

    ls = log_sigmoid(np.array([-1000.0]))
    assert ls[0] == -1000.0
    print("  log_sigmoid avoids underflow at z=-1000 OK")

    # BCE matches the textbook formula on moderate inputs.
    rng = np.random.default_rng(0)
    z = rng.normal(size=100)
    y = rng.integers(0, 2, size=100).astype(float)
    p = sigmoid(z)
    manual = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert abs(binary_cross_entropy(y, z) - manual) < 1e-10
    print("  binary_cross_entropy matches manual formula OK")

    # ------------------------------------------------------------------
    section("gradient correctness (analytic vs finite diff)")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = rng.integers(0, 2, size=50).astype(float)
    model = LogisticRegression(l2=0.1)
    params = rng.normal(scale=0.5, size=5)

    def loss_fn(p):
        w, b = p[:-1], float(p[-1])
        return model.loss(X, y, w, b)

    w, b = params[:-1], float(params[-1])
    gw, gb = model.gradient(X, y, w, b)
    analytic = np.concatenate([gw, [gb]])
    numeric = numerical_gradient(loss_fn, params.copy())
    err = np.max(np.abs(analytic - numeric))
    print(f"  max |analytic - numeric| = {err:.2e}")
    if err > 1e-5:
        fail.append(f"gradient check failed: max err = {err}")
    else:
        print("  gradient check OK")

    # ------------------------------------------------------------------
    section("optimizers")
    # Quadratic bowl: minimum at x = 3.
    opt = GradientDescent(lr=0.1)
    x = np.zeros(3)
    for _ in range(200):
        x = opt.step(x, x - 3.0)
    assert np.allclose(x, 3.0, atol=1e-3), x
    print("  GD converges to x=3 OK")

    opt = Adam(lr=0.1)
    x = np.zeros(3)
    for _ in range(500):
        x = opt.step(x, x - 3.0)
    assert np.allclose(x, 3.0, atol=1e-2), x
    print("  Adam converges to x=3 OK")

    # ------------------------------------------------------------------
    section("end-to-end: full-batch GD")
    data = make_logistic_data(n_samples=600, n_features=4, seed=0)
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        data.X, data.y, seed=0
    )
    scaler = Standardizer().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = (scaler.transform(X) for X in (X_tr, X_va, X_te))

    model = LogisticRegression(l2=0.0)
    model.fit(X_tr_s, y_tr, optimizer="sgd", epochs=200, lr=0.1,
              validation=(X_va_s, y_va), verbose=False)
    losses = np.array(model.history_.loss)
    if not np.all(np.diff(losses) <= 1e-6):
        fail.append("loss not monotonically decreasing in full-batch GD")
    else:
        print(f"  loss monotonic; final={losses[-1]:.4f}")

    p_te = model.predict_proba(X_te_s)
    yhat = (p_te >= 0.5).astype(int)
    acc, ll, auc = accuracy(y_te, yhat), log_loss(y_te, p_te), roc_auc(y_te, p_te)
    print(f"  test acc={acc:.3f}  log_loss={ll:.3f}  AUC={auc:.3f}")
    if auc < 0.75:
        fail.append(f"GD test AUC too low: {auc}")

    # ------------------------------------------------------------------
    section("end-to-end: Adam, mini-batch")
    model_adam = LogisticRegression(l2=0.01)
    model_adam.fit(X_tr_s, y_tr, optimizer="adam", epochs=200, batch_size=32,
                   lr=0.05, validation=(X_va_s, y_va), verbose=False)
    p_te = model_adam.predict_proba(X_te_s)
    yhat = (p_te >= 0.5).astype(int)
    acc = accuracy(y_te, yhat)
    auc = roc_auc(y_te, p_te)
    print(f"  test acc={acc:.3f}  AUC={auc:.3f}")
    if auc < 0.75:
        fail.append(f"Adam test AUC too low: {auc}")

    # ------------------------------------------------------------------
    section("L2 regularization shrinks weights")
    no_l2 = LogisticRegression(l2=0.0)
    no_l2.fit(X_tr_s, y_tr, optimizer="adam", epochs=200, lr=0.05, verbose=False)
    big_l2 = LogisticRegression(l2=10.0)
    big_l2.fit(X_tr_s, y_tr, optimizer="adam", epochs=200, lr=0.05, verbose=False)
    n0, n1 = np.linalg.norm(no_l2.w), np.linalg.norm(big_l2.w)
    print(f"  ||w|| no L2 = {n0:.3f}  vs  L2=10 -> {n1:.3f}")
    if n1 >= n0:
        fail.append("L2 did not shrink weights")

    # ------------------------------------------------------------------
    section("metrics")
    cm = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert cm.tolist() == [[1, 1], [1, 2]], cm
    print(f"  confusion_matrix OK -> {cm.tolist()}")

    rng = np.random.default_rng(0)
    y_rand = rng.integers(0, 2, size=2000)
    p_rand = rng.uniform(size=2000)
    auc = roc_auc(y_rand, p_rand)
    if abs(auc - 0.5) > 0.05:
        fail.append(f"random AUC should be ~0.5, got {auc}")
    else:
        print(f"  random AUC = {auc:.3f} (~0.5 expected) OK")

    perfect_auc = roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect_auc == 1.0
    print("  perfect AUC = 1.0 OK")

    # ------------------------------------------------------------------
    section("error handling")
    try:
        LogisticRegression().predict(np.zeros((3, 3)))
    except RuntimeError:
        print("  predict-before-fit raises RuntimeError OK")
    else:
        fail.append("expected RuntimeError on predict before fit")

    try:
        LogisticRegression().fit(np.zeros((10, 2)), np.array([0, 1, 2] * 3 + [0]))
    except ValueError:
        print("  non-binary labels rejected OK")
    else:
        fail.append("expected ValueError on non-binary labels")

    # ------------------------------------------------------------------
    section("summary")
    if fail:
        print("FAILURES:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
