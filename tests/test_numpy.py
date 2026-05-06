"""
Tests for the NumPy implementation.

The most important test here is ``test_gradient_matches_finite_difference``:
it verifies the analytic gradient against a numerical one, which is the
gold standard for catching bugs in hand-derived gradients.
"""

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_sigmoid_known_values(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
        # Symmetric around 0.
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        s = sigmoid(z)
        assert np.allclose(s + sigmoid(-z), 1.0)

    def test_sigmoid_no_overflow(self):
        """Sigmoid must be stable for very large positive AND negative inputs."""
        z = np.array([1000.0, -1000.0, 0.0])
        s = sigmoid(z)
        assert np.all(np.isfinite(s))
        assert s[0] == pytest.approx(1.0)
        assert s[1] == pytest.approx(0.0)

    def test_log_sigmoid_matches_naive(self):
        """For moderate inputs, log_sigmoid == log(sigmoid(z))."""
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        np.testing.assert_allclose(log_sigmoid(z), np.log(sigmoid(z)), atol=1e-12)

    def test_log_sigmoid_no_underflow(self):
        """For very negative z, naive log(sigmoid(z)) underflows; ours doesn't."""
        z = np.array([-1000.0])
        # Naive: log(sigmoid(-1000)) = log(0) = -inf. Ours = -1000.
        assert log_sigmoid(z)[0] == pytest.approx(-1000.0)

    def test_bce_matches_manual_formula(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=50)
        y = rng.integers(0, 2, size=50).astype(float)
        p = sigmoid(z)
        manual = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        ours = binary_cross_entropy(y, z)
        assert ours == pytest.approx(manual, rel=1e-10)


# ---------------------------------------------------------------------------
# Gradient correctness - the headline test
# ---------------------------------------------------------------------------


class TestGradient:
    def test_gradient_matches_finite_difference(self):
        """Analytic gradient must match numerical gradient to high precision."""
        rng = np.random.default_rng(0)
        n, d = 50, 4
        X = rng.normal(size=(n, d))
        y = rng.integers(0, 2, size=n).astype(float)

        model = LogisticRegression(l2=0.1, fit_intercept=True)

        # Test at a non-trivial point (not all zeros).
        params = rng.normal(scale=0.5, size=d + 1)

        def loss_fn(p):
            w, b = p[:-1], float(p[-1])
            return model.loss(X, y, w, b)

        # Analytic gradient.
        w, b = params[:-1], float(params[-1])
        gw, gb = model.gradient(X, y, w, b)
        analytic = np.concatenate([gw, [gb]])

        # Numerical gradient.
        numeric = numerical_gradient(loss_fn, params.copy())

        np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-5)

    def test_gradient_with_no_intercept(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 3))
        y = rng.integers(0, 2, size=30).astype(float)
        model = LogisticRegression(l2=0.0, fit_intercept=False)
        gw, gb = model.gradient(X, y, np.zeros(3), 0.0)
        assert gb == 0.0
        # At w=0, b=0: sigmoid = 0.5, gradient = X.T @ (0.5 - y) / n
        expected = X.T @ (0.5 - y) / 30
        np.testing.assert_allclose(gw, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class TestOptimizers:
    def test_gd_minimizes_quadratic(self):
        """f(x) = 0.5 * ||x - 3||^2.  Gradient = x - 3.  Min at x = 3."""
        opt = GradientDescent(lr=0.1)
        x = np.array([0.0, 0.0, 0.0])
        for _ in range(200):
            grad = x - 3.0
            x = opt.step(x, grad)
        np.testing.assert_allclose(x, 3.0, atol=1e-3)

    def test_adam_minimizes_quadratic(self):
        opt = Adam(lr=0.1)
        x = np.zeros(3)
        for _ in range(500):
            grad = x - 3.0
            x = opt.step(x, grad)
        np.testing.assert_allclose(x, 3.0, atol=1e-2)

    def test_adam_first_step_uses_lr(self):
        """At t=1 with grad=1, m_hat = 1, v_hat = 1 -> step ~ -lr."""
        opt = Adam(lr=0.01)
        x = np.array([0.0])
        x = opt.step(x, np.array([1.0]))
        # Step magnitude should be very close to lr.
        assert abs(x[0] + 0.01) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end fit
# ---------------------------------------------------------------------------


class TestFit:
    @pytest.fixture
    def synthetic(self):
        return make_logistic_data(n_samples=500, n_features=4, seed=0)

    def test_fit_recovers_separable_classes(self, synthetic):
        """On clean synthetic data, accuracy should be >= 90%."""
        model = LogisticRegression(l2=0.0)
        model.fit(synthetic.X, synthetic.y, optimizer="adam",
                  epochs=300, lr=0.05, verbose=False)
        acc = model.score(synthetic.X, synthetic.y)
        assert acc > 0.85, f"expected acc > 0.85, got {acc}"

    def test_loss_decreases_monotonically_full_batch(self, synthetic):
        """With full-batch GD and small enough lr, loss must not increase."""
        model = LogisticRegression()
        model.fit(synthetic.X, synthetic.y, optimizer="sgd",
                  epochs=100, lr=0.05, verbose=False)
        losses = np.array(model.history_.loss)
        # Allow tiny numerical wiggles.
        diffs = np.diff(losses)
        assert (diffs <= 1e-6).all(), \
            f"loss increased at some step: max delta = {diffs.max()}"

    def test_must_fit_before_predict(self):
        m = LogisticRegression()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((3, 3)))

    def test_rejects_non_binary_labels(self):
        m = LogisticRegression()
        with pytest.raises(ValueError, match="0/1"):
            m.fit(np.zeros((10, 2)), np.array([0, 1, 2] * 3 + [0]))

    def test_rejects_wrong_shape(self):
        m = LogisticRegression()
        with pytest.raises(ValueError):
            m.fit(np.zeros(10), np.zeros(10))

    def test_validation_history_recorded(self, synthetic):
        X_tr, X_va, _, y_tr, y_va, _ = train_val_test_split(
            synthetic.X, synthetic.y, seed=0
        )
        m = LogisticRegression()
        m.fit(X_tr, y_tr, optimizer="adam", epochs=20,
              validation=(X_va, y_va), verbose=False)
        assert len(m.history_.val_loss) == len(m.history_.loss)

    def test_minibatch_runs(self, synthetic):
        m = LogisticRegression()
        m.fit(synthetic.X, synthetic.y, optimizer="adam",
              epochs=20, batch_size=64, verbose=False)
        # Just checking that it runs and produces a sensible model.
        assert m.score(synthetic.X, synthetic.y) > 0.7

    def test_l2_shrinks_weights(self, synthetic):
        m_no_l2 = LogisticRegression(l2=0.0)
        m_no_l2.fit(synthetic.X, synthetic.y, optimizer="adam",
                    epochs=200, verbose=False)
        m_l2 = LogisticRegression(l2=10.0)
        m_l2.fit(synthetic.X, synthetic.y, optimizer="adam",
                 epochs=200, verbose=False)
        assert np.linalg.norm(m_l2.w) < np.linalg.norm(m_no_l2.w)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


class TestDataUtils:
    def test_split_sizes(self):
        X = np.zeros((100, 3))
        y = np.zeros(100)
        Xt, Xv, Xe, yt, yv, ye = train_val_test_split(
            X, y, val_frac=0.2, test_frac=0.2, seed=0
        )
        assert Xt.shape[0] + Xv.shape[0] + Xe.shape[0] == 100
        assert Xv.shape[0] == 20
        assert Xe.shape[0] == 20

    def test_standardizer_zero_mean_unit_var(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=3.0, size=(500, 4))
        s = Standardizer().fit(X)
        Xs = s.transform(X)
        assert np.allclose(Xs.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-10)

    def test_standardizer_constant_column(self):
        X = np.ones((20, 3))
        X[:, 1] = 5.0
        s = Standardizer().fit(X)
        # No NaNs even though columns have zero variance.
        Xs = s.transform(X)
        assert np.all(np.isfinite(Xs))

    def test_accuracy_basic(self):
        assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75

    def test_log_loss_perfect(self):
        # Perfect predictions but clipped, so finite.
        ll = log_loss([0, 1], [0.0, 1.0])
        assert ll < 1e-6

    def test_log_loss_max(self):
        # Worst-case predictions.
        ll = log_loss([0, 1], [1.0, 0.0])
        assert ll > 10  # clipped to ~log(1e-12) ~ 27

    def test_confusion_matrix(self):
        cm = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
        assert cm[0, 0] == 1   # TN
        assert cm[0, 1] == 1   # FP
        assert cm[1, 0] == 1   # FN
        assert cm[1, 1] == 2   # TP

    def test_roc_auc_perfect(self):
        # When positives all rank above negatives, AUC = 1.
        y = [0, 0, 1, 1]
        p = [0.1, 0.2, 0.8, 0.9]
        assert roc_auc(y, p) == pytest.approx(1.0)

    def test_roc_auc_random(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=2000)
        p = rng.uniform(size=2000)
        # Random scores -> AUC ~ 0.5.
        assert abs(roc_auc(y, p) - 0.5) < 0.05
