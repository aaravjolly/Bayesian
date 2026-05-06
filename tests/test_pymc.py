"""
Tests for the Bayesian (PyMC) implementation.

These tests are skipped automatically if PyMC isn't installed, so the
suite still runs in lean environments.
"""

import numpy as np
import pytest

pymc = pytest.importorskip("pymc")

from src.data import accuracy, make_logistic_data
from src.logistic_pymc import BayesianLogisticRegression


@pytest.fixture(scope="module")
def fitted_model():
    """A small Bayesian model fit once and reused across tests."""
    data = make_logistic_data(n_samples=300, n_features=3, seed=0)
    model = BayesianLogisticRegression(
        prior_sigma=2.5,
        draws=400,
        tune=400,
        chains=2,
        target_accept=0.9,
        random_seed=42,
    )
    model.fit(data.X, data.y, progressbar=False)
    return model, data


class TestBayesianFit:
    def test_posterior_samples_have_correct_shape(self, fitted_model):
        model, data = fitted_model
        # 2 chains x 400 draws = 800 samples
        assert model._w_samples.shape == (800, 3)
        assert model._b_samples.shape == (800,)

    def test_predict_proba_in_unit_interval(self, fitted_model):
        model, data = fitted_model
        p = model.predict_proba(data.X)
        assert p.shape == (data.X.shape[0],)
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_predict_proba_return_samples(self, fitted_model):
        model, data = fitted_model
        samples = model.predict_proba(data.X[:10], return_samples=True)
        # (n_samples, n_obs)
        assert samples.shape == (800, 10)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_credible_interval_brackets_mean(self, fitted_model):
        model, data = fitted_model
        mean = model.predict_proba(data.X)
        lo, hi = model.credible_interval(data.X)
        assert np.all(lo <= mean)
        assert np.all(mean <= hi)
        assert np.all(hi - lo >= 0)

    def test_summary_recovers_signs(self, fitted_model):
        """Posterior means should at least have the right sign as true weights
        for synthetic data with a reasonable signal."""
        model, data = fitted_model
        s = model.summary()
        same_sign = np.sign(s.mean) == np.sign(data.true_w)
        # Allow one feature to flip on small data; require majority correct.
        assert same_sign.sum() >= len(data.true_w) - 1

    def test_diagnostics_report_rhat(self, fitted_model):
        model, _ = fitted_model
        diag = model.diagnostics()
        assert "r_hat_max" in diag
        assert "ess_bulk_min" in diag
        # Sanity: with 2 chains and 400 draws, R-hat should be close to 1.
        assert diag["r_hat_max"] < 1.1

    def test_predicts_better_than_chance(self, fitted_model):
        model, data = fitted_model
        yhat = model.predict(data.X)
        assert accuracy(data.y, yhat) > 0.7


class TestErrors:
    def test_must_fit_before_predict(self):
        m = BayesianLogisticRegression(draws=10, tune=10, chains=1)
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((3, 3)))

    def test_rejects_non_binary(self):
        m = BayesianLogisticRegression(draws=10, tune=10, chains=1)
        with pytest.raises(ValueError):
            m.fit(np.zeros((10, 2)), np.array([0, 1, 2] * 3 + [0]))
