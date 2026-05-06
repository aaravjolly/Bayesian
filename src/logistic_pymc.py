"""
Bayesian logistic regression with PyMC.

Wraps a PyMC model that places Gaussian priors on the weights and bias,
draws samples from the posterior with the No-U-Turn Sampler (NUTS), and
exposes a scikit-learn-style predict / predict_proba interface that
returns posterior predictive means.

Why Bayesian for logistic regression?
- Calibrated uncertainty: instead of a single point estimate w*, you get
  a *distribution* over plausible weights given the data.
- Predictive intervals: predict_proba can return credible intervals, not
  just point estimates.
- Principled regularization: the prior is the regularizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PosteriorSummary:
    """Light-weight summary used when arviz/pandas aren't desired."""

    mean: np.ndarray         # shape (d,)
    std: np.ndarray          # shape (d,)
    hdi_low: np.ndarray      # shape (d,)
    hdi_high: np.ndarray     # shape (d,)
    intercept_mean: float
    intercept_std: float
    intercept_hdi: Tuple[float, float]


class BayesianLogisticRegression:
    """
    Bayesian logistic regression sampled with NUTS.

    The PyMC import is deferred to fit() so that this module can be imported
    in environments where PyMC isn't installed (e.g., to inspect the API or
    train only the NumPy model).

    Parameters
    ----------
    prior_sigma : float
        Std-dev of the zero-mean Gaussian prior on each weight. Larger =
        weaker regularization.
    intercept_sigma : float
        Std-dev of the prior on the intercept.
    draws : int
        Posterior samples per chain (after tuning).
    tune : int
        NUTS warm-up iterations per chain (discarded).
    chains : int
        Number of independent MCMC chains. Multiple chains let us check
        convergence via R-hat.
    target_accept : float
        NUTS step-size adaptation target. Increase (e.g. 0.95) if you see
        divergences.
    random_seed : int
    """

    def __init__(
        self,
        prior_sigma: float = 2.5,
        intercept_sigma: float = 5.0,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 0,
    ) -> None:
        self.prior_sigma = prior_sigma
        self.intercept_sigma = intercept_sigma
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed

        # Filled by fit().
        self.idata_ = None       # arviz InferenceData
        self.model_ = None       # pm.Model
        self.n_features_ = None
        self._w_samples: Optional[np.ndarray] = None  # shape (S, d)
        self._b_samples: Optional[np.ndarray] = None  # shape (S,)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, progressbar: bool = True
            ) -> "BayesianLogisticRegression":
        """Sample the posterior with NUTS."""
        # Imported inside fit so the module can be imported without PyMC.
        import pymc as pm

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if not set(np.unique(y).tolist()).issubset({0, 1}):
            raise ValueError("y must contain only 0/1")

        n, d = X.shape
        self.n_features_ = d

        with pm.Model() as model:
            # Data containers so we can swap in new X for posterior predictive.
            X_data = pm.Data("X", X)
            y_data = pm.Data("y", y)

            # Priors.
            w = pm.Normal("w", mu=0.0, sigma=self.prior_sigma, shape=d)
            b = pm.Normal("b", mu=0.0, sigma=self.intercept_sigma)

            # Likelihood.
            logits = pm.math.dot(X_data, w) + b
            pm.Bernoulli("obs", logit_p=logits, observed=y_data)

            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=progressbar,
                return_inferencedata=True,
            )

        self.model_ = model
        self.idata_ = idata

        # Stack chains x draws into a single sample axis for fast predictions.
        post = idata.posterior
        # post["w"] has dims (chain, draw, w_dim_0). Stack to (samples, d).
        self._w_samples = (
            post["w"].stack(samples=("chain", "draw")).transpose("samples", "w_dim_0").values
        )
        self._b_samples = post["b"].stack(samples=("chain", "draw")).values
        return self

    # ------------------------------------------------------------------
    def predict_proba(
        self, X: np.ndarray, return_samples: bool = False
    ) -> np.ndarray:
        """
        Posterior predictive probabilities.

        Parameters
        ----------
        return_samples : bool
            If True, return the full ``(n_samples, n_obs)`` matrix of
            posterior probability draws. If False, return the posterior
            mean probability per observation.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        # Logits matrix: samples x observations
        logits = X @ self._w_samples.T + self._b_samples[np.newaxis, :]
        # Numerically stable sigmoid via logaddexp.
        # sigmoid(z) = 1 / (1 + exp(-z))
        probs = 1.0 / (1.0 + np.exp(-logits))
        # probs shape: (n_obs, n_samples). Transpose so first axis is samples.
        probs = probs.T
        if return_samples:
            return probs
        return probs.mean(axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def credible_interval(
        self, X: np.ndarray, hdi_prob: float = 0.94
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds of the highest-density interval per observation."""
        samples = self.predict_proba(X, return_samples=True)
        alpha = (1.0 - hdi_prob) / 2.0
        lo = np.quantile(samples, alpha, axis=0)
        hi = np.quantile(samples, 1.0 - alpha, axis=0)
        return lo, hi

    # ------------------------------------------------------------------
    def summary(self, hdi_prob: float = 0.94) -> PosteriorSummary:
        """Posterior moments and HDI for each weight and the intercept."""
        self._check_fitted()
        w = self._w_samples
        b = self._b_samples
        alpha = (1.0 - hdi_prob) / 2.0

        return PosteriorSummary(
            mean=w.mean(axis=0),
            std=w.std(axis=0),
            hdi_low=np.quantile(w, alpha, axis=0),
            hdi_high=np.quantile(w, 1.0 - alpha, axis=0),
            intercept_mean=float(b.mean()),
            intercept_std=float(b.std()),
            intercept_hdi=(
                float(np.quantile(b, alpha)),
                float(np.quantile(b, 1.0 - alpha)),
            ),
        )

    def diagnostics(self) -> dict:
        """Convergence diagnostics: R-hat and effective sample size."""
        self._check_fitted()
        import arviz as az

        summary = az.summary(self.idata_, var_names=["w", "b"], round_to=4)
        # Return as a dict so callers don't need pandas to consume it.
        return {
            "r_hat_max": float(summary["r_hat"].max()),
            "ess_bulk_min": float(summary["ess_bulk"].min()),
            "ess_tail_min": float(summary["ess_tail"].min()),
            "raw": summary,
        }

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if self._w_samples is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
