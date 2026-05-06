"""
Compare the from-scratch NumPy logistic regression against the Bayesian
PyMC version on the same dataset.

This is the project's flagship demo: it shows what the Bayesian
treatment buys you (calibrated uncertainty) over the point estimate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    Standardizer,
    accuracy,
    log_loss,
    make_logistic_data,
    roc_auc,
    train_val_test_split,
)
from src.logistic_numpy import LogisticRegression


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=300,
                   help="Small n highlights the value of uncertainty.")
    p.add_argument("--n-features", type=int, default=4)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "comparison.png")
    args = p.parse_args()

    print(f"[compare] generating n={args.n_samples}, d={args.n_features}")
    data = make_logistic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        class_sep=1.5,
        seed=args.seed,
    )
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        data.X, data.y, seed=args.seed
    )
    scaler = Standardizer().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ---- NumPy / point estimate --------------------------------------
    print("\n[compare] fitting NumPy logistic regression (Adam)")
    np_model = LogisticRegression(l2=0.1)
    np_model.fit(X_tr_s, y_tr, optimizer="adam", epochs=300, lr=0.05,
                 verbose=False)
    np_p = np_model.predict_proba(X_te_s)
    np_yhat = (np_p >= 0.5).astype(int)

    print(f"  NumPy   acc={accuracy(y_te, np_yhat):.4f}  "
          f"log_loss={log_loss(y_te, np_p):.4f}  AUC={roc_auc(y_te, np_p):.4f}")

    # ---- Bayesian / posterior ----------------------------------------
    try:
        from src.logistic_pymc import BayesianLogisticRegression
        import pymc as pm  # noqa: F401
    except ImportError:
        print("\n[compare] PyMC not installed - install with: pip install pymc arviz")
        return 0

    print(f"\n[compare] fitting Bayesian logistic regression "
          f"(NUTS, {args.draws} draws/chain)")
    bayes = BayesianLogisticRegression(
        prior_sigma=2.5, draws=args.draws, tune=1000,
        chains=4, target_accept=0.9, random_seed=args.seed,
    )
    bayes.fit(X_tr_s, y_tr, progressbar=False)
    by_p = bayes.predict_proba(X_te_s)
    by_yhat = (by_p >= 0.5).astype(int)
    by_lo, by_hi = bayes.credible_interval(X_te_s, hdi_prob=0.94)

    print(f"  Bayes   acc={accuracy(y_te, by_yhat):.4f}  "
          f"log_loss={log_loss(y_te, by_p):.4f}  AUC={roc_auc(y_te, by_p):.4f}")

    diag = bayes.diagnostics()
    print(f"  diag    R-hat_max={diag['r_hat_max']:.3f}  "
          f"ESS_min={diag['ess_bulk_min']:.0f}")

    # ---- side by side ------------------------------------------------
    summary = bayes.summary()
    print("\n[compare] coefficients (standardized)")
    print(f"  {'feat':<6}  {'NumPy':>8}  {'Bayes mean':>12}  "
          f"{'Bayes sd':>10}  {'94% HDI':>20}")
    for i, name in enumerate(data.feature_names):
        hdi = f"[{summary.hdi_low[i]:+.2f}, {summary.hdi_high[i]:+.2f}]"
        print(f"  {name:<6}  {np_model.w[i]:+8.3f}  "
              f"{summary.mean[i]:+12.3f}  {summary.std[i]:10.3f}  {hdi:>20}")

    # ---- plot --------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Coefficients with uncertainty bands.
    idx = np.arange(len(data.feature_names))
    axes[0].errorbar(
        idx, summary.mean,
        yerr=[summary.mean - summary.hdi_low, summary.hdi_high - summary.mean],
        fmt="o", capsize=5, label="Bayesian (94% HDI)", color="#205493",
    )
    axes[0].scatter(idx, np_model.w, marker="x", s=80,
                    color="#d83933", label="NumPy point estimate", zorder=3)
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(data.feature_names)
    axes[0].set_ylabel("coefficient (standardized)")
    axes[0].set_title("point estimate vs posterior")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Predictive uncertainty on test set, sorted.
    order = np.argsort(by_p)
    axes[1].fill_between(
        np.arange(len(order)),
        by_lo[order], by_hi[order],
        alpha=0.25, color="#205493", label="Bayesian 94% HDI",
    )
    axes[1].plot(by_p[order], color="#205493", lw=1.6, label="Bayes posterior mean")
    axes[1].plot(np_p[order], color="#d83933", lw=1.4, ls="--",
                 label="NumPy point estimate")
    axes[1].scatter(np.arange(len(order)), y_te[order], s=10, alpha=0.4,
                    color="black", label="true label")
    axes[1].set_xlabel("test sample (sorted by posterior mean)")
    axes[1].set_ylabel("p(y = 1 | x)")
    axes[1].set_title("predictive uncertainty")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\n[compare] figure saved to {args.out}")


if __name__ == "__main__":
    main()
