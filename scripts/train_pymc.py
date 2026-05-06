"""
Bayesian training pipeline.

Fits the same logistic regression as ``train_numpy.py`` but treats the
weights as random variables and samples the posterior with NUTS via PyMC.

Outputs
-------
- Convergence diagnostics (R-hat, ESS).
- Posterior summary (mean, std, 94% HDI per coefficient).
- Posterior predictive metrics on the test set.
- Trace + posterior + uncertainty plots saved to ``figures/``.

Usage
-----
    python scripts/train_pymc.py
    python scripts/train_pymc.py --draws 2000 --chains 4 --prior-sigma 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--prior-sigma", type=float, default=2.5)
    p.add_argument("--target-accept", type=float, default=0.9)
    p.add_argument("--n-samples", type=int, default=800)
    p.add_argument("--n-features", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Importing PyMC here so that running --help doesn't require it.
    try:
        from src.logistic_pymc import BayesianLogisticRegression
        import pymc as pm  # noqa: F401  (just for a clearer error message)
    except ImportError as exc:
        print(f"[error] PyMC is required for this script: {exc}", file=sys.stderr)
        print("        install with: pip install pymc arviz", file=sys.stderr)
        return 1

    print(f"[pipeline] generating {args.n_samples} samples in R^{args.n_features}")
    data = make_logistic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        class_sep=1.5,
        seed=args.seed,
    )

    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        data.X, data.y, val_frac=0.15, test_frac=0.15, seed=args.seed
    )
    scaler = Standardizer().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ---- fit ----------------------------------------------------------
    print(f"[pipeline] sampling posterior  draws={args.draws}  "
          f"chains={args.chains}  tune={args.tune}")
    model = BayesianLogisticRegression(
        prior_sigma=args.prior_sigma,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )
    t0 = time.perf_counter()
    model.fit(X_tr_s, y_tr, progressbar=True)
    elapsed = time.perf_counter() - t0
    print(f"[pipeline] sampling complete in {elapsed:.1f}s")

    # ---- diagnostics --------------------------------------------------
    diag = model.diagnostics()
    print("\n[diagnostics] convergence")
    print(f"  max R-hat      : {diag['r_hat_max']:.4f}  (want < 1.01)")
    print(f"  min bulk ESS   : {diag['ess_bulk_min']:.0f}  (want > a few hundred)")
    print(f"  min tail ESS   : {diag['ess_tail_min']:.0f}")
    if diag["r_hat_max"] >= 1.01:
        print("  WARNING: R-hat suggests chains have not mixed - "
              "try more tuning or higher target_accept.")

    # ---- posterior summary --------------------------------------------
    summary = model.summary(hdi_prob=0.94)
    print("\n[posterior] coefficient summary (94% HDI)")
    # Convert to original-scale weights for interpretability.
    w_means_orig = summary.mean / scaler.scale_
    print(f"  {'feature':<10}  {'true':>8}  "
          f"{'mean':>8}  {'sd':>8}  {'hdi_low':>8}  {'hdi_high':>8}")
    for i, name in enumerate(data.feature_names):
        print(f"  {name:<10}  {data.true_w[i]:+8.3f}  "
              f"{w_means_orig[i]:+8.3f}  {summary.std[i]:8.3f}  "
              f"{summary.hdi_low[i] / scaler.scale_[i]:+8.3f}  "
              f"{summary.hdi_high[i] / scaler.scale_[i]:+8.3f}")
    print(f"  intercept:  true={data.true_b:+.3f}  "
          f"posterior mean={summary.intercept_mean:+.3f}  "
          f"sd={summary.intercept_std:.3f}")

    # ---- predictive evaluation ---------------------------------------
    p_te_mean = model.predict_proba(X_te_s)
    yhat = (p_te_mean >= 0.5).astype(int)
    lo, hi = model.credible_interval(X_te_s, hdi_prob=0.94)
    width = hi - lo

    print("\n[eval] test set")
    print(f"  accuracy        : {accuracy(y_te, yhat):.4f}")
    print(f"  log loss        : {log_loss(y_te, p_te_mean):.4f}")
    print(f"  AUC             : {roc_auc(y_te, p_te_mean):.4f}")
    print(f"  mean 94% HDI width on p(y=1): {width.mean():.3f}")

    # ---- plots --------------------------------------------------------
    if not args.no_plot:
        try:
            import arviz as az
            import matplotlib.pyplot as plt
        except ImportError:
            print("[pipeline] arviz/matplotlib not available, skipping plots")
            return 0

        args.fig_dir.mkdir(parents=True, exist_ok=True)

        # Trace plot.
        az.plot_trace(model.idata_, var_names=["w", "b"])
        plt.gcf().tight_layout()
        plt.gcf().savefig(args.fig_dir / "pymc_trace.png", dpi=130, bbox_inches="tight")
        plt.close("all")

        # Forest / posterior summary.
        az.plot_forest(model.idata_, var_names=["w"], hdi_prob=0.94, combined=True)
        plt.gcf().savefig(args.fig_dir / "pymc_forest.png", dpi=130, bbox_inches="tight")
        plt.close("all")

        # Predictive uncertainty: sort test points by predicted probability,
        # plot mean +/- HDI band.
        order = np.argsort(p_te_mean)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(np.arange(len(order)),
                        lo[order], hi[order],
                        alpha=0.25, label="94% HDI")
        ax.plot(p_te_mean[order], lw=1.5, label="posterior mean")
        ax.scatter(np.arange(len(order)), y_te[order],
                   s=8, alpha=0.4, color="black", label="true label")
        ax.set_xlabel("test sample (sorted by predicted p)")
        ax.set_ylabel("p(y = 1 | x)")
        ax.set_title("posterior predictive with credible band")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.fig_dir / "pymc_predictive.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        print(f"\n[pipeline] saved figures to {args.fig_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
