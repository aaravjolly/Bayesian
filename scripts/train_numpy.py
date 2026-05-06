"""
End-to-end training pipeline for the from-scratch logistic regression.

Generates synthetic data, splits, standardizes, fits with the chosen
optimizer, and produces evaluation metrics + diagnostic plots.

Usage
-----
    python scripts/train_numpy.py                     # Adam, defaults
    python scripts/train_numpy.py --optimizer sgd     # Vanilla GD
    python scripts/train_numpy.py --batch-size 32 --epochs 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.logistic_numpy import LogisticRegression


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate. Default: 0.1 for SGD, 0.05 for Adam.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=None,
                   help="None = full-batch GD. Otherwise mini-batch SGD.")
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-features", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip generating diagnostic plots.")
    p.add_argument("--out", type=Path,
                   default=ROOT / "figures" / "numpy_training.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.lr is None:
        args.lr = 0.1 if args.optimizer == "sgd" else 0.05

    print(f"[pipeline] generating {args.n_samples} samples in "
          f"R^{args.n_features}")
    data = make_logistic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        class_sep=1.5,
        seed=args.seed,
    )

    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test_split(
        data.X, data.y, val_frac=0.15, test_frac=0.15, seed=args.seed
    )
    print(f"[pipeline] train={X_tr.shape[0]}  val={X_va.shape[0]}  "
          f"test={X_te.shape[0]}")

    scaler = Standardizer().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # ---- fit ----------------------------------------------------------
    print(f"[pipeline] fitting LogisticRegression "
          f"(optimizer={args.optimizer}, lr={args.lr}, l2={args.l2}, "
          f"epochs={args.epochs}, batch_size={args.batch_size})")

    model = LogisticRegression(l2=args.l2, fit_intercept=True)
    t0 = time.perf_counter()
    model.fit(
        X_tr_s, y_tr,
        optimizer=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation=(X_va_s, y_va),
        verbose=True,
        seed=args.seed,
        lr=args.lr,
    )
    elapsed = time.perf_counter() - t0
    print(f"[pipeline] fit complete in {elapsed:.2f}s")

    # ---- evaluate -----------------------------------------------------
    p_te = model.predict_proba(X_te_s)
    yhat = (p_te >= 0.5).astype(int)
    metrics = {
        "test_accuracy": accuracy(y_te, yhat),
        "test_log_loss": log_loss(y_te, p_te),
        "test_auc": roc_auc(y_te, p_te),
    }
    cm = confusion_matrix(y_te, yhat)

    print("\n[eval] test set")
    print(f"       accuracy : {metrics['test_accuracy']:.4f}")
    print(f"       log loss : {metrics['test_log_loss']:.4f}")
    print(f"       AUC      : {metrics['test_auc']:.4f}")
    print(f"       confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # Compare learned vs true weights (synthetic data ground truth).
    print("\n[eval] weight recovery (after un-standardizing)")
    # Convert standardized weights back to original-scale weights for fair compare.
    w_orig = model.w / scaler.scale_
    b_orig = model.b - float(np.dot(model.w, scaler.mean_ / scaler.scale_))
    for i, (name, true_wi, w_hat_i) in enumerate(
        zip(data.feature_names, data.true_w, w_orig)
    ):
        print(f"       {name}: true={true_wi:+.3f}  learned={w_hat_i:+.3f}"
              f"  diff={w_hat_i - true_wi:+.3f}")
    print(f"       intercept: true={data.true_b:+.3f}  learned={b_orig:+.3f}")

    # ---- save metrics -------------------------------------------------
    metrics_path = ROOT / "figures" / "numpy_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n[pipeline] metrics saved to {metrics_path}")

    # ---- plot ---------------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[pipeline] matplotlib not available, skipping plots")
            return 0

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Loss curves.
        axes[0].plot(model.history_.loss, label="train")
        if model.history_.val_loss:
            axes[0].plot(model.history_.val_loss, label="val")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("cross-entropy")
        axes[0].set_title("training curves")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Gradient norm decay.
        axes[1].semilogy(model.history_.grad_norm)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("|grad|  (log)")
        axes[1].set_title("gradient norm")
        axes[1].grid(alpha=0.3)

        # Weight recovery scatter.
        axes[2].scatter(data.true_w, w_orig, s=60, alpha=0.8)
        lo = min(data.true_w.min(), w_orig.min()) - 0.5
        hi = max(data.true_w.max(), w_orig.max()) + 0.5
        axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y=x")
        axes[2].set_xlabel("true weight")
        axes[2].set_ylabel("learned weight")
        axes[2].set_title("weight recovery")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print(f"[pipeline] plot saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
