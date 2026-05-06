# Bayesian Logistic Regression from Scratch

Logistic regression implemented two ways:

1. **From first principles in NumPy** — log-likelihood, analytic gradient, and optimization (vanilla gradient descent + Adam) all written by hand. No scikit-learn, no autograd.
2. **As a Bayesian model in PyMC** — Gaussian priors over the weights, posterior inference with the No-U-Turn Sampler (NUTS), full predictive uncertainty.

The two implementations share a common data pipeline so you can run them on the same dataset and compare a point estimate against a full posterior.

## What's actually in here

```
bayesian_logreg/
├── src/
│   ├── logistic_numpy.py     # The from-scratch model + Adam, GD, BCE, sigmoid
│   ├── logistic_pymc.py      # PyMC model + NUTS + posterior predictive
│   └── data.py               # synthetic data, splits, standardizer, metrics
├── scripts/
│   ├── train_numpy.py        # full pipeline for the NumPy model
│   ├── train_pymc.py         # full pipeline for the Bayesian model
│   ├── compare.py            # head-to-head on the same data
│   └── smoke_test.py         # pytest-free validator (incl. gradient check)
├── tests/                    # pytest suite (NumPy + PyMC)
├── figures/                  # generated plots and metrics
├── requirements.txt
└── README.md
```

## Highlights

**The from-scratch part is mathematically validated.** The smoke test checks the analytic gradient against a finite-difference numerical gradient and reports max error `~1.8e-11`. If the hand-derived gradient were wrong, this test would catch it immediately.

**The two models share a data layer.** `make_logistic_data`, the train/val/test splitter, the standardizer, and the metrics (`accuracy`, `log_loss`, `roc_auc`, `confusion_matrix`) all live in `src/data.py` and are dependency-free. Both training scripts use them.

**Numerically stable primitives.** `sigmoid` and `log_sigmoid` handle inputs as extreme as ±1000 without overflow or underflow. Cross-entropy is computed from logits via `logaddexp` rather than `log(sigmoid(z))`, which would zero-out for very negative logits.

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. validate the from-scratch math (no PyMC needed)
python scripts/smoke_test.py

# 3. train the NumPy model end-to-end
python scripts/train_numpy.py --optimizer adam --epochs 300

# 4. train the Bayesian model with NUTS
python scripts/train_pymc.py --draws 1000 --chains 4

# 5. compare them side by side on the same data
python scripts/compare.py
```

Each script prints metrics, posterior summaries, and saves figures to `figures/`.

## The from-scratch model

`LogisticRegression` in `src/logistic_numpy.py`:

- **Loss:** mean binary cross-entropy + (optional) L2 penalty on the weights only — bias is never regularized.
- **Gradient:** closed form, derived by hand:
  ```
  ∇w L = (1/n) · Xᵀ(σ(Xw + b) − y) + λw
  ∇b L = (1/n) · Σ(σ(Xw + b) − y)
  ```
- **Optimizers:** `GradientDescent(lr)` and `Adam(lr, beta1, beta2, eps)` are both implemented from scratch — Adam maintains its own moment buffers and applies bias correction. No PyTorch.
- **Mini-batching:** pass `batch_size=N` for SGD; leave it `None` for full-batch GD.
- **Validation tracking:** pass `validation=(X_val, y_val)` and val loss is recorded each epoch.

```python
from src import LogisticRegression, make_logistic_data, Standardizer

data = make_logistic_data(n_samples=2000, n_features=5)
scaler = Standardizer().fit(data.X)
X = scaler.transform(data.X)

model = LogisticRegression(l2=0.01).fit(
    X, data.y,
    optimizer="adam",   # or "sgd"
    epochs=300,
    batch_size=64,
    lr=0.05,
)
print(model.score(X, data.y))
print(model.history_.loss[-1])
```

## The Bayesian model

`BayesianLogisticRegression` in `src/logistic_pymc.py`:

- **Prior:** `w ~ Normal(0, prior_sigma)` per coefficient, `b ~ Normal(0, intercept_sigma)`. The prior is the regularizer.
- **Likelihood:** `y ~ Bernoulli(sigmoid(Xw + b))`.
- **Inference:** PyMC's NUTS sampler — adaptive Hamiltonian Monte Carlo. Multiple chains run in parallel so we can check convergence with R-hat.
- **Predictions:** `predict_proba(X)` returns the posterior mean probability per observation; `predict_proba(X, return_samples=True)` gives the full S × N matrix of probability samples; `credible_interval(X)` returns 94% (or any width) HDI bounds.

```python
from src import BayesianLogisticRegression

bayes = BayesianLogisticRegression(prior_sigma=2.5, draws=1000, chains=4)
bayes.fit(X_train, y_train)

mean_p = bayes.predict_proba(X_test)
lo, hi = bayes.credible_interval(X_test, hdi_prob=0.94)
print(bayes.diagnostics())  # R-hat, ESS
```

## What the Bayesian treatment buys you

The point-estimate model gives one number per prediction: "the probability is 0.73." The Bayesian model gives a *distribution*: "the probability is most likely around 0.73, but the 94% credible interval spans [0.55, 0.88] — there's real uncertainty here."

That matters when:
- You have **small training sets** (you can quantify how much the data actually constrains your weights).
- You need to **abstain or escalate** ambiguous predictions (wide credible interval = "I'm not sure, ask a human").
- You want **calibrated uncertainty** out-of-the-box without separate calibration steps.

`scripts/compare.py` runs both models on the same small dataset and produces a coefficient plot showing the Bayesian posterior intervals around the NumPy point estimates.

## Tests

```bash
pytest -q
```

The PyMC tests skip gracefully (via `pytest.importorskip`) if PyMC isn't installed, so the NumPy tests run anywhere. The headline test is `test_gradient_matches_finite_difference` in `tests/test_numpy.py`, which directly verifies the analytic gradient.

## Implementation notes

**Why pack `[w, b]` into one vector?** It lets the optimizer stay generic — `Adam.step(params, grad)` doesn't need to know the model has a bias term. The `LogisticRegression` class handles packing/unpacking internally.

**Why is L2 not applied to the bias?** Standard practice. Penalizing the intercept shrinks the model toward predicting `p = 0.5`, which is rarely what you want; you want to shrink the *slopes*, not the baseline rate.

**Why use `logaddexp` for cross-entropy?** `log(1 + exp(z))` overflows for large positive `z` and underflows-to-zero on the inside for large negative `z`. `np.logaddexp(0, z)` is numerically stable everywhere.

**Why does `predict_proba` in the Bayesian model average over samples instead of plugging in the posterior mean?** Because they're not the same! `E[σ(z)]` (averaging probabilities over posterior samples) gives the posterior predictive distribution. `σ(E[z])` (plugging in the mean weights) ignores uncertainty and is generally overconfident. This is a subtle but real difference.
