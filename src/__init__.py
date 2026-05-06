"""Bayesian Logistic Regression from Scratch."""

from .data import (
    Standardizer,
    SyntheticData,
    accuracy,
    confusion_matrix,
    log_loss,
    make_logistic_data,
    roc_auc,
    train_val_test_split,
)
from .logistic_numpy import (
    Adam,
    GradientDescent,
    LogisticRegression,
    Optimizer,
    TrainHistory,
    binary_cross_entropy,
    log_sigmoid,
    make_optimizer,
    numerical_gradient,
    sigmoid,
)

# PyMC import is optional - only fails when actually instantiating the class.
try:
    from .logistic_pymc import BayesianLogisticRegression, PosteriorSummary
except ImportError:
    BayesianLogisticRegression = None  # type: ignore
    PosteriorSummary = None  # type: ignore

__all__ = [
    "Adam",
    "BayesianLogisticRegression",
    "GradientDescent",
    "LogisticRegression",
    "Optimizer",
    "PosteriorSummary",
    "Standardizer",
    "SyntheticData",
    "TrainHistory",
    "accuracy",
    "binary_cross_entropy",
    "confusion_matrix",
    "log_loss",
    "log_sigmoid",
    "make_logistic_data",
    "make_optimizer",
    "numerical_gradient",
    "roc_auc",
    "sigmoid",
    "train_val_test_split",
]

__version__ = "1.0.0"
