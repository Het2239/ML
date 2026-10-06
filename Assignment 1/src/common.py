"""Shared helpers: paths, data loading, model construction and CV splits."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROLL = "BT2024052"
ROOT = Path(__file__).resolve().parent.parent  # .../Assignment 1
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
PRED_DIR = ROOT / "predictions"

# Default degree search range per problem.
# var1: 6 features -> degree 7 already has 1715 terms (> 1000 rows), so going
#       further is pointless. var2: 3 features, PDF says "up to degree 20".
DEFAULT_MAX_DEGREE = {"var1": 7, "var2": 20}

SEED = 42


def train_path(var: str) -> Path:
    return ROOT / f"{ROLL}_train_{var}.csv"


def test_path(var: str) -> Path:
    return ROOT / f"{ROLL}_test_{var}.csv"


def pred_path(var: str) -> Path:
    return PRED_DIR / f"{ROLL}_pred_{var}.csv"


def load_train(var: str):
    df = pd.read_csv(train_path(var))
    X = df.drop(columns="y").to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y, [c for c in df.columns if c != "y"]


def load_test(var: str):
    df = pd.read_csv(test_path(var))
    return df.to_numpy(dtype=float), list(df.columns)


def cv_splits(n: int, n_splits: int = 5, n_repeats: int = 3, seed: int = SEED):
    """Fixed list of (train_idx, val_idx) so every model sees identical folds."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rkf.split(np.zeros(n)))


def make_model(degree: int, kind: str = "ridge", alpha: float = 0.0,
               use_gpu: bool = True) -> Pipeline:
    """Polynomial regression pipeline.

    PolynomialFeatures (no bias column) -> StandardScaler -> linear model.
    Standardising the expanded features puts every monomial on the same scale,
    so the Ridge/Lasso penalty treats all terms equally. The intercept is fit
    separately and is never penalised.

    kind = "ols"   : ordinary least squares (alpha ignored)
    kind = "ridge" : L2 penalty, objective ||y - Xw||^2 + alpha * ||w||^2
                     (alpha == 0 falls back to OLS)
    kind = "lasso" : L1 penalty, objective 1/(2n) ||y - Xw||^2 + alpha * ||w||_1
    """
    if kind == "ols" or (kind == "ridge" and alpha == 0):
        reg = LinearRegression()
    elif kind == "ridge":
        reg = Ridge(alpha=alpha)
    elif kind == "lasso":
        reg = None
        if use_gpu:
            try:
                from gpu_lasso import HAS_CUDA, GPULasso
                if HAS_CUDA:
                    reg = GPULasso(alpha=alpha)
            except ImportError:
                pass
        if reg is None:
            reg = Lasso(alpha=alpha, max_iter=100_000, tol=1e-4, selection="cyclic")
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("reg", reg),
    ])


def n_poly_terms(n_features: int, degree: int) -> int:
    """Number of monomials of total degree <= `degree` (excluding the constant)."""
    from math import comb
    return comb(n_features + degree, degree) - 1
