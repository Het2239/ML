"""Degree selection for polynomial regression via repeated K-fold CV.

For every degree d we jointly tune the regularisation strength:
  * Ridge over a log-spaced alpha grid (alpha = 0 is plain OLS). This uses an
    exact SVD-based path so all alphas cost one SVD per (degree, fold).
  * Lasso (optional, --lasso) over its own alpha grid on a subset of degrees.

The final (degree, alpha) is chosen with the one-standard-error rule:
  best   = config with the lowest mean CV MSE
  thresh = best.mean + best.se
  pick the LOWEST degree having any config with mean <= thresh, and within it
  the LARGEST alpha (strongest regularisation) still under the threshold.

Usage:
  python src/select_degree.py --var var1
  python src/select_degree.py --var var2 --lasso --lasso-degrees 6-14
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import PolynomialFeatures

from common import (DEFAULT_MAX_DEGREE, PLOTS_DIR, RESULTS_DIR, cv_splits,
                    load_train, make_model, n_poly_terms)

RIDGE_ALPHAS = np.concatenate([[0.0], np.logspace(-6, 3, 19)])
LASSO_ALPHAS = np.logspace(-5, -1, 9)


# --------------------------------------------------------------------------- #
# Ridge: exact regularisation path via SVD
# --------------------------------------------------------------------------- #
def _ridge_fold(P, y, tr, va, alphas):
    """Return (val_mse, val_r2, train_mse) arrays of shape (n_alphas,)."""
    mu = P[tr].mean(axis=0)
    sd = P[tr].std(axis=0)
    sd[sd == 0] = 1.0
    A = (P[tr] - mu) / sd
    B = (P[va] - mu) / sd
    y_mean = y[tr].mean()
    yc = y[tr] - y_mean

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Uty = U.T @ yc

    # shrinkage factors d_i(alpha) = s_i / (s_i^2 + alpha); alpha=0 -> pseudo-inverse
    D = np.empty((len(alphas), len(s)))
    for k, a in enumerate(alphas):
        if a == 0:
            keep = s > s[0] * max(A.shape) * np.finfo(float).eps
            D[k] = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
        else:
            D[k] = s / (s ** 2 + a)
    W = Vt.T @ (D * Uty).T  # (p, n_alphas)

    pv = B @ W + y_mean
    pt = A @ W + y_mean
    yv, yt = y[va][:, None], y[tr][:, None]
    val_mse = ((pv - yv) ** 2).mean(axis=0)
    tr_mse = ((pt - yt) ** 2).mean(axis=0)
    val_r2 = 1.0 - val_mse / y[va].var()
    return val_mse, val_r2, tr_mse


def ridge_search(X, y, degrees, splits, alphas=RIDGE_ALPHAS, n_jobs=-1):
    rows = []
    for d in degrees:
        t0 = time.time()
        P = PolynomialFeatures(d, include_bias=False).fit_transform(X)
        out = Parallel(n_jobs=n_jobs)(
            delayed(_ridge_fold)(P, y, tr, va, alphas) for tr, va in splits)
        vm = np.stack([o[0] for o in out])  # (n_folds, n_alphas)
        vr = np.stack([o[1] for o in out])
        tm = np.stack([o[2] for o in out])
        nf = len(splits)
        for k, a in enumerate(alphas):
            rows.append(dict(model="ridge", degree=d, alpha=a, n_terms=P.shape[1],
                             cv_mse=vm[:, k].mean(),
                             cv_se=vm[:, k].std(ddof=1) / np.sqrt(nf),
                             cv_r2=vr[:, k].mean(),
                             train_mse=tm[:, k].mean()))
        best = min(r["cv_mse"] for r in rows if r["degree"] == d)
        print(f"  ridge deg {d:2d} ({P.shape[1]:5d} terms)  best CV MSE {best:.4f}"
              f"  [{time.time() - t0:.1f}s]", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Lasso: brute-force grid (sklearn coordinate descent)
# --------------------------------------------------------------------------- #
def _lasso_fold(X, y, d, a, tr, va):
    m = make_model(d, "lasso", a, use_gpu=False).fit(X[tr], y[tr])
    pv, pt = m.predict(X[va]), m.predict(X[tr])
    vmse = ((pv - y[va]) ** 2).mean()
    return vmse, 1 - vmse / y[va].var(), ((pt - y[tr]) ** 2).mean(), \
        int(np.count_nonzero(m.named_steps["reg"].coef_))


def lasso_search_gpu(X, y, degrees, splits, alphas=LASSO_ALPHAS, dtype="float32"):
    from gpu_lasso import default_device, lasso_cv_gpu
    dev = default_device()
    tol = 1e-5 if dtype == "float32" else 1e-6
    rows = []
    for d in degrees:
        t0 = time.time()
        P = PolynomialFeatures(d, include_bias=False).fit_transform(X)
        o = lasso_cv_gpu(P, y, splits, alphas, dtype=dtype, device=dev, tol=tol)
        nf = len(splits)
        for k, a in enumerate(alphas):
            rows.append(dict(model="lasso", degree=d, alpha=a, n_terms=P.shape[1],
                             cv_mse=o["val_mse"][:, k].mean(),
                             cv_se=o["val_mse"][:, k].std(ddof=1) / np.sqrt(nf),
                             cv_r2=o["val_r2"][:, k].mean(),
                             train_mse=o["train_mse"][:, k].mean(),
                             nonzero=o["nonzero"][:, k].mean()))
        best = min(r["cv_mse"] for r in rows if r["degree"] == d)
        print(f"  lasso[{dev}] deg {d:2d} ({P.shape[1]:5d} terms)  best CV MSE {best:.4f}"
              f"  iters {o['n_iter']}  [{time.time() - t0:.1f}s]", flush=True)
    return pd.DataFrame(rows)


def lasso_search(X, y, degrees, splits, alphas=LASSO_ALPHAS, n_jobs=-1):
    rows = []
    for d in degrees:
        t0 = time.time()
        jobs = [(a, tr, va) for a in alphas for tr, va in splits]
        out = Parallel(n_jobs=n_jobs)(
            delayed(_lasso_fold)(X, y, d, a, tr, va) for a, tr, va in jobs)
        out = np.array(out).reshape(len(alphas), len(splits), 4)
        nf = len(splits)
        for k, a in enumerate(alphas):
            o = out[k]
            rows.append(dict(model="lasso", degree=d, alpha=a,
                             n_terms=n_poly_terms(X.shape[1], d),
                             cv_mse=o[:, 0].mean(),
                             cv_se=o[:, 0].std(ddof=1) / np.sqrt(nf),
                             cv_r2=o[:, 1].mean(), train_mse=o[:, 2].mean(),
                             nonzero=o[:, 3].mean()))
        best = min(r["cv_mse"] for r in rows if r["degree"] == d)
        print(f"  lasso deg {d:2d}  best CV MSE {best:.4f}  [{time.time() - t0:.1f}s]",
              flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def one_se_rule(df):
    best = df.loc[df.cv_mse.idxmin()]
    thresh = best.cv_mse + best.cv_se
    ok = df[df.cv_mse <= thresh]
    d = ok.degree.min()
    pick = ok[ok.degree == d].sort_values("alpha").iloc[-1]
    return best, pick, thresh


def per_degree_best(df):
    return df.loc[df.groupby("degree").cv_mse.idxmin()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(var, ridge, lasso, pick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(exist_ok=True)
    rb = per_degree_best(ridge)
    ols = ridge[ridge.alpha == 0].reset_index(drop=True)

    # 1) degree curve: train vs CV error
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ols.degree, ols.cv_mse, yerr=ols.cv_se, marker="s", ms=4,
                capsize=3, label="OLS — CV MSE", color="tab:gray", alpha=.8)
    ax.plot(ols.degree, ols.train_mse, ls="--", color="tab:gray", alpha=.6,
            label="OLS — train MSE")
    ax.errorbar(rb.degree, rb.cv_mse, yerr=rb.cv_se, marker="o", ms=4,
                capsize=3, label="Ridge (tuned α) — CV MSE", color="tab:blue")
    ax.plot(rb.degree, rb.train_mse, ls="--", color="tab:blue", alpha=.6,
            label="Ridge (tuned α) — train MSE")
    if lasso is not None and len(lasso):
        lb = per_degree_best(lasso)
        ax.errorbar(lb.degree, lb.cv_mse, yerr=lb.cv_se, marker="^", ms=4,
                    capsize=3, label="Lasso (tuned α) — CV MSE", color="tab:green")
    ax.axvline(pick.degree, color="tab:red", ls=":",
               label=f"chosen: degree {int(pick.degree)} ({pick.model})")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title(f"{var}: bias–variance trade-off vs degree (repeated 5-fold CV)")
    ax.set_xticks(sorted(ridge.degree.unique()))
    ax.grid(alpha=.3, which="both")
    lo = min(rb.train_mse.min(), rb.cv_mse.min()) * 0.5
    ax.set_ylim(lo, ridge.cv_mse[ridge.degree == ridge.degree.min()].max() * 2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{var}_degree_curve.png", dpi=150)
    plt.close(fig)

    # 2) heatmap of log10 CV MSE over degree x alpha (ridge)
    piv = ridge[ridge.alpha > 0].pivot(index="degree", columns="alpha", values="cv_mse")
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(piv) + 2))
    im = ax.imshow(np.log10(piv.values), aspect="auto", cmap="viridis_r",
                   vmax=np.log10(np.nanpercentile(piv.values, 90)))
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xticks(range(len(piv.columns)), [f"{a:.0e}" for a in piv.columns],
                  rotation=60, fontsize=7)
    ax.set_xlabel("Ridge α")
    ax.set_ylabel("degree")
    ax.set_title(f"{var}: log10 CV MSE (Ridge)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{var}_ridge_heatmap.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def parse_range(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, choices=["var1", "var2"])
    ap.add_argument("--max-degree", type=int)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lasso", action="store_true", help="also run the Lasso grid")
    ap.add_argument("--lasso-degrees", type=str, help="e.g. 2-6 (default: around ridge pick)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--cpu-lasso", action="store_true", help="force sklearn CPU Lasso")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="GPU Lasso precision")
    args = ap.parse_args()

    var = args.var
    X, y, cols = load_train(var)
    max_d = args.max_degree or DEFAULT_MAX_DEGREE[var]
    splits = cv_splits(len(y), args.folds, args.repeats)
    print(f"[{var}] n={len(y)} features={cols}  degrees 1..{max_d}  "
          f"CV {args.folds}x{args.repeats}")

    ridge = ridge_search(X, y, range(1, max_d + 1), splits, n_jobs=args.n_jobs)
    _, ridge_pick, _ = one_se_rule(ridge)

    lasso = None
    if args.lasso:
        degs = (parse_range(args.lasso_degrees) if args.lasso_degrees else
                list(range(max(1, int(ridge_pick.degree) - 2), int(ridge_pick.degree) + 3)))
        use_gpu = False
        if not args.cpu_lasso:
            try:
                from gpu_lasso import HAS_CUDA
                use_gpu = HAS_CUDA
            except Exception:
                pass
        if use_gpu:
            lasso = lasso_search_gpu(X, y, degs, splits, dtype=args.dtype)
        else:
            print("  (no CUDA available -> sklearn CPU Lasso)")
            lasso = lasso_search(X, y, degs, splits, n_jobs=args.n_jobs)

    allres = pd.concat([ridge, lasso]) if lasso is not None else ridge
    allres = allres.reset_index(drop=True)
    best, pick, thresh = one_se_rule(allres)

    RESULTS_DIR.mkdir(exist_ok=True)
    allres.to_csv(RESULTS_DIR / f"{var}_cv_results.csv", index=False)

    print("\nBest config per degree:")
    for m, g in allres.groupby("model"):
        t = per_degree_best(g)[["degree", "n_terms", "alpha", "cv_mse", "cv_se", "cv_r2", "train_mse"]]
        print(f"--- {m}\n" + t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    summary = dict(
        var=var, n_train=len(y), cv=f"{args.folds}-fold x {args.repeats} repeats",
        best=dict(model=best.model, degree=int(best.degree), alpha=float(best.alpha),
                  cv_mse=float(best.cv_mse), cv_se=float(best.cv_se), cv_r2=float(best.cv_r2)),
        one_se_threshold=float(thresh),
        chosen=dict(model=pick.model, degree=int(pick.degree), alpha=float(pick.alpha),
                    cv_mse=float(pick.cv_mse), cv_se=float(pick.cv_se), cv_r2=float(pick.cv_r2)),
    )
    with open(RESULTS_DIR / f"{var}_selection.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLowest CV MSE : {best.model} deg {int(best.degree)} α={best.alpha:.2e} "
          f"MSE {best.cv_mse:.4f} ± {best.cv_se:.4f}  R² {best.cv_r2:.4f}")
    print(f"1-SE choice   : {pick.model} deg {int(pick.degree)} α={pick.alpha:.2e} "
          f"MSE {pick.cv_mse:.4f}  R² {pick.cv_r2:.4f}  (threshold {thresh:.4f})")

    make_plots(var, ridge, lasso, pick)
    print(f"Saved results/{var}_cv_results.csv, results/{var}_selection.json, plots/{var}_*.png")


if __name__ == "__main__":
    main()
