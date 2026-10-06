"""Diagnostics for the chosen polynomial model of a problem.

1. Out-of-fold residuals split by how many inputs sit on the clipping
   boundary (|x| == 1). Tells us whether the error floor comes from clipping.
2. Coefficient magnitude grouped by monomial total degree (fit on all rows).
   If the top-degree terms are ~0 the true degree is probably lower.
3. Lasso support: which monomial degrees survive an L1 fit.
4. Clipping-treatment experiment: does dropping / down-weighting rows with
   clipped inputs during training improve CV MSE (evaluated on ALL val rows)?

Usage:  python src/diagnostics.py --var var1 [--degree 4 --alpha 1e-2 --model ridge]
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import PLOTS_DIR, RESULTS_DIR, cv_splits, load_train, make_model


def load_choice(var, args):
    sel = json.load(open(RESULTS_DIR / f"{var}_selection.json"))["chosen"]
    return (args.model or sel["model"], args.degree or sel["degree"],
            args.alpha if args.alpha is not None else sel["alpha"])


def oof_predict(X, y, model_fn, splits, train_mask=None, weights=None):
    """Out-of-fold predictions averaged over repeats; also per-fold MSE."""
    pred_sum = np.zeros(len(y))
    cnt = np.zeros(len(y))
    fold_mse = []
    for tr, va in splits:
        tr_use = tr if train_mask is None else tr[train_mask[tr]]
        m = model_fn()
        if weights is not None:
            m.fit(X[tr_use], y[tr_use], reg__sample_weight=weights[tr_use])
        else:
            m.fit(X[tr_use], y[tr_use])
        p = m.predict(X[va])
        pred_sum[va] += p
        cnt[va] += 1
        fold_mse.append(((p - y[va]) ** 2).mean())
    return pred_sum / cnt, np.array(fold_mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, choices=["var1", "var2"])
    ap.add_argument("--model")
    ap.add_argument("--degree", type=int)
    ap.add_argument("--alpha", type=float)
    args = ap.parse_args()
    var = args.var

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS_DIR.mkdir(exist_ok=True)

    X, y, cols = load_train(var)
    kind, deg, alpha = load_choice(var, args)
    print(f"[{var}] diagnosing {kind} degree {deg} alpha {alpha:.3g}")
    model_fn = lambda: make_model(deg, kind, alpha)
    splits = cv_splits(len(y))

    # ---------------- 1. residuals vs clipping ---------------- #
    n_clip = (np.abs(X) >= 1.0).sum(axis=1)
    oof, fold_mse = oof_predict(X, y, model_fn, splits)
    res = y - oof
    print(f"\nOOF MSE {np.mean(res ** 2):.4f}  (fold mean {fold_mse.mean():.4f})")
    tab = (pd.DataFrame({"n_clipped": n_clip, "sq_err": res ** 2})
           .groupby("n_clipped").sq_err.agg(["count", "mean"])
           .rename(columns={"mean": "oof_mse"}))
    print("\nOOF MSE by number of inputs at ±1:\n" + tab.to_string(float_format="%.4f"))
    print(f"rows with no clipped input: {(n_clip == 0).sum()}  "
          f"MSE {np.mean(res[n_clip == 0] ** 2):.4f}")
    print(f"rows with >=1 clipped input: {(n_clip > 0).sum()}  "
          f"MSE {np.mean(res[n_clip > 0] ** 2):.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    groups = [res[n_clip == k] for k in sorted(set(n_clip))]
    axs[0].boxplot(groups, tick_labels=[str(k) for k in sorted(set(n_clip))])
    axs[0].set_xlabel("number of inputs clipped at ±1")
    axs[0].set_ylabel("out-of-fold residual")
    axs[0].set_title("residuals vs clipping")
    axs[1].scatter(oof, y, s=6, alpha=.5)
    lim = [min(y.min(), oof.min()), max(y.max(), oof.max())]
    axs[1].plot(lim, lim, "r--", lw=1)
    axs[1].set_xlabel("out-of-fold prediction")
    axs[1].set_ylabel("true y")
    axs[1].set_title("predicted vs actual (OOF)")
    fig.suptitle(f"{var}: {kind} degree {deg}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{var}_residuals.png", dpi=150)
    plt.close(fig)

    # ---------------- 2. coefficient magnitude by degree ---------------- #
    m = model_fn().fit(X, y)
    powers = m.named_steps["poly"].powers_
    tdeg = powers.sum(axis=1)
    coef = m.named_steps["reg"].coef_
    cdf = pd.DataFrame({"deg": tdeg, "abs": np.abs(coef)})
    by_deg = cdf.groupby("deg")["abs"].agg(["count", "sum", "max"])
    by_deg["rms"] = cdf.groupby("deg")["abs"].apply(lambda a: np.sqrt((a ** 2).mean()))
    print("\n|coef| (standardised features) by monomial degree:\n"
          + by_deg.to_string(float_format="%.4f"))

    names = m.named_steps["poly"].get_feature_names_out(cols)
    top = np.argsort(-np.abs(coef))[:15]
    print("\nlargest terms:\n" + "\n".join(f"  {names[i]:>20s}  {coef[i]:+.4f}" for i in top))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_deg.index, by_deg["rms"])
    ax.set_xlabel("monomial total degree")
    ax.set_ylabel("RMS |coefficient| (standardised)")
    ax.set_title(f"{var}: coefficient size by term degree ({kind}, d={deg})")
    ax.set_xticks(by_deg.index)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{var}_coef_by_degree.png", dpi=150)
    plt.close(fig)

    # ---------------- 3. Lasso support ---------------- #
    lasso_alpha = alpha if kind == "lasso" else 1e-3
    lm = make_model(deg, "lasso", lasso_alpha).fit(X, y)
    lc = lm.named_steps["reg"].coef_
    sup = pd.Series(tdeg[lc != 0]).value_counts().sort_index()
    print(f"\nLasso (alpha={lasso_alpha:.0e}) non-zero terms by degree "
          f"({(lc != 0).sum()}/{len(lc)}):\n" + sup.to_string())

    # ---------------- 4. clipping treatment experiment ---------------- #
    print("\nClipping treatment (CV MSE on ALL validation rows):")
    base = fold_mse
    exp = {"baseline (all rows)": base}
    _, exp["train only on unclipped rows"] = oof_predict(
        X, y, model_fn, splits, train_mask=(n_clip == 0))
    if kind == "ridge" and alpha > 0 or kind == "lasso":
        for w in (0.5, 0.25):
            wts = np.where(n_clip > 0, w, 1.0)
            _, exp[f"down-weight clipped rows x{w}"] = oof_predict(
                X, y, model_fn, splits, weights=wts)
    for k, v in exp.items():
        print(f"  {k:32s} {v.mean():.4f} ± {v.std(ddof=1) / np.sqrt(len(v)):.4f}")

    print(f"\nSaved plots/{var}_residuals.png, plots/{var}_coef_by_degree.png")


if __name__ == "__main__":
    main()
