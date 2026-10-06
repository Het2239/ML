"""Fit the selected polynomial model on all training rows and predict the test set.

Reads results/<var>_selection.json (written by select_degree.py) unless the
model is overridden on the command line. Writes
predictions/<ROLLNO>_pred_<var>.csv with a single `y` column (same format as
sample_submission.csv).

Usage:
  python src/predict.py --var var1
  python src/predict.py --var var2 --model ridge --degree 10 --alpha 1e-1
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import (PRED_DIR, RESULTS_DIR, load_test, load_train, make_model,
                    pred_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, choices=["var1", "var2"])
    ap.add_argument("--model", choices=["ols", "ridge", "lasso"])
    ap.add_argument("--degree", type=int)
    ap.add_argument("--alpha", type=float)
    args = ap.parse_args()
    var = args.var

    sel = {}
    sel_file = RESULTS_DIR / f"{var}_selection.json"
    if sel_file.exists():
        sel = json.load(open(sel_file))["chosen"]
    kind = args.model or sel.get("model")
    deg = args.degree or sel.get("degree")
    alpha = args.alpha if args.alpha is not None else sel.get("alpha", 0.0)
    if kind is None or deg is None:
        raise SystemExit("No selection found; run select_degree.py or pass --model/--degree.")

    X, y, cols = load_train(var)
    Xt, tcols = load_test(var)
    assert tcols == cols, f"column mismatch: {tcols} vs {cols}"

    model = make_model(deg, kind, alpha).fit(X, y)
    tr_pred = model.predict(X)
    tr_mse = np.mean((tr_pred - y) ** 2)
    print(f"[{var}] {kind} degree {deg} alpha {alpha:.3g}: "
          f"train MSE {tr_mse:.4f}  R² {1 - tr_mse / y.var():.4f}")

    yt = model.predict(Xt)
    print(f"test preds: n={len(yt)} mean {yt.mean():.3f} std {yt.std():.3f} "
          f"range [{yt.min():.3f}, {yt.max():.3f}]   (train y range [{y.min():.3f}, {y.max():.3f}])")

    PRED_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"y": yt}).to_csv(pred_path(var), index=False)
    print(f"Saved {pred_path(var).relative_to(PRED_DIR.parent)}")


if __name__ == "__main__":
    main()
