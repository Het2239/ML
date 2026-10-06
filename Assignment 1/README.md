# ML Assignment 1: Polynomial Regression

This repository contains the solution for the Machine Learning Assignment 1 on Polynomial Regression.

## Repository Structure
- `src/` - Contains all python scripts:
  - `common.py`: Shared utilities, loaders, and sklearn pipeline creation.
  - `gpu_lasso.py`: Custom batched GPU FISTA solver for Lasso to drastically speed up hyperparameter tuning and model fitting.
  - `select_degree.py`: Main script for performing K-Fold CV (with one-standard-error rule) and grid search across degrees and alpha for both Ridge and Lasso.
  - `diagnostics.py`: Generates validation and coefficient metrics for the selected models.
  - `predict.py`: Fits the chosen configuration on the entire training set and outputs predictions.
- `predictions/` - Final prediction outputs for the test datasets (`BT2024052_pred_var1.csv`, `BT2024052_pred_var2.csv`).
- `results/` & `plots/` - JSON configuration files, CV result metrics, and visual learning curves and diagnostics.
- `report.md` & `BT2024052_Report.pdf` - Writeup containing the approach, degrees chosen, and rationale.

## Requirements
```bash
pip install -r requirements.txt
```
*Note: PyTorch with CUDA is recommended for the fast FISTA Lasso solver. If CUDA is unavailable, the pipeline gracefully falls back to scikit-learn's coordinate descent on the CPU.*

## Usage
To re-run degree selection for both problems:
```bash
python src/select_degree.py --var var1 --lasso
python src/select_degree.py --var var2 --lasso
```

To run diagnostics:
```bash
python src/diagnostics.py --var var1
python src/diagnostics.py --var var2
```

To output predictions:
```bash
python src/predict.py --var var1
python src/predict.py --var var2
```
