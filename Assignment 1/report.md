# Assignment: Polynomial Regression
**Roll Number:** BT2024052

## 1. Introduction and Methodology
In this assignment, two separate polynomial regression models are developed to predict continuous target variables based on distinct datasets. The methodology across both problems is uniform and structured to prevent overfitting while selecting the optimal degree of the polynomial.

### 1.1 Experimental Setup
- **Scaling:** For any given polynomial degree, the expanded features (monomials) can have drastically different ranges. Hence, standard scaling (`StandardScaler`) is applied *after* generating polynomial features and *before* fitting the regression. This ensures the regularization penalties treat all terms equitably.
- **Cross-Validation (CV):** We used repeated $K$-fold cross-validation (5 folds, 3 repeats) to obtain a robust estimate of out-of-sample Mean Squared Error (MSE). This reduces the variance of the evaluation metric compared to a single train-test split.
- **Model Selection Rule:** We utilized the **one-standard-error (1-SE) rule**. Instead of simply picking the configuration with the absolute lowest CV MSE, we selected the simplest model (lowest degree, strongest regularization) whose CV MSE falls within one standard error of the best model's CV MSE. This acts as a safeguard against overfitting by favoring simpler, more generalized models.
- **GPU-Accelerated FISTA Lasso:** To evaluate a wide grid of polynomial degrees and L1 penalty ($\alpha$) values efficiently, a custom batched Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) solver was implemented in PyTorch. This enabled solving all CV folds and alphas simultaneously on the GPU, significantly reducing hyperparameter tuning time for Lasso.

---

## 2. Phase 1: Power Plant Steam Turbine Optimization (var1)
**Problem Context:** Predicting Net Power Score based on 6 operational parameters using up to a degree 10 polynomial.

### 2.1 Model Selection
Given 6 features, a degree 10 polynomial would generate over 8,000 terms, which far exceeds the 1,000 available training samples. We limited our search up to degree 7 (1,715 terms).
We evaluated both Ridge (L2) and Lasso (L1) regularization across an exponentially spaced grid of alphas.

- **Best Ridge Config:** Degree 5, $\alpha = 10$ (CV MSE: 0.4822)
- **Best Lasso Config:** Degree 5, $\alpha = 0.01$ (CV MSE: 0.3309)

Lasso significantly outperformed Ridge. The L1 penalty effectively drove many irrelevant polynomial interaction terms to exactly zero, isolating the true underlying signal.

### 2.2 Chosen Model and Rationale
- **Chosen Model:** Lasso Regression
- **Polynomial Degree:** 5
- **Regularization ($\alpha$):** 0.01

**Rationale:** The 1-SE rule confirmed that Lasso at degree 5 with $\alpha=0.01$ is optimal. The Lasso penalty reduced the model from 461 possible terms down to just 124 non-zero terms (a sparse representation). Going to degree 6 or higher yielded worse CV MSE (around 0.34) and drastically increased the number of terms.
The final model achieves an $R^2$ of approximately 0.969 on cross-validation and a training MSE of 0.2399.

### 2.3 Diagnostics & Observations
Analysis of the out-of-fold residuals showed a baseline error floor largely driven by instances where input features were clipped exactly at $-1$ or $1$ in the dataset. Down-weighting these boundary samples during cross-validation yielded consistent results, confirming that the degree 5 selection is robust to boundary noise. The largest magnitude coefficients pertained to terms like $x_1 x_5$, $x_2^3 x_3$, and $x_3 x_6$.

---

## 3. Phase 2: Subterranean Thermal Reservoir Mapping (var2)
**Problem Context:** Predicting Thermal Anomaly Score based on 3 spatial coordinate offsets using up to a degree 20 polynomial.

### 3.1 Model Selection
With only 3 features, polynomials up to degree 20 are computationally feasible (up to 1,770 terms). We evaluated degrees 1 through 20 using both Ridge and Lasso regularization. 

- **Best Ridge Config:** Degree 10, $\alpha = 1.0$ (CV MSE: 0.2473)
- **Best Lasso Config:** Degree 15, $\alpha = 0.001$ (CV MSE: 0.2493)

Unlike Phase 1, Ridge slightly outperformed Lasso and offered a more stable regularization path across high degrees.

### 3.2 Chosen Model and Rationale
- **Chosen Model:** Ridge Regression
- **Polynomial Degree:** 10
- **Regularization ($\alpha$):** 1.0

**Rationale:** The absolute minimum CV MSE was observed at degree 10 for Ridge ($\approx 0.2473$). Higher degrees (11 to 20) with optimal alpha tuning achieved very similar CV MSE values (e.g., degree 15 MSE is $\approx 0.259$). Applying the 1-SE rule favored the simpler degree 10 model over more complex alternatives. With a CV $R^2$ of 0.9945 and a training MSE of 0.1699, the model captures the high-frequency spatial variation of the thermal anomalies accurately without fitting the noise.

### 3.3 Diagnostics
For the degree 10 Ridge model, the coefficient magnitudes are evenly distributed across degrees 1 through 10, with significant interactions like $x_2^2 x_3^3$ and $x_1^2 x_2 x_3^6$. This confirms that a high-degree polynomial is indeed necessary to model the complex geological structures in this dataset.

---

## 4. Deliverables Summary
- **Report:** This document.
- **Predictions:** `BT2024052_pred_var1.csv` and `BT2024052_pred_var2.csv` containing 1000 rows each.
- **Codebase:** A complete, reproducible GitHub repository is provided at [https://github.com/Het2239/ML](https://github.com/Het2239/ML), containing the data loaders, customized GPU FISTA Lasso solver, exact Ridge path solver, and training/inference scripts.
