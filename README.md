# Customer Annual Spending Prediction

Group project (SI 422, 2025) — predicting a customer's annual spending
from behavioural, demographic, and engagement features using Multiple
Linear Regression (MLR).

> **Guide:** Prof. Monika Bhattacharjee, IIT Bombay

## Overview

This project builds an interpretable MLR model to predict a customer's
**Annual Spending** at an e-commerce platform from account, engagement,
and demographic attributes, following a rigorous statistical-modelling
workflow rather than a black-box ML approach.

**Dataset:** Since no dataset was attached to the original coursework
brief, `data/generate_data.py` synthesizes a realistic 1,200-customer
dataset (demographics, membership/engagement behaviour, and a
Annual_Spending target) with injected missing values and a designed,
noisy-linear relationship — so the full pipeline below is reproducible
end-to-end from a clean environment. Swap in your own CSV (same schema)
by pointing `DATA_PATH` in `src/analysis.py` at it.

## Key Results

| Metric | Value |
|---|---|
| R² | 0.864 |
| **Adjusted R²** | **0.863** |
| Final predictors | 17 (after VIF pruning + BIC backward elimination) |
| Influential points flagged (Cook's D ∩ DFFITS) | ~5–6% of observations |
| Max VIF (final model) | 1.45 (no multicollinearity) |

## Methodology

1. **Data cleaning** — median imputation for missing values in
   `Annual_Income`, `Customer_Satisfaction`, `Email_Engagement_Rate`.
2. **EDA & Box-Cox transformation** — skewness computed for all numeric
   features; right-skewed variables (`Annual_Income`,
   `Membership_Years`, `Days_Since_Last_Purchase`,
   `Num_Purchases_Last_Year`, usage hours, and the target
   `Annual_Spending`) are Box-Cox transformed to stabilize variance and
   approximate normality.
3. **One-hot encoding** — categorical features (`Gender`, `Region`,
   `Membership_Type`, `Device_Preference`) encoded with `drop_first=True`
   to avoid the dummy-variable trap.
4. **Standard scaling** — all predictors standardized (zero mean, unit
   variance) so coefficient magnitudes are comparable and regularization
   (Lasso) behaves correctly.
5. **Multicollinearity check** — iterative Variance Inflation Factor
   (VIF) pruning, dropping any feature with VIF > 8 until all remaining
   features are below the threshold.
6. **Feature selection (three independent methods, cross-checked):**
   - **AIC/BIC-based backward elimination** — greedy backward search
     minimizing BIC at each step.
   - **k-fold cross-validation (k=5)** — validates the selected
     feature set's out-of-sample R² isn't inflated by overfitting.
   - **Lasso regularization** (`LassoCV`, 5-fold) — an independent,
     penalty-based check; the retained feature set overlaps heavily
     with the backward-elimination result.
7. **Final MLR fit** — `statsmodels.OLS` on the selected 17 features →
   **R² = 0.864, Adjusted R² = 0.863**.
8. **Influential point diagnostics:**
   - **Cook's Distance** (threshold `4/n`)
   - **DFFITS** (threshold `2·√(p/n)`)
   - Points flagged by both methods are reported as jointly influential.
9. **Assumption verification:**
   - *Linearity / homoscedasticity* — residuals-vs-fitted and
     scale-location plots.
   - *Normality of residuals* — Q-Q plot + Shapiro-Wilk test.
   - *Homoscedasticity* — Breusch-Pagan test.
   - *Independence* — Durbin-Watson statistic.
   - *Multicollinearity* — final VIF re-check.

   Full numeric output is in [`results/model_summary.txt`](results/model_summary.txt).
   As is typical with n=1,200 and several standardized dummy variables,
   the Shapiro-Wilk and Breusch-Pagan tests show statistically
   significant (but practically mild) deviations from the ideal — this
   is called out honestly in the report rather than hidden.

## Repository Structure

```
customer-spending-prediction/
├── data/
│   ├── generate_data.py        # synthesizes the dataset
│   └── customer_spending.csv   # generated dataset (1200 rows)
├── src/
│   └── analysis.py             # full pipeline (steps 1-10 above)
├── results/
│   ├── model_summary.txt       # full text report (all steps, OLS summary)
│   ├── boxcox_target.png       # target distribution before/after Box-Cox
│   ├── influence_diagnostics.png  # Cook's Distance & DFFITS plots
│   └── assumption_checks.png   # residual diagnostics (4-panel)
├── requirements.txt
└── README.md
```

## Setup & Reproduction

```bash
git clone <this-repo-url>
cd customer-spending-prediction
pip install -r requirements.txt

# 1. Generate the synthetic dataset (skip if using your own CSV)
python data/generate_data.py

# 2. Run the full analysis pipeline
python src/analysis.py
```

Running `src/analysis.py` regenerates everything in `results/` and
prints a step-by-step log to the console.

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| Age | int | Customer age |
| Gender | categorical | Male / Female |
| Region | categorical | North / South / East / West |
| Annual_Income | float | Annual income (USD) |
| Membership_Type | categorical | Basic / Silver / Gold / Platinum |
| Membership_Years | float | Years as a member |
| Device_Preference | categorical | Mobile App / Website / Both |
| Avg_Session_Length_Min | float | Average session length (minutes) |
| App_Usage_Hours_Week | float | Weekly app usage (hours) |
| Website_Usage_Hours_Week | float | Weekly website usage (hours) |
| Num_Purchases_Last_Year | int | Purchases in the last 12 months |
| Customer_Satisfaction | float | Survey score, 1–10 |
| Days_Since_Last_Purchase | float | Recency (days) |
| Email_Engagement_Rate | float | Fraction of promo emails opened |
| **Annual_Spending** | float | **Target variable** |

## Tech Stack

`pandas`, `numpy`, `scipy` (Box-Cox, Shapiro-Wilk), `statsmodels` (OLS,
VIF, Breusch-Pagan, Durbin-Watson, Cook's Distance, DFFITS),
`scikit-learn` (StandardScaler, KFold CV, LassoCV), `matplotlib` /
`seaborn` (diagnostics).

## Team

Group project for **SI 422**, IIT Bombay (2025).
Guide: **Prof. Monika Bhattacharjee**.

## License

MIT — see [`LICENSE`](LICENSE).
