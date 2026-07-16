# Customer Churn Prediction

Predicts which customers are likely to churn using demographic, account,
and billing data — trained and evaluated with scikit-learn, with a
Power BI–ready export for a risk dashboard.

## Problem
Telecom companies lose revenue when customers cancel service. Spotting
at-risk customers early lets a retention team intervene (discounts,
outreach) before they leave.

## Data
`data/telco_churn.csv` — customer demographics, contract type, billing
method, tenure, monthly/total charges, and churn label (Yes/No).

> This repo ships with a synthetic dataset (same structure, realistic
> signal) so the pipeline runs out of the box. To use the real data,
> download **Telco Customer Churn** from Kaggle (search "Telco Customer
> Churn" by blastchar) and replace `data/telco_churn.csv` — no code
> changes needed.

## Approach
1. **Clean** — handle missing `TotalCharges`, drop identifier columns,
   encode categorical features.
2. **Explore** — churn distribution, tenure vs. churn (`eda_overview.png`).
3. **Model** — Logistic Regression (scaled features) and Random Forest,
   compared on accuracy, precision, recall, F1, and ROC AUC.
4. **Evaluate** — confusion matrix and feature-importance chart for the
   best-performing model.
5. **Export** — test-set predictions with a churn risk score, ready to
   load into Power BI for a retention dashboard.

## Results (on the included dataset)
| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 68.8% | 65.7% |
| Precision | 60.8% | 52.9% |
| Recall | 34.3% | 29.6% |
| F1 score | 43.8% | 38.0% |
| ROC AUC | 0.71 | 0.68 |

**Best model: Logistic Regression.** Numbers will change (and likely
improve) once run on the real Kaggle dataset — re-run `churn_prediction.py`
after swapping the CSV to get your own numbers, and use those on your resume.

## Files
- `make_dataset.py` — generates the synthetic dataset (skip this if using real data)
- `churn_prediction.py` — full pipeline: clean → explore → train → evaluate → export
- `eda_overview.png`, `confusion_matrix.png`, `feature_importance.png` — charts
- `churn_predictions_for_powerbi.csv` — risk-scored predictions for dashboarding

## Run it
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python make_dataset.py       # only if you don't have the real Kaggle CSV
python churn_prediction.py
```

## Next steps
- Try `GridSearchCV` to tune Random Forest hyperparameters
- Add SMOTE or class weighting to improve recall on the minority (churn) class
- Build the Power BI dashboard on `churn_predictions_for_powerbi.csv`
