"""
Generates a synthetic customer-churn dataset shaped like the popular
Kaggle "Telco Customer Churn" dataset (same columns, realistic signal).

WHY SYNTHETIC: this sandbox has no internet access, so the real Kaggle
CSV can't be downloaded here. The pipeline in churn_prediction.py is
written to work on the REAL file too -- just download it from Kaggle
("Telco Customer Churn" by blastchar) and replace data/telco_churn.csv
with the real one. No code changes needed.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

contract = rng.choice(["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.25, 0.20])
tenure = rng.integers(0, 73, size=N)
monthly_charges = np.round(rng.normal(65, 20, size=N).clip(18, 120), 2)
internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=N, p=[0.35, 0.45, 0.20])
tech_support = rng.choice(["Yes", "No", "No internet service"], size=N, p=[0.30, 0.50, 0.20])
paperless_billing = rng.choice(["Yes", "No"], size=N, p=[0.6, 0.4])
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=N
)
senior_citizen = rng.choice([0, 1], size=N, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], size=N, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], size=N, p=[0.30, 0.70])
total_charges = np.round(monthly_charges * np.maximum(tenure, 1) * rng.uniform(0.9, 1.0, size=N), 2)

# Build churn probability from realistic drivers, then sample the label
logit = (
    -1.2
    + (contract == "Month-to-month") * 1.4
    + (internet_service == "Fiber optic") * 0.5
    + (tech_support == "No") * 0.6
    + (payment_method == "Electronic check") * 0.5
    + (senior_citizen == 1) * 0.3
    - (tenure / 72) * 2.0
    - (partner == "Yes") * 0.3
    + rng.normal(0, 0.5, size=N)
)
prob = 1 / (1 + np.exp(-logit))
churn = (rng.uniform(0, 1, size=N) < prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(N)],
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "Contract": contract,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn_label,
})

# Sprinkle a few missing values + blank strings, like the real dataset has
missing_idx = rng.choice(N, size=25, replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan

import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/telco_churn.csv", index=False)
print(f"Saved {len(df)} rows to data/telco_churn.csv")
print(df["Churn"].value_counts(normalize=True))
