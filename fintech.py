"""
Fintech project: predicting overdraft before payday with logistic regression.

Problem : Will a current account go negative before the next salary arrives?
Users   : Retail bank customers who receive a monthly salary in their account.
Task    : Binary classification (logistic regression).
Target  : overdraft_before_payday (1 = account goes negative, 0 = it does not).
Features: balance_5_days_before_payday, average_daily_spend,
          number_of_recurring_bills, days_until_payday.

All data is synthetic and generated with numpy using a fixed seed, so every
run produces exactly the same dataset, model and metrics.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
n_customers = 5000

# ---------------------------------------------------------------------------
# 2. Generate the predictor variables randomly with numpy
# ---------------------------------------------------------------------------
# Continuous: balance 5 days before payday (EUR). Right-skewed, floored at 0
# so the data represents customers who are still in positive territory.
balance_5_days_before_payday = np.round(rng.gamma(shape=2.0, scale=250.0, size=n_customers), 2)

# Continuous: average daily spend (EUR), always positive.
average_daily_spend = np.round(rng.gamma(shape=4.0, scale=11.0, size=n_customers), 2)

# Discrete: number of recurring bills (rent, utilities, subscriptions...), 0 to 12.
number_of_recurring_bills = np.clip(rng.poisson(lam=4.0, size=n_customers), 0, 12)

# Discrete: days until payday, 1 to 30.
days_until_payday = rng.integers(low=1, high=31, size=n_customers)

# ---------------------------------------------------------------------------
# 3. Generate the binary target from a realistic logistic process
# ---------------------------------------------------------------------------
# Expected cash needed until payday = daily spend over the remaining days plus
# a typical charge per recurring bill that has not been paid yet. The bigger
# the gap between that need and the current balance, the likelier the overdraft.
avg_bill_amount = 35.0
expected_need = (
    average_daily_spend * days_until_payday
    + number_of_recurring_bills * avg_bill_amount
)
shortfall_score = (expected_need - balance_5_days_before_payday) / 250.0

# Add some unexplained noise so the problem is realistic and not perfectly separable.
noise = rng.normal(loc=0.0, scale=0.6, size=n_customers)
logit = shortfall_score + noise

# Calibrate the intercept so that roughly 30% of customers overdraft.
low, high = -20.0, 20.0
for _ in range(60):
    mid = (low + high) / 2
    mean_prob = (1 / (1 + np.exp(-(logit + mid)))).mean()
    if mean_prob > 0.30:
        high = mid
    else:
        low = mid
intercept = (low + high) / 2

probability_overdraft = 1 / (1 + np.exp(-(logit + intercept)))
overdraft_before_payday = rng.binomial(n=1, p=probability_overdraft)

# ---------------------------------------------------------------------------
# 4. Build the data frame and save it to Excel (no index)
# ---------------------------------------------------------------------------
df = pd.DataFrame(
    {
        "balance_5_days_before_payday": balance_5_days_before_payday,
        "average_daily_spend": average_daily_spend,
        "number_of_recurring_bills": number_of_recurring_bills,
        "days_until_payday": days_until_payday,
        "overdraft_before_payday": overdraft_before_payday,
    }
)

df.to_excel("fintech.xlsx", index=False)
print(f"Dataset saved to fintech.xlsx: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Overdraft rate: {df['overdraft_before_payday'].mean():.1%}\n")

# ---------------------------------------------------------------------------
# 5. Train / test split (stratified so both sets keep the same overdraft rate)
# ---------------------------------------------------------------------------
feature_columns = [
    "balance_5_days_before_payday",
    "average_daily_spend",
    "number_of_recurring_bills",
    "days_until_payday",
]
x = df[feature_columns]
y = df["overdraft_before_payday"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=SEED, stratify=y
)

# ---------------------------------------------------------------------------
# 6. Logistic regression (features are standardised inside the pipeline)
# ---------------------------------------------------------------------------
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=SEED),
)
model.fit(x_train, y_train)

# ---------------------------------------------------------------------------
# 7. Validate with classification report on train and test
# ---------------------------------------------------------------------------
target_names = ["no_overdraft", "overdraft"]

print("=== Classification report: TRAIN ===")
print(classification_report(y_train, model.predict(x_train), target_names=target_names))

print("=== Classification report: TEST ===")
print(classification_report(y_test, model.predict(x_test), target_names=target_names))

# Coefficients (on standardised variables) to check the model learned sensible signs.
coefficients = pd.Series(
    model.named_steps["logisticregression"].coef_[0], index=feature_columns
).round(3)
print("Standardised coefficients:")
print(coefficients.to_string())

# ---------------------------------------------------------------------------
# 8. Dump the model with joblib
# ---------------------------------------------------------------------------
joblib.dump(model, "fintech.joblib")
print("\nModel saved to fintech.joblib")
