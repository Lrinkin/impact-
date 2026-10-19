# Predicting Overdraft Before Payday with Logistic Regression

A small fintech machine learning project that predicts whether a retail bank customer's current account will go negative before their next salary arrives.

## Problem

Many customers run short of money in the days before payday. If a bank can flag these customers early, it can send a timely alert, offer a short-term buffer, or suggest budgeting actions before an overdraft fees situation occurs.

- **Users:** Retail bank customers who receive a monthly salary in their account
- **Task:** Binary classification
- **Model:** Logistic regression (with standardised features)
- **Target:** `overdraft_before_payday` (1 = account goes negative, 0 = it does not)

## Features

| Feature | Type | Description |
|---|---|---|
| `balance_5_days_before_payday` | Continuous | Account balance (EUR) five days before payday |
| `average_daily_spend` | Continuous | Average daily spending (EUR) |
| `number_of_recurring_bills` | Discrete | Number of recurring bills (0 to 12) |
| `days_until_payday` | Discrete | Days remaining until the next salary (1 to 30) |

## Data

All data is **synthetic**, generated with NumPy using a fixed seed (`42`), so every run produces exactly the same dataset, model and metrics.

- 5,000 customers
- Predictors drawn from gamma, Poisson and uniform distributions
- The target is generated from a logistic process: the larger the gap between expected cash needed until payday (daily spend over the remaining days plus unpaid recurring bills) and the current balance, the higher the overdraft probability
- Random noise is added so the problem is realistic and not perfectly separable
- The intercept is calibrated so that roughly **30%** of customers overdraft

## Project Structure
