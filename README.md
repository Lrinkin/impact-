Setup & Installation
1. Clone the Repository
Bash
git clone [https://github.com/](https://github.com [https://github.com/Lrinkin/impact.git]
cd (https://github.com/Lrinkin/impact.git)
2. Set Up Virtual Environment
Bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
How to Run
Train the Model & Generate Data
Bash
python fintech.py
Launch the Flask Web Application
Bash
python flask_app.py
API Documentation
Endpoint
POST /api/predict

Request Header
Content-Type: application/json

Example Request Body
JSON
{
  "balance_5_days_before_payday": 120.50,
  "average_daily_spend": 35.00,
  "number_of_recurring_bills": 4,
  "days_until_payday": 7
}
Example Response Body
JSON
{
  "label": "Likely overdraft",
  "prediction": 1,
  "probability": 0.842
}
Example curl Command
Bash
curl -X POST [http://127.0.0.1:5000/api/predict](http://127.0.0.1:5000/api/predict) \
  -H "Content-Type: application/json" \
  -d '{
    "balance_5_days_before_payday": 120.50,
    "average_daily_spend": 35.00,
    "number_of_recurring_bills": 4,
    "days_until_payday": 7
  }'
EOF
'''

print("Bash script generated successfully length:", len(bash_script))


```text?code_stdout&code_event_index=1
Bash script generated successfully length: 9878

To create all missing files (requirements.txt, .gitignore, flask_app.py, and README.md) at once, open a Bash Console in PythonAnywhere, navigate to your project directory, and paste the following single script block:

Bash
cd /home/Lerin2026/mydemo

cat << 'EOF' > requirements.txt
flask>=2.3.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
openpyxl>=3.1.0
gunicorn>=21.0.0
EOF

cat << 'EOF' > .gitignore
__pycache__/
*.py[cod]
*$py.class
.env
.venv
env/
venv/
.DS_Store
Thumbs.db
EOF

cat << 'EOF' > flask_app.py
"""
Flask web application and REST API for the fintech overdraft risk predictor.
"""

import importlib
import warnings
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

import fintech

FEATURES = getattr(fintech, "FEATURE_COLUMNS", getattr(fintech, "feature_columns", None))
MODEL_PATH = getattr(fintech, "MODEL_PATH", Path(__file__).parent / "fintech.joblib")

LIMITS = {
    "balance_5_days_before_payday": (0, 100000),
    "average_daily_spend": (0, 10000),
    "number_of_recurring_bills": (0, 12),
    "days_until_payday": (1, 30),
}

DEFAULT_VALUES = {
    "balance_5_days_before_payday": 300,
    "average_daily_spend": 40,
    "number_of_recurring_bills": 4,
    "days_until_payday": 10,
}


def load_model():
    sample = pd.DataFrame([DEFAULT_VALUES], columns=FEATURES)

    if MODEL_PATH.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loaded = joblib.load(MODEL_PATH)
                loaded.predict_proba(sample)
            print("Loaded model from %s" % MODEL_PATH.name)
            return loaded
        except Exception as exc:
            print("Could not use %s (%s); retraining." % (MODEL_PATH.name, exc.__class__.__name__))

    print("Training model with fintech.py code...")
    if hasattr(fintech, "train_model"):
        trained = fintech.train_model()
    else:
        importlib.reload(fintech)
        trained = getattr(fintech, "model", None)

    try:
        joblib.dump(trained, MODEL_PATH)
        print("Saved new %s" % MODEL_PATH.name)
    except OSError as exc:
        print("Could not save model (%s); using in memory." % exc)
    return trained


app = Flask(__name__)
model = load_model()

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Overdraft Predictor</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #f4f6f8; margin: 0; padding: 2rem; color: #1f2933; }
    .card { max-width: 480px; margin: 0 auto; background: #fff; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.08); }
    h1 { font-size: 1.4rem; margin-top: 0; }
    label { display: block; margin-top: 1rem; font-weight: 600; font-size: .9rem; }
    input { width: 100%; padding: .6rem; margin-top: .3rem; border: 1px solid #cbd2d9; border-radius: 6px; box-sizing: border-box; }
    button { margin-top: 1.5rem; width: 100%; padding: .8rem; background: #2563eb; color: #fff; border: 0; border-radius: 6px; font-size: 1rem; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .result { margin-top: 1.5rem; padding: 1rem; border-radius: 8px; }
    .high { background: #fee2e2; color: #991b1b; }
    .low { background: #dcfce7; color: #166534; }
    .error { background: #fef3c7; color: #92400e; margin-top: 1rem; padding: 1rem; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Overdraft before payday</h1>
    <p>Enter the customer's details to estimate the risk of the account going negative before the next salary.</p>
    <form method="post" action="/predict">
      <label>Balance 5 days before payday (EUR)
        <input type="number" step="0.01" min="0" name="balance_5_days_before_payday" value="{{ values.balance_5_days_before_payday }}" required>
      </label>
      <label>Average daily spend (EUR)
        <input type="number" step="0.01" min="0" name="average_daily_spend" value="{{ values.average_daily_spend }}" required>
      </label>
      <label>Number of recurring bills (0 to 12)
        <input type="number" step="1" min="0" max="12" name="number_of_recurring_bills" value="{{ values.number_of_recurring_bills }}" required>
      </label>
      <label>Days until payday (1 to 30)
        <input type="number" step="1" min="1" max="30" name="days_until_payday" value="{{ values.days_until_payday }}" required>
      </label>
      <button type="submit">Predict</button>
    </form>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if result %}
      <div class="result {{ 'high' if result.prediction == 1 else 'low' }}">
        <strong>{{ result.label }}</strong><br>
        Probability of overdraft: {{ "%.1f"|format(result.probability * 100) }}%
      </div>
    {% endif %}
  </div>
</body>
</html>"""


def parse_and_validate(data):
    clean = {}
    for name in FEATURES:
        if name not in data or data[name] in (None, ""):
            raise ValueError("Missing value for '%s'." % name)
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            raise ValueError("'%s' must be a number." % name)
        low, high = LIMITS[name]
        if not (low <= value <= high):
            raise ValueError("'%s' must be between %s and %s." % (name, low, high))
        clean[name] = value
    return clean


def run_model(clean):
    df = pd.DataFrame([clean], columns=FEATURES)
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {
        "prediction": prediction,
        "label": "Likely overdraft" if prediction == 1 else "No overdraft expected",
        "probability": probability,
    }


@app.route("/", methods=["GET"])
def home():
    return render_template_string(PAGE, values=DEFAULT_VALUES, result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict_form():
    values = {name: request.form.get(name, "") for name in FEATURES}
    try:
        result = run_model(parse_and_validate(values))
        return render_template_string(PAGE, values=values, result=result, error=None)
    except ValueError as exc:
        return render_template_string(PAGE, values=values, result=None, error=str(exc)), 400


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with the four features."}), 400
    try:
        return jsonify(run_model(parse_and_validate(data)))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False)
EOF

cat << 'EOF' > README.md
# Predicting Overdraft Before Payday with Logistic Regression

A fintech machine learning web application and REST API that predicts whether a retail bank customer's current account will go negative before their next salary arrives.

## Overview & Problem Statement

Many customers run short of money in the days leading up to payday. If a bank can flag these customers early, it can send a timely alert, offer a short-term buffer, or suggest budgeting actions before overdraft fees occur.

- **Users:** Retail bank customers who receive a monthly salary in their account
- **Task:** Binary classification (1 = Likely overdraft, 0 = No overdraft expected)
- **Model:** Logistic Regression (with standardized features via scikit-learn Pipeline)
- **Target:** overdraft_before_payday

## Features & Predictors

| Feature | Type | Range / Constraints | Description |
|---|---|---|---|
| balance_5_days_before_payday | Continuous | 0.00 - 100,000.00 EUR | Current account balance 5 days before salary arrival |
| average_daily_spend | Continuous | 0.00 - 10,000.00 EUR | Average daily spending over recent cycles |
| number_of_recurring_bills | Discrete | 0 - 12 | Number of recurring bills due before payday |
| days_until_payday | Discrete | 1 - 30 Days | Days remaining until next salary arrives |

## Dataset

All data is synthetic and generated reproducibly using NumPy with a fixed seed (42):
- 5,000 customers sampled from Gamma, Poisson, and Uniform distributions.
- Logistic Process Target: Derived from the gap between expected cash needed (daily spend over remaining days plus unpaid bills) and current balance, plus random noise.
- Calibrated intercept resulting in a realistic ~30% overdraft rate.

## Project Structure

```text
.
├── .gitignore
├── README.md
├── fintech.py          # Data generation and model training pipeline
├── flask_app.py        # Flask web UI and JSON API
├── fintech.joblib      # Saved trained model pipeline
├── fintech.xlsx        # Exported synthetic dataset
└── requirements.txt    # Dependencies
Setup & Installation
1. Clone the Repository
Bash
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd <YOUR_REPOSITORY_NAME>
2. Set Up Virtual Environment
Bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
How to Run
Train the Model & Generate Data
Bash
python fintech.py
Launch the Flask Web Application
Bash
python flask_app.py
API Documentation
Endpoint
POST /api/predict

Request Header
Content-Type: application/json

Example Request Body
JSON
{
  "balance_5_days_before_payday": 120.50,
  "average_daily_spend": 35.00,
  "number_of_recurring_bills": 4,
  "days_until_payday": 7
}
Example Response Body
JSON
{
  "label": "Likely overdraft",
  "prediction": 1,
  "probability": 0.842
}
Example curl Command
Bash
curl -X POST [http://127.0.0.1:5000/api/predict](http://127.0.0.1:5000/api/predict) \
  -H "Content-Type: application/json" \
  -d '{
    "balance_5_days_before_payday": 120.50,
    "average_daily_spend": 35.00,
    "number_of_recurring_bills": 4,
    "days_until_payday": 7
  }'
EOF

python fintech.py
