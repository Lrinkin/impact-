"""
Flask web app for the fintech overdraft project.

Routes:
  GET  /             web form
  POST /predict      form submission (shows result on the page)
  POST /api/predict  JSON API
  GET  /health       health check
"""

import importlib
import warnings
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

import fintech  # imports fintech.py without modifying it

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
    """Load the saved model; if incompatible or missing, train via fintech.py."""
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
    """Convert raw input to numbers and check ranges."""
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
