from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load trained model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "cvd_model.joblib"
model_art = joblib.load(MODEL_PATH)
model = model_art["model"]
features = model_art["features"]
threshold = model_art["threshold"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    X = pd.DataFrame([data], columns=features)
    prob = model.predict_proba(X)[:, 1][0]
    label = int(prob >= threshold)
    return jsonify({"probability": float(prob), "label": label})

@app.route("/predict_form", methods=["POST"])
def predict_form():
    form_data = {f: float(request.form[f]) for f in features}
    X = pd.DataFrame([form_data], columns=features)
    prob = model.predict_proba(X)[:, 1][0]
    label = int(prob >= threshold)
    return render_template("index.html", result={"probability": round(float(prob), 3), "label": label})

if __name__ == "__main__":
    app.run(debug=True)
