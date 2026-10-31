import os
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from src.travel_ml.common import flight_features

app = Flask(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/flight_price_model.joblib"))
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train regression first.")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return jsonify({"status":"ok","model_loaded":MODEL_PATH.exists(),"model_path":str(MODEL_PATH)}), 200

@app.get("/")
def index():
    return jsonify({"service":"Travel Flight Price Prediction API",
                    "endpoints":{"health":"GET /health","predict":"POST /predict"}})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error":"Send a JSON object in the request body."}), 400
    required = ["from","to","flightType","time","distance","agency","date"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error":"Missing required fields","fields":missing}), 400
    try:
        row = pd.DataFrame([{k:payload[k] for k in required}])
        features = flight_features(row)
        prediction = float(get_model().predict(features)[0])
        return jsonify({"predicted_price":round(prediction,2),"currency":"dataset currency (not specified)",
                        "model":"flight_price_model.joblib"}), 200
    except FileNotFoundError as e:
        return jsonify({"error":str(e)}), 503
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error":"Prediction failed","detail":str(e)}), 400

if __name__ == "__main__":
    app.run(host=os.getenv("FLASK_HOST","127.0.0.1"), port=int(os.getenv("FLASK_PORT","5000")), debug=False)
