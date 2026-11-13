from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

RAW_FEATURES = ["from", "to", "flightType", "time", "distance", "agency", "date"]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts" / "flight_price_model.joblib"))
COMMON_PATH = PROJECT_ROOT / "src" / "travel_ml" / "common.py"

def load_feature_function():
    if not COMMON_PATH.is_file():
        raise FileNotFoundError(f"Expected common module at {COMMON_PATH}")
    spec = importlib.util.spec_from_file_location("travel_ml_common", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "flight_features"):
        raise AttributeError("src/travel_ml/common.py must expose flight_features(df)")
    return module.flight_features

def create_app():
    app = Flask(__name__)
    model = None
    feature_fn = None
    load_error = None
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        feature_fn = load_feature_function()
    except Exception as exc:
        load_error = str(exc)

    def prepare_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
        for i, row in enumerate(records):
            missing = [c for c in RAW_FEATURES if c not in row or row[c] in (None, "")]
            if missing:
                raise ValueError(f"Record {i}: missing required fields: {', '.join(missing)}")
        frame = pd.DataFrame(records)
        # The shared project feature function may validate the full training schema.
        # These placeholders are internal only; they are not model features.
        for col, default in {"travelCode": "API_REQUEST", "userCode": "API_REQUEST", "price": 0.0}.items():
            if col not in frame.columns:
                frame[col] = default
        transformed = feature_fn(frame.copy())
        if not isinstance(transformed, pd.DataFrame):
            transformed = pd.DataFrame(transformed)
        transformed = transformed.drop(columns=["travelCode", "userCode", "price"], errors="ignore")
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            missing = [c for c in expected if c not in transformed.columns]
            if missing:
                raise ValueError(f"Feature transformation did not produce model columns: {missing}")
            transformed = transformed.loc[:, list(expected)]
        return transformed

    @app.get("/health")
    def health():
        ok = model is not None and feature_fn is not None
        return jsonify({"status": "ok" if ok else "error", "model_loaded": model is not None,
                        "feature_function_loaded": feature_fn is not None,
                        "error": load_error}), (200 if ok else 503)

    @app.get("/model-info")
    def model_info():
        if model is None:
            return jsonify({"error": load_error or "Model unavailable"}), 503
        names = getattr(model, "feature_names_in_", None)
        return jsonify({"model_type": type(model).__name__, "model_path": str(MODEL_PATH),
                        "n_features": int(getattr(model, "n_features_in_", len(names) if names is not None else 0)),
                        "feature_names": list(names) if names is not None else None})

    @app.post("/predict")
    def predict():
        if model is None or feature_fn is None:
            return jsonify({"error": load_error or "Model unavailable"}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object."}), 400
        records = payload.get("records", [payload])
        if not isinstance(records, list) or not records or not all(isinstance(x, dict) for x in records):
            return jsonify({"error": "'records' must be a non-empty list of JSON objects."}), 400
        try:
            features = prepare_frame(records)
            values = np.asarray(model.predict(features), dtype=float).reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError("Model returned a non-finite prediction.")
            return jsonify({"predictions": [{"record": i, "predicted_price": round(float(v), 2)}
                                             for i, v in enumerate(values)],
                            "count": len(values)})
        except (ValueError, TypeError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("Prediction failed")
            return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=False)
