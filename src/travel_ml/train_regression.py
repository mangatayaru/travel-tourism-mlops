import argparse, json, os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.travel_ml.common import FLIGHT_REQUIRED, validate_csv, flight_features

def build_pipeline():
    categorical = ["from", "to", "flightType", "agency"]
    numeric = ["time", "distance", "travel_year", "travel_month", "travel_dayofweek", "travel_day"]
    prep = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), categorical),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric)
    ])
    return Pipeline([("preprocess", prep),
                     ("model", RandomForestRegressor(n_estimators=120, min_samples_leaf=2,
                                                     random_state=42, n_jobs=-1))])

def train(data_path, output_dir, sample_size=None):
    df = validate_csv(data_path, FLIGHT_REQUIRED, "flights dataset")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] >= 0].copy()
    if df.empty:
        raise ValueError("No non-negative numeric flight prices remain.")
    # Sample only if explicitly requested; preserves manageable local experimentation.
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    X = flight_features(df)
    y = df["price"].astype(float)
    # Random split is a baseline; for forecasting, use a chronological split instead.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_pipeline()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("travel-flight-price-regression")
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test))
        }
        mlflow.log_params({"model": "RandomForestRegressor", "n_estimators": 120,
                           "test_size": 0.2, "random_state": 42, "features": ",".join(X.columns)})
        mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v, (int,float))})
        mlflow.sklearn.log_model(model, artifact_path="flight_price_model",serialization_format="cloudpickle")
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        artifact = out / "flight_price_model.joblib"
        joblib.dump(model, artifact)
        metadata = {"model_type":"RandomForestRegressor", "target":"price", "features":list(X.columns),
                    "metrics":metrics, "mlflow_run_id":run.info.run_id, "training_rows":int(len(df))}
        (out / "flight_price_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    return model, metadata

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/flights.csv")
    p.add_argument("--output", default="artifacts")
    p.add_argument("--sample-size", type=int, default=None, help="Optional row cap for quicker local experiments")
    a = p.parse_args()
    train(a.data, a.output, a.sample_size)
