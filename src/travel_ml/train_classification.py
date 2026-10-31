import argparse, json, os
from pathlib import Path
import joblib, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.travel_ml.common import USER_REQUIRED, validate_csv

def train(data_path, output_dir):
    df = validate_csv(data_path, USER_REQUIRED, "users dataset")
    df = df.dropna(subset=["gender"]).copy()
    X = df[["company", "age"]].copy()
    y = df["gender"].astype(str)
    if y.nunique() < 2:
        raise ValueError("Gender classification requires at least two target classes.")
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    pre = ColumnTransformer([
        ("company", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), ["company"]),
        ("age", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age"])
    ])
    model = Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("travel-gender-classification")
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {"accuracy": float(accuracy_score(y_test, pred)), "macro_f1": float(f1_score(y_test, pred, average="macro"))}
        mlflow.log_params({"model":"LogisticRegression", "features":"company,age", "random_state":42})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="gender_classifier",serialization_format="cloudpickle")
        out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out/"gender_classifier.joblib")
        report=classification_report(y_test, pred, output_dict=True, zero_division=0)
        payload={"metrics":metrics,"classification_report":report,"mlflow_run_id":run.info.run_id}
        (out/"classification_metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
        print(json.dumps(payload, indent=2))
    return model

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/users.csv"); p.add_argument("--output",default="artifacts")
    a=p.parse_args(); train(a.data,a.output)
