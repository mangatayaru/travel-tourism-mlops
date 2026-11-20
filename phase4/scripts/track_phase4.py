from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd


def find_metrics_file(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = project_root / p
        if p.exists():
            return p
        raise FileNotFoundError(f"Metrics file not found: {p}")

    candidates = [
        project_root / "phase3" / "reports" / "phase3_metrics_comparison.csv",
        project_root / "phase3" / "reports_optimized" / "phase3_metrics_comparison.csv",
        project_root / "phase3" / "phase3_metrics_comparison.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find phase3_metrics_comparison.csv in the standard Phase 3 "
        "report folders. Pass --metrics-file with its relative or absolute path."
    )


def set_reasonable_parallelism(model) -> list[str]:
    """Limit estimator parallelism in memory only; never save/overwrite the model."""
    changed = []
    try:
        params = model.get_params(deep=True)
        updates = {name: 2 for name in params if name.endswith("n_jobs") and params[name] in (-1, None)}
        if updates:
            model.set_params(**updates)
            changed = list(updates)
    except Exception:
        pass
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Log completed Phase 3 evaluation to MLflow.")
    parser.add_argument("--project-root", default=".", help="Travel MLOps project root")
    parser.add_argument("--metrics-file", default=None, help="Optional Phase 3 metrics CSV path")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="Travel-Tourism-Flight-Price", help="MLflow experiment name")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    model_path = root / "artifacts" / "flight_price_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Existing model not found: {model_path}")

    metrics_path = find_metrics_file(root, args.metrics_file)
    metrics_df = pd.read_csv(metrics_path)

    required = {"split", "r2", "mae", "rmse"}
    missing = required - set(metrics_df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing columns: {sorted(missing)}")

    print(f"[1/5] Project root: {root}", flush=True)
    print(f"[2/5] Reading completed Phase 3 metrics: {metrics_path}", flush=True)
    print(metrics_df.to_string(index=False), flush=True)

    print("[3/5] Loading existing model (no training/retraining)...", flush=True)
    model = joblib.load(model_path)
    adjusted = set_reasonable_parallelism(model)
    if adjusted:
        print(f"Limited in-memory estimator parallelism for: {adjusted}", flush=True)

    tracking_uri = args.tracking_uri
    if not tracking_uri.startswith(("file:", "http://", "https://", "sqlite:")):
        tracking_uri = str((root / tracking_uri).resolve())
    elif tracking_uri.startswith("file:./"):
        tracking_uri = "file:" + str((root / tracking_uri[6:]).resolve())

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    print("[4/5] Logging metrics and model to MLflow...", flush=True)
    with mlflow.start_run(run_name="phase3-evaluation-summary") as parent:
        mlflow.set_tag("project", "travel-tourism-mlops")
        mlflow.set_tag("phase", "4")
        mlflow.set_tag("source_phase", "3")
        mlflow.set_tag("model_action", "log-existing-model-no-retraining")
        mlflow.log_param("evaluation_source", str(metrics_path.relative_to(root)) if metrics_path.is_relative_to(root) else str(metrics_path))
        mlflow.log_param("model_source", "artifacts/flight_price_model.joblib")
        mlflow.log_param("evaluation_splits", int(len(metrics_df)))

        for _, row in metrics_df.iterrows():
            split = str(row["split"])
            safe_name = split.replace(" ", "_")
            with mlflow.start_run(run_name=safe_name, nested=True):
                mlflow.set_tag("evaluation_split", split)
                for col in ("train_rows", "test_rows", "travel_code_overlap_with_train"):
                    if col in metrics_df.columns and pd.notna(row[col]):
                        mlflow.log_param(col, int(row[col]))
                for metric in ("r2", "mae", "rmse"):
                    value = pd.to_numeric(row[metric], errors="coerce")
                    if pd.notna(value):
                        mlflow.log_metric(metric, float(value))

        # Preserve the original model artifact. This logs a copy to MLflow;
        # it does not write to or modify the source artifact.
        mlflow.log_artifact(str(model_path), artifact_path="existing_model_joblib")
        mlflow.log_artifact(str(metrics_path), artifact_path="phase3_evaluation")

        report_json = metrics_path.with_name("phase3_evaluation_report.json")
        if report_json.exists():
            mlflow.log_artifact(str(report_json), artifact_path="phase3_evaluation")

        # Log the sklearn model when supported; the joblib file above remains
        # the guaranteed portable copy if the MLflow sklearn flavor is unavailable.
        try:
            mlflow.sklearn.log_model(model, artifact_path="sklearn_model")
            print("Logged sklearn model flavor.", flush=True)
        except Exception as exc:
            print(f"Sklearn flavor logging skipped; joblib artifact is logged. Reason: {exc}", flush=True)

        run_id = parent.info.run_id
        experiment_id = parent.info.experiment_id

    print("[5/5] Phase 4 tracking completed.", flush=True)
    print(f"Experiment: {args.experiment}", flush=True)
    print(f"Run ID: {run_id}", flush=True)
    print(f"Experiment ID: {experiment_id}", flush=True)
    print(f"Tracking URI: {tracking_uri}", flush=True)
    print("\nStart the UI with:", flush=True)
    print(f'python -m mlflow ui --backend-store-uri "{tracking_uri}" --host 127.0.0.1 --port 5000', flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nPHASE 4 FAILED: {exc}", file=sys.stderr, flush=True)
        raise
