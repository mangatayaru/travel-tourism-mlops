#!/usr/bin/env python
"""Read-only Phase 3 evaluation of the existing flight-price model.

Runs random, travelCode-grouped, and chronological holdouts. It does not fit,
retrain, or overwrite any model. Reports and prediction CSVs are written only
under phase3/reports/.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Cannot JSON encode {type(value)}")


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rows": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "actual_mean": float(np.mean(y_true)),
        "predicted_mean": float(np.mean(y_pred)),
        "mean_absolute_percentage_error_pct": float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
        ),
    }


def evaluate_split(name, model, X, y, row_ids, groups, out_dir):
    pred = model.predict(X)
    metrics = calculate_metrics(y, pred)
    metrics["split"] = name
    if groups is not None:
        metrics["distinct_travel_codes"] = int(pd.Series(groups).nunique())
    predictions = pd.DataFrame({
        "row_index": np.asarray(row_ids),
        "actual_price": np.asarray(y, dtype=float),
        "predicted_price": np.asarray(pred, dtype=float),
    })
    predictions["absolute_error"] = (
        predictions["actual_price"] - predictions["predicted_price"]
    ).abs()
    predictions.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project root folder")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    data_path = root / "data" / "flights.csv"
    model_path = root / "artifacts" / "flight_price_model.joblib"
    # The project's shared module is located at src/travel_ml/common.py.
    # Import from that exact package folder; do not scan the project tree.
    package_dir = root / "src" / "travel_ml"
    common_file = package_dir / "common.py"
    if not common_file.is_file():
        raise FileNotFoundError(
            f"Expected common.py at {common_file}, but it was not found. "
            "Confirm that src\\travel_ml\\common.py exists in the project."
        )
    sys.path.insert(0, str(package_dir))
    sys.path.insert(0, str(root))

    missing = [str(p) for p in (data_path, model_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Required project files not found: " + ", ".join(missing))

    try:
        from common import flight_features
    except Exception as exc:
        raise RuntimeError(
            f"Could not import flight_features from {common_file}. "
            "Confirm that src\\common.py defines flight_features(df)."
        ) from exc

    out_dir = root / "phase3" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    required = {"travelCode", "price", "date"}
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError(f"flights.csv is missing required columns: {absent}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    valid = df["price"].notna() & np.isfinite(pd.to_numeric(df["price"], errors="coerce")) & df["date"].notna()
    df = df.loc[valid].reset_index(drop=True)
    y = pd.to_numeric(df["price"], errors="raise").astype(float)
    X = flight_features(df)
    if len(X) != len(df):
        raise ValueError("flight_features returned a different number of rows than the filtered dataset.")
    if X.isna().any().any():
        warnings.warn("Feature matrix contains missing values; the existing pipeline must handle them.")

    model = joblib.load(model_path)
    all_idx = np.arange(len(df))
    results = []
    prediction_frames = {}

    # 1) Random row split: directly replicates the existing training split.
    train_idx, test_idx = train_test_split(
        all_idx, test_size=args.test_size, random_state=42
    )
    random_groups = df.iloc[test_idx]["travelCode"].astype(str).to_numpy()
    m, p = evaluate_split(
        "random_row", model, X.iloc[test_idx], y.iloc[test_idx],
        test_idx, random_groups, out_dir
    )
    m["train_rows"] = int(len(train_idx))
    m["test_rows"] = int(len(test_idx))
    m["travel_code_overlap_with_train"] = int(
        set(df.iloc[train_idx]["travelCode"].astype(str))
        .intersection(set(random_groups))
        .__len__()
    )
    results.append(m)
    prediction_frames["random_row"] = p

    # 2) Group-aware split: no travelCode may appear in both partitions.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    group_train, group_test = next(
        splitter.split(X, y, groups=df["travelCode"].astype(str))
    )
    group_test_codes = set(df.iloc[group_test]["travelCode"].astype(str))
    group_train_codes = set(df.iloc[group_train]["travelCode"].astype(str))
    m, p = evaluate_split(
        "group_aware", model, X.iloc[group_test], y.iloc[group_test],
        group_test, df.iloc[group_test]["travelCode"].astype(str).to_numpy(), out_dir
    )
    m["train_rows"] = int(len(group_train))
    m["test_rows"] = int(len(group_test))
    m["travel_code_overlap_with_train"] = int(len(group_train_codes & group_test_codes))
    results.append(m)
    prediction_frames["group_aware"] = p

    # 3) Chronological split. Determine cutoff by row-date quantile, then purge
    # training rows whose travelCode occurs in the future test partition.
    cutoff = df["date"].quantile(1.0 - args.test_size)
    chronological_test = np.flatnonzero(df["date"].ge(cutoff).to_numpy())
    chronological_train_initial = np.flatnonzero(df["date"].lt(cutoff).to_numpy())
    future_codes = set(df.iloc[chronological_test]["travelCode"].astype(str))
    chronological_train = np.asarray([
        i for i in chronological_train_initial
        if str(df.iloc[i]["travelCode"]) not in future_codes
    ], dtype=int)
    if len(chronological_train) == 0 or len(chronological_test) == 0:
        raise ValueError("Chronological split produced an empty train or test partition.")

    m, p = evaluate_split(
        "chronological_purged", model, X.iloc[chronological_test],
        y.iloc[chronological_test], chronological_test,
        df.iloc[chronological_test]["travelCode"].astype(str).to_numpy(), out_dir
    )
    m["train_rows"] = int(len(chronological_train))
    m["test_rows"] = int(len(chronological_test))
    m["cutoff_date"] = cutoff.isoformat()
    m["train_date_max"] = df.iloc[chronological_train]["date"].max().isoformat()
    m["test_date_min"] = df.iloc[chronological_test]["date"].min().isoformat()
    m["travel_code_overlap_with_train"] = int(
        len(set(df.iloc[chronological_train]["travelCode"].astype(str)) & future_codes)
    )
    m["rows_removed_from_earlier_period_to_prevent_group_overlap"] = int(
        len(chronological_train_initial) - len(chronological_train)
    )
    results.append(m)
    prediction_frames["chronological_purged"] = p

    comparison = pd.DataFrame(results)
    comparison.to_csv(out_dir / "phase3_metrics_comparison.csv", index=False)

    error_summary = {}
    for name, frame in prediction_frames.items():
        error_summary[name] = {
            "median_absolute_error": float(frame["absolute_error"].median()),
            "p90_absolute_error": float(frame["absolute_error"].quantile(0.90)),
            "p95_absolute_error": float(frame["absolute_error"].quantile(0.95)),
            "max_absolute_error": float(frame["absolute_error"].max()),
            "within_5_percent_pct": float(
                (frame["absolute_error"] / frame["actual_price"].abs().clip(lower=1e-9) <= 0.05).mean() * 100
            ),
            "within_10_percent_pct": float(
                (frame["absolute_error"] / frame["actual_price"].abs().clip(lower=1e-9) <= 0.10).mean() * 100
            ),
        }

    report = {
        "phase": "Phase 3 - Existing model evaluation",
        "evaluation_only": True,
        "model_path": str(model_path),
        "dataset_path": str(data_path),
        "model_retrained": False,
        "model_artifact_modified": False,
        "rows_evaluated": int(len(df)),
        "feature_columns": list(X.columns),
        "target_column": "price",
        "date_range": {
            "min": df["date"].min().isoformat(),
            "max": df["date"].max().isoformat(),
        },
        "split_results": results,
        "error_analysis": error_summary,
        "interpretation_notes": [
            "All metrics are calculated by applying the existing saved model; this script does not fit or retrain it.",
            "The random row split can place records sharing a travelCode in both partitions.",
            "The group-aware split keeps every travelCode entirely in one partition.",
            "The chronological split tests later dates and purges earlier training rows whose travelCode appears in the future test partition.",
            "The chronological test set may be smaller or larger than the nominal test fraction because the split is date-based.",
            "Compare all three results before deciding which validation strategy best represents deployment.",
            "A saved model evaluated on a new split is not equivalent to a model trained on that split; use these results as diagnostic holdout evaluations, not as final unbiased estimates of a newly trained model."
        ],
    }
    report_path = out_dir / "phase3_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=json_safe), encoding="utf-8")

    print("\nPHASE 3 EVALUATION COMPLETED")
    print(comparison[["split", "train_rows", "test_rows", "r2", "mae", "rmse", "travel_code_overlap_with_train"]].to_string(index=False))
    print(f"\nReport: {report_path}")
    print(f"Metrics CSV: {out_dir / 'phase3_metrics_comparison.csv'}")
    print("Prediction CSVs: random_row_predictions.csv, group_aware_predictions.csv, chronological_purged_predictions.csv")
    print("\nNo model was trained or modified.")


if __name__ == "__main__":
    main()
