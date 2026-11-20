from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path

def describe_model(model, label):
    info = {
        "label": label,
        "python_type": f"{type(model).__module__}.{type(model).__name__}",
        "has_predict": callable(getattr(model, "predict", None)),
        "n_features_in": getattr(model, "n_features_in_", None),
        "feature_names_in": None,
        "pipeline_steps": None,
    }
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        info["feature_names_in"] = [str(x) for x in names]
    if hasattr(model, "named_steps"):
        info["pipeline_steps"] = list(model.named_steps.keys())
        # Also inspect the final estimator without assuming its input is raw data.
        final = model.named_steps.get("model")
        if final is not None:
            info["final_estimator_type"] = f"{type(final).__module__}.{type(final).__name__}"
            info["final_estimator_n_features_in"] = getattr(final, "n_features_in_", None)
    return info

def main():
    parser = argparse.ArgumentParser(description="Load and inspect existing model artifacts; no training or writes to artifacts.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    artifact_dir = root / "artifacts"
    report_dir = root / "phase2_step2" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {"project_root": str(root), "models": {}, "warnings": [], "flight_prediction_test": {}}
    try:
        import joblib
        import pandas as pd
    except Exception as exc:
        print(f"Missing dependency: {exc}")
        print("Stop here and share this error. Do not install or upgrade packages yet.")
        return 2

    model_files = {
        "flight_price": artifact_dir / "flight_price_model.joblib",
        "gender_classifier": artifact_dir / "gender_classifier.joblib",
    }
    loaded = {}
    for key, path in model_files.items():
        if not path.is_file():
            results["warnings"].append(f"Missing artifact: {path}")
            continue
        try:
            obj = joblib.load(path)
            loaded[key] = obj
            results["models"][key] = {"artifact": str(path), **describe_model(obj, key), "load_status": "SUCCESS"}
        except Exception as exc:
            results["models"][key] = {"artifact": str(path), "load_status": "FAILED", "error": repr(exc)}
            results["warnings"].append(f"Could not load {key}: {exc!r}")

    # Use the project's own feature function to ensure the prediction input matches training.
    flight_model = loaded.get("flight_price")
    if flight_model is not None:
        try:
            sys.path.insert(0, str(root))
            from src.travel_ml.common import flight_features
            flights_path = root / "data" / "flights.csv"
            df = pd.read_csv(flights_path, nrows=5)
            features = flight_features(df.copy())
            prediction = flight_model.predict(features)
            results["flight_prediction_test"] = {
                "status": "SUCCESS",
                "input_rows": int(len(features)),
                "input_columns": [str(c) for c in features.columns],
                "prediction_count": int(len(prediction)),
                "sample_predictions": [float(x) for x in prediction[:5]],
                "note": "Predictions are a load/shape smoke test, not an accuracy or leakage evaluation."
            }
        except Exception as exc:
            results["flight_prediction_test"] = {"status": "FAILED", "error": repr(exc)}
            results["warnings"].append("Flight prediction smoke test failed; share the full error.")

    gender_model = loaded.get("gender_classifier")
    if gender_model is not None:
        results["models"]["gender_classifier"]["prediction_test"] = (
            "Not run in this step: the classifier's exact training feature transformation must be checked first."
        )

    output = report_dir / "phase2_model_load_report.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nReport saved to: {output}")
    print("\nNo model was trained, modified, or overwritten.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
