#!/usr/bin/env python3
"""Read-only Travel & Tourism MLOps project audit; writes reports only in this package."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
from datetime import datetime, timezone

EXPECTED = {
    "data/flights.csv": 10,
    "data/hotels.csv": 8,
    "data/users.csv": 5,
}
JSON_FILES = [
    "docs/DATASET_AUDIT.json",
    "artifacts/flight_price_metrics.json",
    "artifacts/classification_metrics.json",
    "artifacts/recommender_metadata.json",
]
EXPECTED_FILES = [
    "README.md", "requirements.txt", "Dockerfile", "Dockerfile.streamlit",
    "docker-compose.yml", "Jenkinsfile", "api/app.py", "streamlit_app/app.py",
    "src/travel_ml/train_regression.py", "src/travel_ml/train_classification.py",
    "src/travel_ml/train_recommender.py", "airflow/dags/travel_model_training.py",
    "k8s/deployment.yaml", "k8s/service.yaml", "tests/test_smoke.py",
    "artifacts/flight_price_model.joblib", "artifacts/gender_classifier.joblib",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def inspect_csv(path: Path) -> dict:
    result = {"exists": path.exists()}
    if not path.exists():
        return result
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        result.update({"columns": len(header), "header": header, "data_rows": rows})
    except Exception as exc:
        result["read_error"] = str(exc)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".", help="Path to travel-tourism-mlops project root")
    p.add_argument("--output-dir", default=None, help="Optional report output directory")
    args = p.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        print(f"Project root does not exist: {root}", file=sys.stderr)
        return 2
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(__file__).resolve().parents[1] / "reports"
    output.mkdir(parents=True, exist_ok=True)

    inventory = {}
    for rel in EXPECTED_FILES + JSON_FILES:
        fp = root / rel
        inventory[rel] = {"exists": fp.is_file(), "size_bytes": fp.stat().st_size if fp.is_file() else None}
        if fp.is_file() and fp.suffix.lower() in {".json", ".yaml", ".yml"}:
            try:
                if fp.suffix.lower() == ".json":
                    json.loads(fp.read_text(encoding="utf-8"))
                    inventory[rel]["parse_status"] = "valid JSON"
            except Exception as exc:
                inventory[rel]["parse_status"] = f"INVALID: {exc}"

    csv_results = {rel: inspect_csv(root / rel) for rel in EXPECTED}
    warnings = []
    for rel, expected_cols in EXPECTED.items():
        item = csv_results[rel]
        if not item["exists"]:
            warnings.append(f"Missing dataset: {rel}")
        elif item.get("columns") != expected_cols:
            warnings.append(f"{rel}: expected {expected_cols} columns per project baseline, found {item.get('columns')}")

    for rel, item in inventory.items():
        if not item["exists"]:
            warnings.append(f"Missing expected file: {rel}")
        if item.get("parse_status", "").startswith("INVALID"):
            warnings.append(f"Invalid JSON: {rel}")

    payload = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "scope": "read-only file inventory, CSV header/row counts, JSON syntax checks",
        "datasets": csv_results,
        "expected_files": inventory,
        "warnings": warnings,
        "limitations": [
            "Does not load or execute model binaries.",
            "Does not verify prediction quality, data leakage, API runtime, Docker, Jenkins, Airflow, or Kubernetes.",
            "Does not modify project files; reports are written only to the selected output directory."
        ]
    }
    json_path = output / "phase1_audit.json"
    txt_path = output / "phase1_audit.txt"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "TRAVEL & TOURISM MLOPS — PHASE 1 LOCAL AUDIT",
        f"Timestamp UTC: {payload['audit_timestamp_utc']}",
        f"Project root: {root}",
        "",
        "DATASETS"
    ]
    for rel, item in csv_results.items():
        lines.append(f"- {rel}: " + (f"{item.get('data_rows')} rows, {item.get('columns')} columns" if item.get("exists") else "MISSING"))
    lines += ["", "EXPECTED FILES"]
    for rel, item in inventory.items():
        lines.append(f"- [{'OK' if item['exists'] else 'MISSING'}] {rel}" +
                     (f" ({item['size_bytes']} bytes)" if item["exists"] else ""))
    lines += ["", "WARNINGS"] + ([f"- {w}" for w in warnings] if warnings else ["- None from these file-level checks."])
    lines += ["", "LIMITATIONS"] + [f"- {x}" for x in payload["limitations"]]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Audit completed.")
    print(f"JSON report: {json_path}")
    print(f"Text report: {txt_path}")
    print(f"Warnings: {len(warnings)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
