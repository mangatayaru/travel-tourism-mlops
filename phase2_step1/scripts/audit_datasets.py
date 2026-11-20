from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

EXPECTED = {
    "flights.csv": {
        "required": ["travelCode", "userCode", "from", "to", "flightType", "price", "time", "distance", "agency", "date"],
        "numeric": ["price", "time", "distance"],
        "nonnegative": ["price", "time", "distance"],
        "date": ["date"],
        "likely_id": ["travelCode"],
    },
    "hotels.csv": {
        "required": ["travelCode", "userCode", "name", "place", "days", "price", "total", "date"],
        "numeric": ["days", "price", "total"],
        "nonnegative": ["days", "price", "total"],
        "date": ["date"],
        "likely_id": ["travelCode"],
    },
    "users.csv": {
        "required": ["code", "company", "name", "gender", "age"],
        "numeric": ["age"],
        "nonnegative": ["age"],
        "date": [],
        "likely_id": ["code"],
    },
}

def audit_csv(path: Path, spec: dict) -> dict:
    df = pd.read_csv(path, low_memory=False)
    result = {
        "file": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "required_columns_missing": [c for c in spec["required"] if c not in df.columns],
        "numeric_checks": {},
        "date_checks": {},
        "candidate_key_checks": {},
        "warnings": [],
    }

    for col in spec["numeric"]:
        if col not in df.columns:
            continue
        parsed = pd.to_numeric(df[col], errors="coerce")
        invalid = int((parsed.isna() & df[col].notna()).sum())
        result["numeric_checks"][col] = {
            "non_numeric_non_null_values": invalid,
            "min": None if parsed.dropna().empty else float(parsed.min()),
            "max": None if parsed.dropna().empty else float(parsed.max()),
            "negative_values": int((parsed < 0).sum()),
        }

    for col in spec["date"]:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        result["date_checks"][col] = {
            "unparseable_non_null_values": int((parsed.isna() & df[col].notna()).sum()),
            "min": None if parsed.dropna().empty else str(parsed.min()),
            "max": None if parsed.dropna().empty else str(parsed.max()),
        }

    for col in spec["likely_id"]:
        if col not in df.columns:
            continue
        result["candidate_key_checks"][col] = {
            "nulls": int(df[col].isna().sum()),
            "distinct_values": int(df[col].nunique(dropna=True)),
            "duplicate_values_beyond_first": int(df[col].duplicated(keep="first").sum()),
        }

    for col, n in result["missing_by_column"].items():
        if n:
            result["warnings"].append(f"{col}: {n} missing values")
    if result["exact_duplicate_rows"]:
        result["warnings"].append(f"{result['exact_duplicate_rows']} exact duplicate rows")
    if result["required_columns_missing"]:
        result["warnings"].append("Missing required columns: " + ", ".join(result["required_columns_missing"]))
    for col, check in result["numeric_checks"].items():
        if check["non_numeric_non_null_values"]:
            result["warnings"].append(f"{col}: {check['non_numeric_non_null_values']} non-numeric values")
        if check["negative_values"]:
            result["warnings"].append(f"{col}: {check['negative_values']} negative values")
    for col, check in result["date_checks"].items():
        if check["unparseable_non_null_values"]:
            result["warnings"].append(f"{col}: {check['unparseable_non_null_values']} non-parseable dates")
    return result

def main():
    parser = argparse.ArgumentParser(description="Read-only dataset quality audit.")
    parser.add_argument("--data-dir", default="data", help="Folder containing flights.csv, hotels.csv, users.csv")
    parser.add_argument("--output-dir", default="phase2_audit", help="Where to write reports")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results, missing_files = {}, []
    for filename, spec in EXPECTED.items():
        path = data_dir / filename
        if not path.is_file():
            missing_files.append(filename)
            continue
        results[filename] = audit_csv(path, spec)

    report = {
        "audit_type": "read-only dataset quality audit",
        "data_directory": str(data_dir),
        "datasets": results,
        "missing_files": missing_files,
        "summary": {
            "datasets_found": len(results),
            "datasets_expected": len(EXPECTED),
            "warning_count": sum(len(x["warnings"]) for x in results.values()) + len(missing_files),
        },
        "limitations": [
            "A candidate ID column is not assumed to be a primary key; repeated IDs may be valid for transaction data.",
            "Date parsing checks format compatibility, not business correctness.",
            "This script does not change source CSV files or train models.",
        ],
    }
    (output_dir / "phase2_dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["PHASE 2 — STEP 1 DATASET QUALITY AUDIT", f"Data folder: {data_dir}", ""]
    for filename, item in results.items():
        lines += [
            f"{filename}: {item['rows']:,} rows x {item['columns']} columns",
            f"  Exact duplicate rows: {item['exact_duplicate_rows']:,}",
            f"  Missing values total: {sum(item['missing_by_column'].values()):,}",
            f"  Missing required columns: {item['required_columns_missing'] or 'None'}",
        ]
        for warning in item["warnings"]:
            lines.append(f"  WARNING: {warning}")
        if not item["warnings"]:
            lines.append("  No issues detected by these checks.")
        lines.append("")
    if missing_files:
        lines.append("MISSING DATASETS: " + ", ".join(missing_files))
    lines += ["", "Full column-level results are in phase2_dataset_audit.json."]
    (output_dir / "phase2_dataset_audit.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReports saved in: {output_dir}")
    if missing_files:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
