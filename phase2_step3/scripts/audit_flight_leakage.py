from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["from", "to", "flightType", "time", "distance", "agency",
            "travel_year", "travel_month", "travel_dayofweek", "travel_day"]

def main():
    p = argparse.ArgumentParser(description="Read-only flight data leakage and split audit.")
    p.add_argument("--project-root", default=".")
    a = p.parse_args()
    root = Path(a.project_root).resolve()
    csv_path = root / "data" / "flights.csv"
    out_dir = root / "phase2_step3" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"travelCode", "userCode", "price", "date", *FEATURES[:6]}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    dt = df["date_parsed"]
    df["travel_year"] = dt.dt.year
    df["travel_month"] = dt.dt.month
    df["travel_dayofweek"] = dt.dt.dayofweek
    df["travel_day"] = dt.dt.day

    # Match the project's feature normalization for categorical/numeric columns.
    for c in ["from", "to", "flightType", "agency"]:
        df[c] = df[c].astype("string").fillna("Unknown").astype(str)
    for c in ["time", "distance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    clean = df.dropna(subset=["price"]).copy()
    X = clean[FEATURES]
    y = clean["price"]

    # Random split mirrors train_regression.py; no model is fitted.
    idx_train, idx_test = train_test_split(clean.index, test_size=0.2, random_state=42)
    train = clean.loc[idx_train]
    test = clean.loc[idx_test]

    # Exact feature-row repetition and whether repeated feature rows carry one or many target values.
    feature_key = X.astype("string").fillna("<NA>").agg("¦".join, axis=1)
    feature_counts = feature_key.value_counts()
    repeated_keys = feature_counts[feature_counts > 1].index
    repeated_mask = feature_key.isin(repeated_keys)
    repeated_feature_groups = clean.loc[repeated_mask].assign(_key=feature_key[repeated_mask]).groupby("_key")["price"].agg(["size", "nunique"])
    exact_feature_groups = int(len(repeated_feature_groups))
    same_features_same_price_groups = int((repeated_feature_groups["nunique"] == 1).sum())
    same_features_multiple_prices_groups = int((repeated_feature_groups["nunique"] > 1).sum())

    # Group overlap: whether a travelCode appears in both random partitions.
    train_codes = set(train["travelCode"].dropna().astype(str))
    test_codes = set(test["travelCode"].dropna().astype(str))
    overlap_codes = train_codes & test_codes

    # Summarize travelCode cardinality and within-group variation.
    temp = clean.assign(_feature_key=feature_key)
    tc = temp.groupby("travelCode", dropna=False).agg(
        rows=("travelCode", "size"),
        unique_prices=("price", "nunique"),
        unique_feature_rows=("_feature_key", "nunique")
    )
    tc_multi = tc[tc["rows"] > 1]

    # Target distribution and temporal coverage for context.
    report = {
        "audit_type": "read-only flight leakage and split audit",
        "source": str(csv_path),
        "rows_total": int(len(df)),
        "rows_with_valid_price": int(len(clean)),
        "feature_columns_used_by_training": FEATURES,
        "target_column": "price",
        "target_in_feature_columns": "price" in FEATURES,
        "identifier_columns_excluded_by_training": ["travelCode", "userCode"],
        "random_split_replication": {
            "method": "train_test_split(test_size=0.2, random_state=42)",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "distinct_travel_codes_train": int(len(train_codes)),
            "distinct_travel_codes_test": int(len(test_codes)),
            "travel_codes_in_both_train_and_test": int(len(overlap_codes)),
            "overlap_share_of_test_codes": float(len(overlap_codes) / max(1, len(test_codes)))
        },
        "repeated_feature_rows": {
            "distinct_feature_groups_repeated": exact_feature_groups,
            "groups_with_one_unique_price": same_features_same_price_groups,
            "groups_with_multiple_prices": same_features_multiple_prices_groups,
            "rows_in_repeated_feature_groups": int(repeated_mask.sum())
        },
        "travelCode_grouping": {
            "distinct_travel_codes": int(clean["travelCode"].nunique(dropna=True)),
            "codes_with_multiple_rows": int(len(tc_multi)),
            "multirow_codes_with_multiple_prices": int((tc_multi["unique_prices"] > 1).sum()),
            "multirow_codes_with_multiple_feature_rows": int((tc_multi["unique_feature_rows"] > 1).sum()),
            "max_rows_per_travel_code": int(tc["rows"].max())
        },
        "date_range": {
            "unparseable_dates": int(df["date_parsed"].isna().sum()),
            "min": str(df["date_parsed"].min()),
            "max": str(df["date_parsed"].max())
        },
        "price_range": {
            "min": float(clean["price"].min()),
            "max": float(clean["price"].max()),
            "mean": float(clean["price"].mean()),
            "std": float(clean["price"].std())
        },
        "interpretation": [
            "This audit does not train or modify a model.",
            "Travel-code overlap across a random split is a potential dependence/leakage risk if rows sharing a travelCode are related.",
            "Repeated feature rows with one price may make random holdout metrics optimistic.",
            "A group-aware or chronological evaluation should be considered after interpreting these counts."
        ]
    }
    output = out_dir / "phase2_flight_leakage_audit.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved: {output}")
    print("Read-only audit complete. No model training or artifact modification occurred.")

if __name__ == "__main__":
    main()
