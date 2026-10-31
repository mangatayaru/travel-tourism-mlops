from pathlib import Path
import json
import pandas as pd

FLIGHT_REQUIRED = ["travelCode", "userCode", "from", "to", "flightType", "price", "time", "distance", "agency", "date"]
HOTEL_REQUIRED = ["travelCode", "userCode", "name", "place", "days", "price", "total", "date"]
USER_REQUIRED = ["code", "company", "name", "gender", "age"]

def validate_csv(path, required_columns, label="dataset"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    missing_cols = sorted(set(required_columns) - set(df.columns))
    if missing_cols:
        raise ValueError(f"{label} is missing required columns: {missing_cols}")
    if df.empty:
        raise ValueError(f"{label} contains no rows")
    return df

def flight_features(df):
    """Return inference/training features. IDs are deliberately excluded."""
    out = df[["from", "to", "flightType", "time", "distance", "agency", "date"]].copy()
    dt = pd.to_datetime(out.pop("date"), errors="coerce")
    out["travel_year"] = dt.dt.year
    out["travel_month"] = dt.dt.month
    out["travel_dayofweek"] = dt.dt.dayofweek
    out["travel_day"] = dt.dt.day
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    for col in ["from", "to", "flightType", "agency"]:
        out[col] = out[col].astype("string").fillna("Unknown").astype(str)
    return out

def json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    return value

def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
