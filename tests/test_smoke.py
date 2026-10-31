from pathlib import Path
import pandas as pd
from src.travel_ml.common import flight_features

def test_flight_features_excludes_identifiers_and_target():
    sample = pd.DataFrame([{
        "travelCode": 1, "userCode": 7, "from": "A", "to": "B",
        "flightType": "economy", "price": 100.0, "time": 2.0,
        "distance": 500.0, "agency": "X", "date": "2026-01-01"
    }])
    features = flight_features(sample)
    assert "price" not in features.columns
    assert "userCode" not in features.columns
    assert "travelCode" not in features.columns
    assert "travel_month" in features.columns

def test_csv_files_present():
    root = Path(__file__).resolve().parents[1]
    assert (root / "data" / "flights.csv").exists()
    assert (root / "data" / "hotels.csv").exists()
    assert (root / "data" / "users.csv").exists()
