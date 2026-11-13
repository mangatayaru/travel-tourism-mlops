import json
import math
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phase5.api.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    flask_app = create_app()
    flask_app.config.update(TESTING=True)

    with flask_app.test_client() as test_client:
        yield test_client


def load_sample_request():
    sample_path = PROJECT_ROOT / "phase5" / "sample_request.json"
    with sample_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["feature_function_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["model_type"]
    assert int(payload["n_features"]) > 0


def test_prediction_endpoint_returns_finite_price(client):
    response = client.post(
        "/predict",
        data=json.dumps(load_sample_request()),
        content_type="application/json",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] >= 1
    assert len(payload["predictions"]) >= 1
    predicted_price = float(payload["predictions"][0]["predicted_price"])
    assert math.isfinite(predicted_price)
    assert predicted_price >= 0
