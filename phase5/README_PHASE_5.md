# Phase 5 — Flask REST API

## Expected project layout
Run from the existing project root, which must contain:
- `artifacts/flight_price_model.joblib`
- `src/travel_ml/common.py` (the API imports this exact path)
- `data/` (not needed to serve predictions)

## Install and start (Windows PowerShell)
```powershell
cd "C:\Users\kmkb6\Downloads\travel-tourism-mlops-capstone\travel-tourism-mlops"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\phase5\requirements-phase5.txt
$env:PROJECT_ROOT = (Get-Location).Path
python .\phase5\api\app.py
```
If script execution is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window and activate again.

## Verify in a second PowerShell window
```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
Invoke-RestMethod http://127.0.0.1:5000/model-info
$body = Get-Content .\phase5\sample_request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType "application/json" -Body $body
```
The request contains the seven raw fields used by `flight_features`: `from`, `to`, `flightType`, `time`, `distance`, `agency`, `date`. Add more records by sending `{"records":[{...},{...}]}`.

## Important implementation note
The API uses the existing model and calls `flight_features` from `src/travel_ml/common.py`; it does not retrain. If your shared feature function requires the full training schema, the API adds internal placeholder `travelCode`, `userCode`, and `price` columns, then removes them before prediction. If `/health` returns 503 or prediction returns a feature-column error, send the complete error and the contents of `src/travel_ml/common.py`; do not edit the original model or common module blindly.

Keep the Flask window open while using Streamlit in Phase 6. Stop with Ctrl+C when finished.
