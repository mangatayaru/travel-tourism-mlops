# Phase 8 — Automated API Tests with Pytest

Adds automated tests for the existing Flask API. It does not retrain or modify the model.

## Prerequisites
Run commands from the project root (the folder containing `phase5`, `phase6`, `src`, and `artifacts`).

## 1. Install pytest
```powershell
python -m pip install -r .\phase8\requirements-phase8.txt
```

## 2. Run tests
```powershell
python -m pytest -v .\phase8\tests
```

## 3. What is tested
- `/health`: API is healthy and model/feature function are loaded.
- `/model-info`: model metadata endpoint responds and reports a positive feature count.
- `/predict`: the sample request returns a finite, non-negative prediction.

Tests use Flask's in-process test client. You do not need to stop or restart the Flask server or Docker containers.

## 4. Save a report (optional)
```powershell
New-Item -ItemType Directory -Force .\reports\phase8 | Out-Null
python -m pytest -v .\phase8\tests 2>&1 | Tee-Object .\reports\phase8\pytest-results.txt
```

## Troubleshooting
- Run from the project root, not from inside `phase8`.
- Confirm `phase5\sample_request.json` and `phase5\api\app.py` exist.
- If pytest is missing, run step 1.
- The tests expect `phase5\api\app.py` to expose the Flask application as `app`. If the import fails, share the traceback. Do not change the model or `src\travel_ml\common.py` to fix an import issue.

## Completion criteria
All three tests pass. Keep the terminal output or saved report as capstone evidence.
