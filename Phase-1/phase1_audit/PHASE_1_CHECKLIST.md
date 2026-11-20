# Phase 1 Checklist

Run these checks from the project root in PowerShell. Do not delete or overwrite anything.

- [ ] Keep a backup copy of the current project folder.
- [ ] Confirm `data\flights.csv`, `data\hotels.csv`, and `data\users.csv` exist.
- [ ] Confirm `artifacts\flight_price_model.joblib` and `artifacts\gender_classifier.joblib` exist.
- [ ] Confirm the three JSON metric/metadata files exist in `artifacts\`.
- [ ] Run the read-only audit script.
- [ ] Review `reports\phase1_audit.json` and `reports\phase1_audit.txt`.
- [ ] Confirm Python version and dependency installation separately before running the API.
- [ ] Do not retrain models in Phase 1.
- [ ] Do not start Jenkins, Airflow, and Kubernetes together on the current laptop.

**Completion condition:** The script completes and its report has been reviewed. Runtime checks (API, Docker, Jenkins, Airflow, Kubernetes) are separate later steps.
