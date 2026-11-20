# Phase 1 — Project Audit & Baseline Review

**Project:** Travel & Tourism MLOps Capstone  
**Package type:** Audit-only; no training, deployment, or source-code changes  
**Audit basis:** Project archive and separately uploaded configuration, model, and metrics files supplied in this conversation.

## 1. Scope and safety

This phase establishes what is already present before making changes. It does not overwrite, retrain, delete, or modify any existing project file, model, dataset, MLflow database, Docker image, or Jenkins configuration.

The project archive contains 37 files. Its repository structure includes:
- `data/` — flights, hotels, and users CSV datasets
- `notebooks/` — three educational workflow notebooks
- `src/travel_ml/` — shared utilities, training scripts, and recommender
- `api/` — Flask API
- `streamlit_app/` — Streamlit interface
- `tests/` — smoke test
- `airflow/dags/` — model-training DAG
- `k8s/` — Deployment, Service, and HPA manifests
- `docs/` — workflow, report, video, checklist, and dataset audit documents
- Root-level Docker, Compose, Jenkins, requirements, and environment-example files

Separately uploaded files include `.env.example`, `.gitignore`, `Dockerfile`, `Dockerfile.streamlit`, `Jenkinsfile`, `docker-compose.yml`, `mlflow.db`, `requirements.txt`, `needed.ps1`, `omg_script.ps1`, `NEEDED.PY`, `README(3).md`, `app.py`, the three CSV datasets, two model binaries, and three JSON metric/metadata files.

## 2. Dataset baseline

The project README's existing audit records:

| Dataset | Rows | Columns |
|---|---:|---:|
| `flights.csv` | 271,888 | 10 |
| `hotels.csv` | 40,552 | 8 |
| `users.csv` | 1,340 | 5 |

The README reports no missing values or exact duplicate rows in its initial file-level checks. These are previously recorded results, not a fresh re-validation performed by this package. The included audit script can re-check the CSV files locally.

## 3. Existing model and workflow baseline

### Flight-price regression
- Model artifact supplied: `flight_price_model.joblib`
- Model type in metrics: `RandomForestRegressor`
- Target: `price`
- Recorded train/test rows: 217,510 / 54,378
- Recorded MAE: 0.05674
- Recorded RMSE: 0.73060
- Recorded R²: 0.999996

**Audit flag — unusually high R²:** This result is exceptionally close to 1.0. It is not proof of data leakage by itself. Before relying on it, inspect the target distribution, feature/target construction, duplicate or near-duplicate rows across the random split, and performance on a chronological or otherwise independent holdout. The training script currently uses a random 80/20 split and includes route, flight type, time, distance, agency, and date-derived features.

### Classification
- Model artifact supplied: `gender_classifier.joblib`
- Recorded accuracy: 0.3806
- Recorded macro F1: 0.3459
- Test support: 268 records
- Female recall: 0.10; male recall: 0.556; `none` recall: 0.489

**Audit flag — weak baseline:** The supplied classification results indicate limited predictive performance, particularly for the female class. This workflow should be treated as an educational experiment only. Gender is a sensitive personal attribute and must not be used for consequential decisions or operational profiling. Consider whether this workflow is necessary for the capstone deliverable.

### Hotel recommendation
- Supplied metadata records 40,552 hotel rows and 1,340 user rows.
- 40,552 hotel-user matches are recorded.
- 9 unique hotel-place pairs are recorded.
- The documented method is popularity ranking with optional place and daily-price filters.

**Audit flag:** The metadata does not establish recommendation quality. A later evaluation should define a suitable holdout and ranking metrics if reliable interaction labels are available.

## 4. Application and MLOps components found

The project archive includes a Flask API with `/health` and `/predict`, a Streamlit app, Dockerfiles, a Compose file, a Jenkins pipeline, an Airflow DAG, Kubernetes manifests, and a smoke test.

The supplied Compose configuration maps API port 5000 and Streamlit port 8501. Its Streamlit configuration uses `http://api:5000`, which is appropriate for service-to-service communication within the Compose network; when running Streamlit directly on the host, a different API URL may be needed.

The Jenkinsfile uses Windows `bat` commands, installs dependencies, runs pytest, trains a sampled regression model, builds a Docker image, and has an opt-in Kubernetes deployment stage. This assumes the Jenkins agent is a Windows agent with Python 3.11, Docker CLI/daemon access, and (for deployment) kubectl credentials/configuration. These prerequisites need to be checked against the actual Jenkins agent before running the pipeline.

The project documentation notes that Kubernetes HPA requires metrics-server and resource metrics. Airflow is documented as a Docker/Linux learning setup.

## 5. Risks and items to verify in later phases

1. Verify the very high regression R² with an independent evaluation; do not claim leakage until evidence confirms it.
2. Check whether the price target is derived from, or deterministically encoded by, input columns.
3. Verify model artifacts match the current training code and feature schema.
4. Validate API request schema and prediction behavior using real values from the dataset.
5. Confirm the Docker build context includes the trained artifact or mounts it at runtime.
6. Confirm Jenkins agent OS and Docker access; the current pipeline uses Windows batch syntax.
7. Validate the Airflow DAG imports and runs before enabling scheduled retraining.
8. Validate Kubernetes cluster connectivity and metrics-server before applying HPA.
9. Keep secrets out of `.env.example`, source control, and logs.
10. Keep the gender classification workflow strictly educational and non-consequential.

## 6. Phase 1 outcome

**Status: Baseline inventory prepared; runtime verification remains to be performed on the user's machine.**

No code changes are included in this phase. The next phase should begin with a controlled local validation of the datasets, Python environment, model loading, and existing API health endpoint. Only after those checks pass should we change code or package a Phase 2 ZIP.

## 7. Files in this Phase 1 package

- `PHASE_1_AUDIT_REPORT.md` — this report
- `PHASE_1_CHECKLIST.md` — user-run checklist
- `scripts/phase1_audit.py` — read-only local inventory and CSV/JSON audit script
- `README_PHASE_1.md` — instructions to run the audit
