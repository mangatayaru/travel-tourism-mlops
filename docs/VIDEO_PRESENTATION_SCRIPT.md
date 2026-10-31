# Regression MLOps Capstone — 40–45 Minute Video Runbook

Use this as a speaking guide, not a claim that stages have run. Record the actual screens after running the project. Total target: 42 minutes.

| Time | Section | What to explain and show |
|---|---|---|
| 00:00–03:00 | Introduction | Business problem, expected prediction, scope, architecture. Explain why price estimates help travel planning and why estimates are not guaranteed quotes. |
| 03:00–07:00 | Dataset | Open `flights.csv`; explain each column, 271,888 rows, no missing/duplicate rows in initial audit, currency/units not specified, and why IDs are excluded. |
| 07:00–12:00 | EDA and feature engineering | Show target distribution and categorical/numeric features. Explain route, flight type, agency, time, distance, date-derived month/day features. Discuss outliers, data leakage, and date-aware validation for future prediction. |
| 12:00–17:00 | Model training and evaluation | Run training script. Explain train/test split, one-hot encoding, imputers, Random Forest, reproducible seed, MAE/RMSE/R², baseline comparison opportunity, and why metrics must be interpreted in target units. Read actual generated metrics rather than inventing them. |
| 17:00–20:00 | MLflow | Open MLflow UI; explain experiment, run, parameters, metrics, artifact, model versioning. Explain the difference between local file tracking and a shared production tracking server. |
| 20:00–24:00 | Flask REST API | Show API code, `/health`, required JSON fields, `/predict`, shared feature transformation, error handling. Send one valid request and one invalid request. Explain serialization and why the exact training pipeline is loaded. |
| 24:00–27:00 | Streamlit | Demonstrate the app's flight tab and explain that it calls the API rather than duplicating model logic. Briefly show hotel recommendation UI, but keep the main video centered on regression. |
| 27:00–31:00 | Docker | Explain Dockerfile layers, dependencies, app source, model artifact, port mapping, image build, container isolation, and why training must happen before building the image. Show container logs and health endpoint. |
| 31:00–35:00 | Kubernetes | Explain Deployment, replicas, Service, readiness/liveness probes, resource requests/limits, image tags, and port forwarding. Show pods and service. Explain that HPA requires metrics-server and that production clusters need a registry. |
| 35:00–38:00 | Airflow | Show DAG graph, validation task, training task, retry policy, weekly schedule, task logs, and MLflow run created by a successful task. Distinguish orchestration from CI/CD. |
| 38:00–41:00 | Jenkins CI/CD | Walk through checkout, tests, model training, image build, and gated Kubernetes deploy. Explain immutable tags, credentials, test gates, and why deployment is opt-in in the sample. |
| 41:00–43:00 | Monitoring, limitations, conclusion | Show health endpoint and evaluation JSON. Explain missing production drift/latency monitoring, model rollback, API security, currency/units ambiguity, and next improvements. Summarize outcomes and close. |

## Suggested opening
“Hello, in this presentation I will demonstrate an end-to-end MLOps workflow for predicting flight prices. I will move from dataset understanding and model evaluation to MLflow tracking, REST API serving, containerization, Kubernetes deployment, scheduled training with Airflow, and CI/CD using Jenkins.”

## Suggested closing
“This project demonstrates how a trained regression pipeline can move from experimentation into a repeatable and deployable service. The current version is a learning-grade local implementation; production readiness would require stronger data validation, time-aware evaluation, security, durable artifact storage, monitoring, and controlled model promotion.”

## Evidence rule
Use actual results from your run. Never state a metric, deployment success, or screenshot result that you have not verified on your machine.
