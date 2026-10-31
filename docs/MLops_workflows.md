# MLOps Workflow Guide (local Windows + VS Code)

This guide documents the workflow and gives screenshot checkpoints for the separate Google Doc requested in the assignment. Do not include tool installation steps in the final Google Doc; this guide focuses on how each stage works and how to demonstrate it.

## 1. Model training and MLflow
**Flow:** CSV validation → feature engineering → train/test split → pipeline fit → metrics → model artifact.
1. Run `python -m src.travel_ml.train_regression --data data/flights.csv --output artifacts`.
2. The script stores `artifacts/flight_price_model.joblib` and `artifacts/flight_price_metrics.json`.
3. MLflow logs parameters, MAE, RMSE, R², and a model artifact to `mlruns/`.
4. Start `mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001`.
5. Open the experiment and capture the run list, parameters/metrics, and artifact view.
**Screenshot checkpoints:** terminal training result; MLflow experiment table; one run's metrics and artifacts.

## 2. REST API (Flask)
**Flow:** JSON request → schema check → same feature engineering as training → load serialized pipeline → predict → JSON response.
1. Start `python -m api.app`.
2. Open `http://127.0.0.1:5000/health`; show status and model availability.
3. Send a POST to `/predict` with `from`, `to`, `flightType`, `time`, `distance`, `agency`, and `date`.
4. Demonstrate a missing-field request and explain the 400 validation response.
**Screenshot checkpoints:** health endpoint; successful prediction response; validation error response.

Example PowerShell:
```powershell
$body = @{
  from="New York"; to="London"; flightType="firstClass"
  time=7.5; distance=5585; agency="CloudFy"; date="2026-11-15"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType "application/json" -Body $body
```
Use values consistent with the dataset for credible demonstration. The currency is not defined in the dataset schema, so API responses intentionally do not claim USD/INR.

## 3. Streamlit application
**Flow:** user enters a flight request → Streamlit calls Flask → prediction is displayed; hotel tab filters historical hotels and plots booking counts.
1. Keep Flask running.
2. Run `streamlit run streamlit_app/app.py`.
3. Demonstrate flight price prediction, hotel filtering, and dataset overview.
**Screenshot checkpoints:** app tabs; price prediction; filtered hotel table and bar chart.

## 4. Docker deployment
**Flow:** Dockerfile installs runtime dependencies and copies app code/data/model into a portable image.
1. Train the model first.
2. Build: `docker build -t travel-price-api:local .`
3. Run: `docker run --rm -p 5000:5000 travel-price-api:local`
4. Check `/health` and `/predict`.
5. Alternatively run API and Streamlit with `docker compose up --build api streamlit`.
**Screenshot checkpoints:** successful image build; running container; health endpoint from host.
**Note:** Dockerfile copies `artifacts/` at build time. Train before building. For production, store model artifacts externally or use a model registry rather than rebuilding images for every model version.

## 5. Kubernetes deployment and scaling
**Flow:** Deployment maintains replica count; Service gives a stable in-cluster endpoint; readiness/liveness probes help restart or route around unhealthy pods.
1. Build the image in Docker Desktop's local image store with tag `travel-price-api:local`.
2. Apply `kubectl apply -f k8s/deployment.yaml` and `kubectl apply -f k8s/service.yaml`.
3. Check `kubectl get deployments,pods,services`.
4. Forward the service: `kubectl port-forward service/flight-price-api 5000:80`.
5. In a second terminal check `/health`.
6. Optional HPA: `kubectl apply -f k8s/hpa.yaml`. HPA needs metrics-server and working resource metrics; it will not scale without them.
**Screenshot checkpoints:** deployment and pods ready; service; port-forward and health response; HPA output if configured.
**Local image note:** Docker Desktop Kubernetes usually sees images built locally. On another cluster, push the image to a registry and update the manifest image.
**Artifact note:** the Docker image must contain `artifacts/flight_price_model.joblib` at build time. Rebuild after training a new model.

## 6. Airflow scheduling
**Flow:** scheduler triggers DAG → validation task checks required columns and usable target → training task executes the regression script → MLflow records the run.
1. Airflow is typically run in Docker/Linux or WSL2, not directly as a native Windows service.
2. Mount this repository into the Airflow container at `/opt/travel-project`, or set `TRAVEL_PROJECT_ROOT` to the mounted path.
3. Place `airflow/dags/travel_model_training.py` in Airflow's DAG folder.
4. Enable `travel_flight_price_training` in the UI and trigger it manually.
5. Inspect task logs, then confirm a new MLflow run.
**Screenshot checkpoints:** DAG graph; successful validation and training tasks; task log with training metrics.
**Scheduling:** configured for 06:00 every Monday in the scheduler's timezone. Change to your operational requirement. This DAG demonstrates orchestration; production should publish model artifacts atomically and gate deployment on validation metrics.

## 7. Jenkins CI/CD
**Flow:** checkout → install dependencies → tests → train model → Docker build → optional Kubernetes rollout.
1. Configure a Windows Jenkins agent with Python and Docker CLI access.
2. Ensure the repository is checked out into the workspace.
3. Create a Boolean/string parameter `DEPLOY_TO_K8S`; only set it to `true` after Kubernetes credentials/context are configured.
4. Run the pipeline and review test, training, and build output.
5. For real CI, pin dependencies, scan images, store credentials in Jenkins credentials, push tagged images to a registry, and deploy immutable tags. Do not store secrets in Git.
**Screenshot checkpoints:** successful stages; test results; image tag; Kubernetes rollout only if explicitly enabled.
**Note:** the Jenkinsfile uses Windows `bat` steps. For a Linux agent, replace `bat` with `sh`.

## 8. Monitoring and model lifecycle
This capstone logs offline evaluation metrics to MLflow and exposes an API health endpoint. It does not claim to include full production drift monitoring. A production extension should record request latency/error rate, input distributions, prediction distributions, data-quality failures, model version, and periodic ground-truth error. Alert on thresholds and retain a known-good model for rollback.

## 9. Limitations and production hardening
- The sample flight fields have no declared currency or units for time/distance; preserve the dataset's conventions.
- The default train/test split is random and appropriate as a baseline. If predicting future fares, use chronological validation and investigate leakage.
- The API does not perform authentication or rate limiting; do not expose it publicly without controls.
- The recommender ranks historical hotel/place pairs by booking count with optional filters; it is a transparent baseline, not a personalized deep-learning model.
- Gender inference is sensitive and should remain an educational classifier only.
- Local files and MLflow tracking are not durable multi-user infrastructure.
