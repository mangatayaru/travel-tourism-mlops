# Travel & Tourism MLOps Capstone

End-to-end travel analytics capstone using `flights.csv`, `hotels.csv`, and `users.csv`. The flight-price regression model is the main MLOps deployment example; gender classification and hotel recommendations are separate educational workflows.

## Dataset audit (provided CSVs)
- `flights.csv`: 271,888 rows, 10 columns; no missing values or exact duplicate rows detected.
- `hotels.csv`: 40,552 rows, 8 columns; no missing values or exact duplicate rows detected.
- `users.csv`: 1,340 rows, 5 columns; no missing values or exact duplicate rows detected.

These are initial file-level checks, not a guarantee of valid ranges or correct business meaning. Run validation before training.

## Architecture
CSV data → validation/features → model training/evaluation → MLflow tracking/artifacts → Flask REST API + Streamlit UI → Docker → Kubernetes. Airflow schedules retraining; Jenkins tests, builds, and deploys.

## Repository map
```text
travel-tourism-mlops/
├── data/                         # supplied CSV datasets
├── notebooks/                    # regression, classification, recommender notebooks
├── src/travel_ml/                 # reusable validation, training, recommendation code
├── api/                           # Flask inference API
├── streamlit_app/                 # interactive recommendation and prediction UI
├── airflow/dags/                  # scheduled retraining DAG
├── k8s/                           # Deployment, Service, optional HPA
├── tests/                         # smoke/unit tests
├── docs/                          # MLOps workflow and presentation guide
├── artifacts/                     # generated model artifacts (created by training)
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
├── requirements.txt
└── .env.example
```

## Windows quick start (PowerShell)
1. Install Python 3.11, Git, Docker Desktop, and (for Kubernetes) enable Kubernetes in Docker Desktop.
2. Open this folder in VS Code and create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Train all three workflows:
   ```powershell
   python -m src.travel_ml.train_regression --data data/flights.csv --output artifacts
   python -m src.travel_ml.train_classification --data data/users.csv --output artifacts
   python -m src.travel_ml.train_recommender --hotels data/hotels.csv --users data/users.csv --output artifacts
   ```
4. Start the API:
   ```powershell
   python -m api.app
   ```
5. In a second terminal, check the API:
   ```powershell
   Invoke-RestMethod http://127.0.0.1:5000/health
   Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType 'application/json' -Body '{"from":"New York","to":"London","flightType":"firstClass","time":7.5,"distance":5585,"agency":"CloudFy","date":"2026-11-15"}'
   ```
   The sample values demonstrate request shape only; use valid values from your data for meaningful predictions.
6. Start Streamlit in another terminal:
   ```powershell
   streamlit run streamlit_app/app.py
   ```

## MLflow
Training scripts log parameters, metrics, and model artifacts to the local `mlruns/` folder. View runs with:
```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5001
```
Open http://127.0.0.1:5001. MLflow tracking is local by default; use a persistent tracking server/database and artifact store for a shared production deployment.

## Docker
Train the regression model first so `artifacts/flight_price_model.joblib` exists, then:
```powershell
docker build -t travel-price-api:local .
docker run --rm -p 5000:5000 -v "${PWD}/artifacts:/app/artifacts" travel-price-api:local
```
Or use `docker compose up --build api streamlit`. Docker Compose is a local demonstration, not a substitute for production secrets, storage, or observability.

## Kubernetes (Docker Desktop)
Build the image in the Docker Desktop image store:
```powershell
docker build -t travel-price-api:local .
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl port-forward service/flight-price-api 5000:80
```
In a separate terminal, call `http://127.0.0.1:5000/health`. The deployment mounts a ConfigMap-free image artifact by default only if the artifact is copied into the image at build time; see `docs/MLops_workflows.md` for the recommended local build and production alternatives. For a simple local run, the Dockerfile copies `artifacts/` if it exists when the image is built. Retrain, then rebuild the image when replacing the model. HPA requires metrics-server and resource metrics to be available.

## Airflow
The DAG is in `airflow/dags/travel_model_training.py`. It runs validation and regression training using the configured project path. Airflow is most straightforward in Docker/Linux; the provided Compose profile is a local learning setup. Do not run an Airflow scheduler on a Windows host directly. See the workflow guide.

## Jenkins
The `Jenkinsfile` runs tests, builds the Docker image, and includes an opt-in deployment stage. Configure Docker access and Kubernetes credentials on the Jenkins agent. Deployment is gated by `DEPLOY_TO_K8S=true` so a local build does not accidentally deploy.

## Evaluation and responsible use
- Regression: MAE, RMSE, R²; compare with a naive baseline.
- Classification: accuracy, precision, recall, F1, confusion matrix. Gender prediction is a teaching task only and should not be used for consequential decisions.
- Recommendation: ranked hotel suggestions using historical popularity and optional place/price filters; evaluate with a holdout split and ranking metrics if interaction labels support it.
- Avoid using personal identifiers (`userCode`, `travelCode`, user names) as predictive features.
- Predictions are estimates, not guaranteed quoted fares. Production systems need monitoring for drift, latency, data quality, access control, and model rollback.

## Reproducibility
The default training script uses a fixed random seed and logs data size, feature list, metrics, and model artifact. Exact metrics may vary by dependency version and machine. The `artifacts/` and `mlruns/` folders are generated locally and should not be committed unless explicitly required; large datasets should normally be stored outside Git in production.

## Submission checklist
See `docs/SUBMISSION_CHECKLIST.md` for notebook, repo, screenshots, and video requirements.
