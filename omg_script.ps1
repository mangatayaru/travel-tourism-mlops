py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
python -m src.travel_ml.train_regression
python -m src.travel_ml.train_classification
python -m src.travel_ml.train_recommender

dir artifacts

docker build -t travel-price-api:local .
docker run --rm -p 5000:5000 -v "${PWD}/artifacts:/app/artifacts" travel-price-api:local

docker compose up --build api streamlit


docker build -t travel-price-api:local .
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl port-forward service/flight-price-api 5000:80

$body = @{
    from = "New York"
    to = "London"
    flightType = "firstClass"
    time = 7.5
    distance = 5585
    agency = "CloudFy"
    date = "2026-11-15"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

docker info
Get-Content docker-compose.yml
docker compose --profile airflow config --services

docker compose --profile airflow up -d
docker compose ps

docker compose logs -f
docker compose --profile airflow down

$body = @{
    from = "New York"
    to = "London"
    flightType = "firstClass"
    time = 7.5
    distance = 5585
    agency = "CloudFy"
    date = "2026-11-15"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
