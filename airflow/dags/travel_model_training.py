"""Airflow DAG: validate data and retrain the flight-price model."""
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from airflow import DAG
from airflow.operators.python import PythonOperator

# Override TRAVEL_PROJECT_ROOT in the Airflow environment for your mounted project directory.
PROJECT_ROOT = Path(__import__("os").getenv("TRAVEL_PROJECT_ROOT", "/opt/travel-project"))

def validate_inputs():
    import pandas as pd
    flight_path = PROJECT_ROOT / "data" / "flights.csv"
    df = pd.read_csv(flight_path)
    required = {"from","to","flightType","price","time","distance","agency","date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Flight dataset missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Flight dataset is empty")
    if pd.to_numeric(df["price"], errors="coerce").isna().all():
        raise ValueError("No numeric flight prices found")
    return {"rows": int(len(df)), "columns": int(len(df.columns))}

def train_model():
    command = ["python", "-m", "src.travel_ml.train_regression",
               "--data", str(PROJECT_ROOT / "data" / "flights.csv"),
               "--output", str(PROJECT_ROOT / "artifacts")]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

with DAG(
    dag_id="travel_flight_price_training",
    description="Validate flight data and train/log the regression model",
    start_date=datetime(2026, 1, 1),
    schedule="0 6 * * 1",
    catchup=False,
    default_args={"owner":"capstone", "retries":2, "retry_delay":timedelta(minutes=5)},
    tags=["travel","mlops","regression"],
) as dag:
    validate = PythonOperator(task_id="validate_flight_data", python_callable=validate_inputs)
    train = PythonOperator(task_id="train_flight_price_model", python_callable=train_model)
    validate >> train
