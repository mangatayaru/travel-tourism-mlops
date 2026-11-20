from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

# Update PROJECT_DIR to the absolute path visible to the Airflow worker.
PROJECT_DIR = "/opt/travel-tourism-mlops"
with DAG(
    dag_id="travel_tourism_mlops_validation",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["travel", "mlops"],
) as dag:
    validate_inputs = BashOperator(
        task_id="validate_inputs",
        bash_command=f"cd {PROJECT_DIR} && python phase2/scripts/validate_data.py",
    )
    evaluate_model = BashOperator(
        task_id="evaluate_existing_model",
        bash_command=f"cd {PROJECT_DIR} && python phase3/scripts/evaluate_existing_model.py --project-root .",
    )
    validate_inputs >> evaluate_model
