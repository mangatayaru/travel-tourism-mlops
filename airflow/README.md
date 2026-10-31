# Airflow DAG
The DAG assumes the project is mounted at `/opt/travel-project` and has Python dependencies available to the Airflow task environment. Set `TRAVEL_PROJECT_ROOT` if the path differs. For local development on Windows, use Docker Desktop/WSL2 for Airflow. Copy the DAG into the Airflow DAGs directory or mount `airflow/dags` there.
