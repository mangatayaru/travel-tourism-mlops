# Capstone Submission Checklist

## A. Modeling notebooks
- [ ] `notebooks/01_flight_price_regression.ipynb`
- [ ] `notebooks/02_gender_classification.ipynb`
- [ ] `notebooks/03_hotel_recommendation.ipynb`
- [ ] Each notebook has business context, project summary, GitHub repository URL placeholder, EDA, preprocessing rationale, modeling, metrics, interpretation, and limitations.
- [ ] Re-run notebooks from top to bottom in the target environment.

## B. Single GitHub repository
- [ ] Push the complete repository, not three separate repositories.
- [ ] Add meaningful incremental commits.
- [ ] Add the final public/private repository URL to notebooks and final report.
- [ ] Do not commit `.venv`, credentials, secrets, or unnecessary large `mlruns` files.
- [ ] Confirm the repository includes source code, Dockerfiles, Kubernetes YAML, DAG, Jenkinsfile, requirements, tests, and README.

## C. Workflow documentation with screenshots
- [ ] Flask REST API: health, successful prediction, validation error.
- [ ] Streamlit: prediction screen and hotel recommendations.
- [ ] Docker: build and running container.
- [ ] Kubernetes: ready pods, service, health request, optional HPA.
- [ ] Airflow: DAG graph, successful tasks, task logs.
- [ ] Jenkins: tests/build stages and opt-in deployment stage.
- [ ] MLflow: experiment, parameters, metrics, model artifacts.
- [ ] Add captions and explain what each screenshot proves.
- [ ] Do not include installation instructions in the Google Doc, per assignment.

## D. Video presentation
The brief states both a 40-minute minimum and a separate 15-minute minimum. Use **40–45 minutes** to satisfy the stricter requirement. Explain the regression model only in the video, while mentioning the other models briefly if necessary.
- [ ] Dataset/business context and regression objective.
- [ ] EDA, feature selection, leakage, preprocessing, training, validation, metrics.
- [ ] MLflow experiment tracking.
- [ ] Flask request/response demonstration.
- [ ] Docker image and container.
- [ ] Kubernetes Deployment, Service, probes, scaling.
- [ ] Airflow DAG and scheduling.
- [ ] Jenkins CI/CD.
- [ ] Monitoring limitations, rollback, lessons learned.
- [ ] Confirm the final recording duration before submission.
