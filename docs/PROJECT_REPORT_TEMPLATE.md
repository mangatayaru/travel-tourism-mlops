# Project Summary — Travel & Tourism MLOps Capstone

**GitHub repository:** [Insert the single repository URL after publishing]

## Business problem
Describe how an estimated flight price can support travel planning. Clarify that the model estimates prices from historical dataset patterns and is not a live fare quote.

## Data and scope
Describe `flights.csv`, `hotels.csv`, and `users.csv`, the initial quality audit, column meaning, and unknown currency/measurement units.

## Regression approach
Explain target, feature selection, excluded identifiers, date feature extraction, preprocessing pipeline, train/test strategy, model choice, and random seed.

## Evaluation
Insert actual MAE, RMSE, R², train/test counts, and compare to a simple baseline. Discuss whether errors are acceptable in the context of the dataset.

## Deployment and MLOps
Document Flask, Docker, Kubernetes, Airflow, Jenkins, and MLflow with screenshots and captions. Explain the data flow between components.

## Results and limitations
Report actual outputs. Discuss generalization, leakage risks, temporal validation, unknown units/currency, local tracking limitations, and missing production monitoring/security.

## Future improvements
Chronological backtesting, stronger baselines and tuning, model promotion gates, external artifact storage, request metrics, drift checks, auth/rate limits, and rollback.
