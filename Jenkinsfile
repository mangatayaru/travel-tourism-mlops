pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 45, unit: 'MINUTES')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Python Environment') {
            steps {
                sh '''
                    python3 --version
                    python3 -m venv .venv-ci
                    . .venv-ci/bin/activate
                    python -m pip install --upgrade pip
                    python -m pip install -r requirements.txt
                    python -m pip install -r phase5/requirements-phase5.txt
                    python -m pip install -r phase8/requirements-phase8.txt
                '''
            }
        }

        
        stage('Train Regression Model') {
            steps {
                sh '''
                    . .venv-ci/bin/activate
                    export MLFLOW_TRACKING_URI="sqlite:///mlflow-ci.db"
                    python -m src.travel_ml.train_regression \
                        --data data/flights.csv \
                        --output artifacts \
                        --sample-size 50000
                '''
            }
        }


        
        stage('API Contract Tests') {
            steps {
                sh '''
                    . .venv-ci/bin/activate
                    mkdir -p reports
                    python -m pytest -v phase8/tests \
                        --junitxml=reports/junit.xml
                '''
            }
        }

        stage('Build API Docker Image') {
            steps {
                sh '''
                    docker build \
                        -t travel-price-api:${BUILD_NUMBER} \
                        -t travel-price-api:local .
                '''
            }
        }

        stage('Build Streamlit Docker Image') {
            steps {
                sh '''
                    docker build \
                        -f Dockerfile.streamlit \
                        -t travel-tourism-mlops-streamlit:${BUILD_NUMBER} \
                        -t travel-tourism-mlops-streamlit:latest .
                '''
            }
        }


    }

    
    post {
        always {
            junit allowEmptyResults: true,
                testResults: 'reports/junit.xml'

            archiveArtifacts artifacts: 'artifacts/**/*',
                            allowEmptyArchive: true

            echo 'Jenkins CI pipeline finished.'
        }
        success {
            echo 'SUCCESS: Model training and API tests completed.'
        }
        failure {
            echo 'FAILED: Check the stage logs for the error.'
        }
        }
}
