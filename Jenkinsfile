pipeline {
  agent any
  environment {
    IMAGE_NAME = 'travel-price-api'
    IMAGE_TAG = "${BUILD_NUMBER}"
  }
  stages {
    stage('Checkout') {
      steps { checkout scm }
    }
    stage('Install and Test') {
      steps {
        bat 'py -3.11 -m pip install -r requirements.txt'
        bat 'py -3.11 -m pytest -q'
      }
    }
    stage('Train Regression Model') {
      steps {
        bat 'py -3.11 -m src.travel_ml.train_regression --data data/flights.csv --output artifacts --sample-size 50000'
      }
    }
    stage('Build Container') {
      steps {
        bat 'docker build -t %IMAGE_NAME%:%IMAGE_TAG% .'
      }
    }
    stage('Deploy to Kubernetes (opt-in)') {
      when { environment name: 'DEPLOY_TO_K8S', value: 'true' }
      steps {
        bat 'kubectl apply -f k8s/deployment.yaml'
        bat 'kubectl apply -f k8s/service.yaml'
        bat 'kubectl rollout status deployment/flight-price-api --timeout=120s'
      }
    }
  }
  post {
    always { echo 'Pipeline finished. Review test output and training metrics.' }
  }
}
