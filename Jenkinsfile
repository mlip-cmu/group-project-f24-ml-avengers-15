pipeline {
    agent any

    environment {
        SERVER_IP = credentials('SERVER_IP') 
        SSH_USER = credentials('SSH_USER')           
        SSH_PASSWORD = credentials('SSH_PASSWORD')   
        KAFKA_PORT = credentials('KAFKA_PORT')
        LOCAL_PORT = credentials('LOCAL_PORT')
        PYTHONPATH = "${WORKSPACE}"
        DOCKER_IMAGE = "recommender-service:latest"
    }

    stages {
        stage('Build') {
            steps {
                sh '''
                # Create a virtual environment and activate it
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                deactivate
                '''
            }
        }

        stage('Run Unit Tests and generate test report') {
            steps {
                sh '''
                . venv/bin/activate
                pytest test/ --junitxml=report.xml
                deactivate
                '''
            }
        }

        stage('Run Offline Evaluation') {
            steps {
                echo 'Running Offline Evaluation'
                sh '''
                . venv/bin/activate
                cd evaluation
                python offline.py 
                deactivate
                '''
            }
        }

        stage('Run Online Evaluation') {
            steps {
                echo 'Running Online Evaluation'
                sh '''
                . venv/bin/activate
                python evaluation/online_evaluation.py
                deactivate
                '''
            }
        }

        stage('Run Data Qauality') {
            steps {
                echo 'Running Data Qauality'
                sh '''
                . venv/bin/activate
                python evaluation/data_qualitycheck.py
                deactivate
                '''
            }
        }

        stage('Run Data Drift') {
            steps {
                echo 'Running Data Drift'
                sh '''
                . venv/bin/activate
                python evaluation/data_drift.py
                deactivate
                '''
            }
        }

        stage('Retrain Model') {
            steps {
                echo 'Retraining Model'
                sh '''
                . venv/bin/activate
                python retrain.py --input_data data/input.csv
                deactivate
                '''
            }
        }

        // New Docker-related stages
        stage('Build Docker Image') {
            steps {
                echo 'Building Docker Image'
                sh '''
                docker build -t ${DOCKER_IMAGE} .
                '''
            }
        }

        stage('Run Docker Container Locally') {
            steps {
                echo 'Running Docker Container Locally'
                sh '''
                docker stop recommender-service || true
                docker rm recommender-service || true
                docker run -d --name recommender-service -p 8082:8082 ${DOCKER_IMAGE}
                '''
            }
        }
    }

    post {
        success {
            junit 'report.xml' // Publish test results
        }
    }
}
