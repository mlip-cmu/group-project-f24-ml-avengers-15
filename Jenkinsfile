pipeline {
    agent any

    environment {
        SERVER_IP = credentials('SERVER_IP') 
        SSH_USER = credentials('SSH_USER')           
        SSH_PASSWORD = credentials('SSH_PASSWORD')   
        KAFKA_PORT = credentials('KAFKA_PORT')
        LOCAL_PORT = credentials('LOCAL_PORT')
        PYTHONPATH = "${WORKSPACE}" 
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
    }

    post {

        success {
            junit 'report.xml' // Publish test results
            
        }
    }
}
