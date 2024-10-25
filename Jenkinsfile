pipeline {
    agent any

    environment {
        SERVER_IP = credentials('SERVER_IP') 
        SSH_USER = credentials('SSH_USER')           
        SSH_PASSWORD = credentials('SSH_PASSWORD')   
        KAFKA_PORT = credentials('KAFKA_PORT')
        LOCAL_PORT = credentials('LOCAL_PORT')
    }

    stages {
        stage('Build') {
            steps {
                sh '''

                # Create a virtual environment and activate it
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt

                '''
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh '''
                . venv/bin/activate
                pytest
                '''
            }
        }

        stage('Deploy Flask App') {
            steps {
                echo 'Deploying the Flask application'
                sh '''

                . venv/bin/activate
                python app.py &
                '''
            }
        }

        stage('Setup Kafka SSH Tunnel') {
            steps {
                sh '''
                echo "Setting up SSH Tunnel to Kafka"
                ssh -L ${LOCAL_PORT}:localhost:${KAFKA_PORT} ${SSH_USER}@${SERVER_IP} -NT &
                '''
            }
        }

        stage('Check Model and API') {
            steps {
                script {
                    def response = sh(
                        script: '''
                        curl -X GET http://localhost:8082/recommend/1
                        ''',
                        returnStdout: true
                    ).trim()
                    echo "Response from Flask App: ${response}"
                }
            }
        }
    }

    post {
        always {
            echo 'Cleaning up background processes'
            sh 'kill $(jobs -p)'  
        }
    }
}
