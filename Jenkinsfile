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

        stage('Run Unit Tests') {
            steps {
                sh '''
                . venv/bin/activate
                pytest
                deactivate
                '''
            }
        }

        stage('Deploy Flask App') {
            steps {
                echo 'Deploying the Flask application'
                sh '''
                . venv/bin/activate
                python app.py &
                echo $! > flask_pid.txt
                deactivate
                '''
            }
        }

        stage('Setup Kafka SSH Tunnel') {
            steps {
                sh '''
                echo "Setting up SSH Tunnel to Kafka"
                ssh -L ${LOCAL_PORT}:localhost:${KAFKA_PORT} ${SSH_USER}@${SERVER_IP} -NT &
                echo $! > ssh_tunnel_pid.txt
                '''
            }
        }

        stage('Check Model and API') {
            steps {
                script {
                    def response = sh(
                        script: '''
                        . venv/bin/activate
                        curl -X GET http://localhost:8082/recommend/1
                        deactivate
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
            echo 'Cleaning up specific background processes'
            sh '''
            if [ -f flask_pid.txt ]; then
                kill $(cat flask_pid.txt)
                rm flask_pid.txt
            fi
            if [ -f ssh_tunnel_pid.txt ]; then
                kill $(cat ssh_tunnel_pid.txt)
                rm ssh_tunnel_pid.txt
            fi
            '''
        }
    }
}
