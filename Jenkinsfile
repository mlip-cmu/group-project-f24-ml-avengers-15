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
                pytest test/ 
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
