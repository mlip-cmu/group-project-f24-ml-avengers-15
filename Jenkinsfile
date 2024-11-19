pipeline {
    agent any

    environment {
        SERVER_IP = credentials('SERVER_IP') 
        SSH_USER = credentials('SSH_USER')           
        SSH_PASSWORD = credentials('SSH_PASSWORD')   
        KAFKA_PORT = credentials('KAFKA_PORT')
        LOCAL_PORT = credentials('LOCAL_PORT')
        PYTHONPATH = "${WORKSPACE}"
        PROJECT_NAME = "group-project-f24-ml-avengers-15"
        TESTING = "1"

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
        

        stage('Consume Data from Kafka') {
            steps {
                echo 'Starting Kafka data consumer...'
                sh '''
                . venv/bin/activate
                python consume_kafka_logs.py
                echo "Done"
                deactivate
                '''
            }
        }

        // stage('Cleanup Kafka Tunnel') {
        //     steps {
        //         script {
        //             sh 'pkill -f "ssh -o ServerAliveInterval=60 -L 9092:localhost:9092" || true'
        //         }
        //     }
        // }

        stage('Run Unit Tests and generate test report') {
            steps {
                sh '''
                . venv/bin/activate
                TESTING=true pytest test/ --junitxml=report.xml
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
                ONLINE_EVALUATION=true python evaluation/online_evaluation.py
                deactivate
                '''
            }
        }

        stage('Run Data Quality') {
            steps {
                echo 'Running Data Quality'
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
                echo 'Retraining Model...'
                script {
                    def status = sh(script: '''
                        . venv/bin/activate
                        python retrain.py
                        deactivate
                    ''', returnStatus: true)

                    if (status == 0) {
                        echo 'Retraining completed successfully.'
                    } else if (status == 1) {
                        echo 'No data available for retraining. Skipping model deployment.'
                    } else if (status == 42) {
                        echo 'An error occurred during retraining. Initiating rollback...'
                        sh '''
                        . venv/bin/activate
                        python rollback_model.py
                        deactivate
                        '''
                        error 'Retraining failed due to an error.'
                    } else {
                        echo "Unknown exit code: ${status}. Check logs for details."
                        error 'Retraining failed with an unknown error.'
                    }
                }
            }
        }

        stage('Deploy Using Docker Compose') {
            steps {
                script {
                    echo 'Deploying Using Docker Compose'
                    sh '''
                    docker-compose -p ${PROJECT_NAME} down || true
                    docker-compose -p ${PROJECT_NAME} up -d --build
                    '''
                }
            }
        }
    }

    post {
        success {
            junit 'report.xml' // Publish test results
            echo 'Pipeline completed successfully!'
        }
        failure {
        echo 'Pipeline failed.'
        }
    }
}
