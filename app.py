from flask import Flask, jsonify, render_template, request, url_for
import kafka_server_apis as kafka
import utils as utils
import pickle
import os
import traceback
from config import MODEL_PATH, MODEL_PATH_2
import hashlib
import pandas as pd
import time
import signal
import json
import sys
import atexit
from experiments.experiment_manager import ExperimentManager
from experiments.api import experiment_api
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import mlflow
import uuid
import logging
from flask import Response 

IS_TESTING = os.getenv("TESTING", "false").lower() == "true"
IS_ONLINE_EVALUATION = os.getenv("ONLINE_EVALUATION", "false").lower() == "true"


# Initialize Flask app
app = Flask(__name__, template_folder='experiments/templates')
app.register_blueprint(experiment_api)

SERVER_IP = os.getenv("TEAM_15_SERVER_IP")
JENKINS_BUILD_NUMBER = os.getenv("BUILD_NUMBER", "unknown")  
MOVIES_CSV = os.path.join("data", "movies.csv")

kafka_server = kafka.KafkaServerApi()
experiment_manager = ExperimentManager()

# Prometheus metrics
REQUEST_COUNT = Counter('total_requests_total', 'Total number of requests to the application')
SUCCESSFUL_REQUESTS = Counter('successful_requests_total', 'Number of successful requests')
FAILED_REQUESTS = Counter('failed_requests_total', 'Number of failed requests')
UPTIME_SECONDS = Gauge('uptime_seconds', 'Application uptime in seconds')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request processing time', buckets=[0.1, 0.5, 1, 2, 5, 10])
HEALTH_CHECK_SUCCESS = Counter('health_check_success_total', 'Successful health check requests')
HEALTH_CHECK_FAILURE = Counter('health_check_failure_total', 'Failed health check requests')
MODEL_ACCURACY = Gauge('model_accuracy', 'Precision at K (e.g., Precision@10) of the recommendation model')

# Start time
start_time = time.time()

model_parameters = {
    "SVD_movie_recommender.pkl": {
        'model_version': 'SVDv1',
        'parameters': {'n_factors': 100, 'n_epochs': 20, 'biased': True, 'lr_all': 0.005, 'reg_all': 0.02},
    },
    "SVD_movie_recommender_2.pkl": {
        'model_version': 'SVDv2',
        'parameters': {'n_factors': 50, 'n_epochs': 10, 'biased': True, 'lr_all': 0.001, 'reg_all': 0.5},
    },
}
# Load all models and keep in a dictionary
models = {}
for path in [MODEL_PATH, MODEL_PATH_2]:
    with open(path, 'rb') as f:
        model_id = os.path.basename(path)
        models[model_id] = pickle.load(f)

# Get absolute path for data files
base_dir = os.path.dirname(os.path.abspath(__file__))
ratings_file = os.path.join(base_dir, 'data', 'extracted_ratings.csv')
train_df, val_df = utils.prepare_data_csv(ratings_file)
train_data, valid_data = utils.prepare_data_model(train_df, val_df)
all_movies_list = train_df['movie_id'].unique().tolist()
user_movie_list = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()

if not IS_TESTING and not IS_ONLINE_EVALUATION:
    mlflow.set_tracking_uri("http://mlflow:6001")
    mlflow.set_experiment("Movie Recommendation Predictions")

def log_prediction_provenance(model_id, response_time, status_code, recommendations, user_id):

    pipeline_version = JENKINS_BUILD_NUMBER  # Track Jenkins pipeline version
    model_info = model_parameters.get(model_id, {})
    run_id = uuid.uuid4()
    with mlflow.start_run(run_name=f"Prediction-{run_id}"):

        mlflow.set_tag("Model ID", model_id)
        mlflow.set_tag("Model Version", model_info.get('model_version', 'Unknown'))
        mlflow.set_tag("Pipeline Version", pipeline_version)

        mlflow.log_params(model_info.get('parameters', {}))
        mlflow.log_param("response_time_ms", response_time)
        mlflow.log_param("status_code", status_code)

        # mlflow.log_artifact(ratings_file, artifact_path="training_data")

        mlflow.log_param("training_data_path", ratings_file) 
        mlflow.log_dict(
            {"user_id": user_id, "recommendations": recommendations},
            artifact_file=f"recommendations_{user_id}_{run_id}.json"
        )

def log_retraining_provenance(model_id, training_duration, training_data_path):
    
    pipeline_version = JENKINS_BUILD_NUMBER  #
    model_info = model_parameters.get(model_id, {})
    parameters = model_info.get('parameters', {}) 
    run_id = uuid.uuid4()

    with mlflow.start_run(run_name=f"Retraining-{run_id}"):
        # Log model metadata
        mlflow.set_tag("Model ID", model_id)
        mlflow.set_tag("Model Version", model_info.get('model_version', 'Unknown'))
        mlflow.set_tag("Pipeline Version", pipeline_version)

        # Log training details
        mlflow.log_param("training_data_path", training_data_path)
        mlflow.log_param("model_parameters", json.dumps(parameters))  
        mlflow.log_metric("training_duration_seconds", training_duration)

        # Save full training data as an artifact
        mlflow.log_dict(
            {
                "model_id": model_id,
                "model_version": model_info.get('model_version', 'Unknown'),
                "parameters": parameters
            },
            artifact_file=f"model_parameters_{model_id}_{run_id}.json"
        )


# Function to calculate Precision@K
def calculate_precision_at_k(user_recommendations, user_relevant_movies, k=10):
    total_precision = 0
    total_users = 0

    for user_id, rec_movies in user_recommendations.items():
        relevant_movies = user_relevant_movies.get(user_id, set())
        if relevant_movies:
            top_k_recommendations = set(rec_movies[:k])  # Get top K recommendations
            relevant_count = len(top_k_recommendations.intersection(relevant_movies))
            precision = relevant_count / k  
            total_precision += precision
            total_users += 1

    return total_precision / total_users if total_users > 0 else 0  # Average Precision@K over users

def get_user_relevant_movies(df, rating_threshold=4):
    return df[df['rating'] >= rating_threshold].groupby('user_id')['movie_id'].apply(set).to_dict()

def evaluate_snapshot():
    try:
        snapshot_df = pd.read_csv(os.path.join(base_dir, "data", "extracted_ratings.csv"))
        user_relevant_movies = get_user_relevant_movies(snapshot_df)
        user_recommendations = {user_id: recommend_movies(user_id) for user_id in user_relevant_movies.keys()}
        
        # Calculate Precision@K
        precision = calculate_precision_at_k(user_recommendations, user_relevant_movies, 10)
        with open("online_evaluation_output.txt", 'w') as f:
            f.write(f"Precision@10: {precision:.4f}\n")
    
    except Exception as e:
        with open("online_evaluation_output.txt", 'w') as f:
            f.write(f"An error occurred while calculating Precision@K: {e}")
def cleanup_experiments():
    """Clean up all active experiments."""
    print("Cleaning up experiments before shutdown...")
    active_experiments = experiment_manager.get_active_experiments()
    experiment_names = list(active_experiments.keys())

    for exp_name in experiment_names:
        try:
            experiment_manager.delete_experiment(exp_name)
            print(f"Successfully deleted experiment: {exp_name}")
        except Exception as e:
            print(f"Error deleting experiment {exp_name}: {str(e)}")
    print("Experiment cleanup complete")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"Received signal {signum}")
    cleanup_experiments()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_experiments)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def select_model_for_experiment(user_id, experiment):
    split_point = int(experiment.traffic_split * 100)
    user_bucket = hash(str(user_id)) % 100
    if user_bucket < split_point:
        return experiment.model_a_id, models[experiment.model_a_id]
    return experiment.model_b_id, models[experiment.model_b_id]

def select_model(user_id):
    for experiment in experiment_manager.active_experiments.values():
        model_id, selected_model = select_model_for_experiment(user_id, experiment)
        if model_id:
            return model_id, selected_model
    default_model_id = MODEL_PATH.split('/')[-1]
    return default_model_id, models[default_model_id]

prediction_counter = 0
def recommend_movies(user_id):
    global prediction_counter 
    """Recommend movies for a user"""
    REQUEST_COUNT.inc()  # Increment request count for every recommendation request
    try:
        start_time_inner = time.time()
        model_id, selected_model = select_model(user_id)

        recommendations = utils.predict(
                selected_model, user_id, all_movies_list, user_movie_list, K=20
            )

            # Log metrics
        latency = (time.time() - start_time_inner) * 1000 
            # mlflow.log_metric("latency_seconds", latency)
        
        log_prediction_provenance(
            model_id=model_id,
            response_time=latency,
            status_code=200,  
            recommendations=recommendations,
            user_id=user_id
        )

        user_ratings = utils.get_user_ratings(user_id)
        rmse=None

        if user_ratings:
            movie_ids = [movie_id for movie_id, _ in user_ratings]
            predicted_values = utils.get_predicted_ratings(selected_model, user_id, movie_ids)
            predicted_ratings = list(zip(movie_ids, predicted_values))
            rmse = utils.calculate_rmse(predicted_ratings, user_ratings)
                # mlflow.log_metric("rmse", rmse)

                # Record experiment performance
            for experiment in experiment_manager.active_experiments.values():
                if model_id in [experiment.model_a_id, experiment.model_b_id]:
                    experiment_manager.record_performance(
                        experiment.name,
                        model_id,
                        1 - rmse,  # Already normalized
                        latency
                    )

            # Update Prometheus metrics
        SUCCESSFUL_REQUESTS.inc()
        REQUEST_LATENCY.observe(latency)
        uptime = int(time.time() - start_time)
        UPTIME_SECONDS.set(uptime)


            # Calculate Precision@10
        precision_at_10 = 0.0  # Default value in case of errors
        try:
            with open("evaluation/online_evaluation_output.txt", "r") as f:
                for line in f:
                    if "Precision@10:" in line:
                            # Extract the precision value from the line
                        precision_at_10 = float(line.split("Precision@10:")[1].strip())
                        break
        except Exception as file_error:
            print(f"Error reading precision from file: {file_error}")

            # Update Prometheus metric with the precision value
        MODEL_ACCURACY.set(precision_at_10)

            # Log health check success
        HEALTH_CHECK_SUCCESS.inc()

            # Return recommendations
        # return jsonify(recommendations)
        return recommendations

    except Exception as e:
        # Handle and log errors
        FAILED_REQUESTS.inc()
        HEALTH_CHECK_FAILURE.inc()
        print(f"Error in recommend_movies: {e}")
        traceback.print_exc()

        response_time = (time.time() - start_time_inner) * 1000 

        log_prediction_provenance(
            model_id="unknown",
            response_time=response_time,
            status_code=500,  # Error status
            recommendations=[],
            user_id=user_id
        )


        REQUEST_LATENCY.observe(time.time() - start_time_inner)
        return jsonify({'error': str(e)}), 500

    # finally:
    #     # Ensure no lingering active MLflow runs
    #     mlflow.end_run()

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    try:
        recommendations = recommend_movies(user_id)
        return recommendations
    except Exception as e:
        print(f"An error occurred while generating recommendations: {e}")
        traceback.print_exc()
        return []

@app.route('/experiments/dashboard')
def experiments_dashboard():
    return render_template('experiments.html')

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


if __name__ == '__main__':
    try:
        # if not IS_TESTING:
        #     initialize_mlflow()
        app.run(host='0.0.0.0', port=8082)
    finally:
        # if not IS_TESTING:
        cleanup_experiments()
