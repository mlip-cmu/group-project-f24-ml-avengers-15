from flask import Flask, jsonify, render_template, url_for
import kafka_server_apis as kafka
import utils as utils
import pickle
import os
import traceback
from config import MODEL_PATH, MODEL_PATH_2
import hashlib
import time
import signal
import sys
import atexit
from experiments.experiment_manager import ExperimentManager
from experiments.api import experiment_api
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from evaluation.online_evaluation import evaluate_snapshot

# Initialize Flask app
app = Flask(__name__, template_folder='experiments/templates')
app.register_blueprint(experiment_api)

SERVER_IP = os.getenv("TEAM_15_SERVER_IP")
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

def recommend_movies(user_id):
    """Recommend movies for a user."""
    REQUEST_COUNT.inc()
    try:
        start_time = time.time()
        model_id, selected_model = select_model(user_id)
        recommendations = utils.predict(selected_model, user_id, all_movies_list, user_movie_list)
        SUCCESSFUL_REQUESTS.inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        uptime = int(time.time() - start_time)
        UPTIME_SECONDS.set(uptime)

        # Update model accuracy metric
        precision_at_10 = 0.0
        try:
            with open("evaluation/online_evaluation_output.txt", "r") as f:
                for line in f:
                    if "Precision@10:" in line:
                        precision_at_10 = float(line.split("Precision@10:")[1].strip())
                        break
        except Exception as file_error:
            print(f"Error reading precision from file: {file_error}")
        MODEL_ACCURACY.set(precision_at_10)

        return recommendations
    except Exception as e:
        FAILED_REQUESTS.inc()
        HEALTH_CHECK_FAILURE.inc()
        print(f"Error in recommend_movies: {e}")
        traceback.print_exc()
        return []

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    try:
        recommendations = recommend_movies(user_id)
        return jsonify(recommendations)
    except Exception as e:
        print(f"An error occurred while generating recommendations: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/experiments/dashboard')
def experiments_dashboard():
    return render_template('experiments.html')

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8082)
    finally:
        cleanup_experiments()
