from flask import Flask, jsonify
import utils as utils
import pickle
import os
import traceback
from config import MODEL_PATH
import time
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from evaluation.online_evaluation import evaluate_snapshot

app = Flask(__name__)

SERVER_IP = os.getenv("TEAM_15_SERVER_IP")
MOVIES_CSV = os.path.join("data", "movies.csv")

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

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

ratings_file = 'data/extracted_ratings.csv'
train_df, val_df = utils.prepare_data_csv(ratings_file)
train_data, valid_data = utils.prepare_data_model(train_df, val_df)
all_movies_list = train_df['movie_id'].unique().tolist()
user_movie_list = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()


def recommend_movies(user_id):
    REQUEST_COUNT.inc()
    
    try:
        recommendations = utils.predict(model, user_id, all_movies_list, user_movie_list)
        SUCCESSFUL_REQUESTS.inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        uptime = int(time.time() - start_time)
        UPTIME_SECONDS.set(uptime)

        # Read model accuracy from online_evaluation_output.txt
        precision_at_10 = 0.0  # Default value in case of errors
        try:
            with open("evaluation/online_evaluation_output.txt", "r") as f:
                for line in f:
                    if "Precision@10:" in line:
                        # Extract the precision value from the line
                        precision_at_10 = float(line.split("Precision@10:")[1].strip())
                        print(precision_at_10)
                        break
        except Exception as file_error:
            print(f"Error reading precision from file: {file_error}")

        # Update Prometheus metric with the precision value
        MODEL_ACCURACY.set(precision_at_10)

        HEALTH_CHECK_SUCCESS.inc()
        return ','.join(recommendations)
    except Exception as e:
        FAILED_REQUESTS.inc()
        HEALTH_CHECK_FAILURE.inc()
        print(f"Error during health check: {e}")
        REQUEST_LATENCY.observe(time.time() - start_time)
        print(f"An error occurred while fetching user {user_id} info: {e}")
        traceback.print_exc()
        return []

# @app.route('/health', methods=['GET'])
# def health_check():
#     try:
#         uptime = int(time.time() - start_time)
#         UPTIME_SECONDS.set(uptime)

#         # Read model accuracy from online_evaluation_output.txt
#         precision_at_10 = 0.0  # Default value in case of errors
#         try:
#             with open("evaluation/online_evaluation_output.txt", "r") as f:
#                 for line in f:
#                     if "Precision@10:" in line:
#                         # Extract the precision value from the line
#                         precision_at_10 = float(line.split("Precision@10:")[1].strip())
#                         print(precision_at_10)
#                         break
#         except Exception as file_error:
#             print(f"Error reading precision from file: {file_error}")

#         # Update Prometheus metric with the precision value
#         MODEL_ACCURACY.set(precision_at_10)

#         HEALTH_CHECK_SUCCESS.inc()
#         return jsonify({
#             "status": "UP",
#             "uptime_seconds": uptime,
#             "precision_at_10": precision_at_10
#         })
#     except Exception as e:
#         HEALTH_CHECK_FAILURE.inc()
#         print(f"Error during health check: {e}")
#         return jsonify({"status": "DOWN"}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    # time.sleep(10)  # Sleep for 10 seconds to simulate latency
    # Expose metrics in Prometheus format
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    
    try:
        recommendations = recommend_movies(user_id)
        return jsonify(recommendations)
    except Exception as e:
        print(f"An error occurred while generating recommendations: {e}")
        traceback.print_exc()
        return []

if __name__ == '__main__':
    start_time = time.time()
    app.run(host='0.0.0.0', port=8083)
