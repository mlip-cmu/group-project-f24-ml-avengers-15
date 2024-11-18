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

app = Flask(__name__, template_folder='experiments/templates')
app.register_blueprint(experiment_api)

SERVER_IP = os.getenv("TEAM_15_SERVER_IP")
MOVIES_CSV = os.path.join("data", "movies.csv")

kafka_server = kafka.KafkaServerApi()
experiment_manager = ExperimentManager()

# Load all models and keep in a list
models = {}  # Change to dictionary for easier lookup
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
    """Clean up all active experiments"""
    print("Cleaning up experiments before shutdown...")
    # Get a list of experiment names first to avoid dictionary modification during iteration
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
    """Handle shutdown signals"""
    print(f"Received signal {signum}")
    cleanup_experiments()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_experiments)  # For normal termination
signal.signal(signal.SIGTERM, signal_handler)  # For graceful termination
signal.signal(signal.SIGINT, signal_handler)   # For keyboard interrupt

def select_model_for_experiment(user_id, experiment):
    """Select model based on experiment traffic split"""
    # Convert traffic split to percentage (0-100)
    split_point = int(experiment.traffic_split * 100)
    
    # Use modulo to get a number between 0-99
    user_bucket = hash(str(user_id)) % 100
    
    # Assign to Model A if bucket is below split point
    if user_bucket < split_point:
        #print(f"Assigning user {user_id} to model A: {experiment.model_a_id} (bucket: {user_bucket})")
        return experiment.model_a_id, models[experiment.model_a_id]
    
    #print(f"Assigning user {user_id} to model B: {experiment.model_b_id} (bucket: {user_bucket})")
    return experiment.model_b_id, models[experiment.model_b_id]

def select_model(user_id):
    """Select model based on active experiments or default to first model"""
    #print(f"\nSelecting model for user {user_id}")
    #print(f"Active experiments: {list(experiment_manager.active_experiments.keys())}")
    
    for experiment in experiment_manager.active_experiments.values():
        #print(f"\nChecking experiment: {experiment.name}")
        #print(f"Models: A={experiment.model_a_id}, B={experiment.model_b_id}")
        #print(f"Traffic split: {experiment.traffic_split}")
        
        model_id, selected_model = select_model_for_experiment(user_id, experiment)
        if model_id:
            #print(f"User {user_id} assigned to model {model_id} in experiment {experiment.name}")
            return model_id, selected_model
    
    # Default to first model if no experiment matches
    default_model_id = MODEL_PATH.split('/')[-1]
    #print(f"No active experiments, using default model {default_model_id}")
    return default_model_id, models[default_model_id]

def recommend_movies(user_id):
    """Recommend movies for a user"""
    try:
        start_time = time.time()
        model_id, selected_model = select_model(user_id)
        #print(f"Selected model {model_id} for user {user_id}")
        
        # Get recommendations
        recommendations = utils.predict(selected_model, user_id, all_movies_list, user_movie_list)
        latency = time.time() - start_time
        
        # Calculate accuracy based on user's actual ratings
        user_ratings = utils.get_user_ratings(user_id)
        if user_ratings:
            # Get movie IDs and actual ratings
            movie_ids = [movie_id for movie_id, _ in user_ratings]
            actual_ratings = user_ratings  # Already in (movie_id, rating) format
            
            # Get predicted ratings in same format
            predicted_values = utils.get_predicted_ratings(selected_model, user_id, movie_ids)
            predicted_ratings = list(zip(movie_ids, predicted_values))
            
            #print(f"Predicted ratings: {predicted_ratings}")
            #print(f"Actual ratings: {actual_ratings}")

            # Calculate RMSE
            rmse = utils.calculate_rmse(predicted_ratings, actual_ratings)
            #print(f"Calculated RMSE: {rmse}")

            # Record metrics only if we have actual ratings to compare against
            for experiment in experiment_manager.active_experiments.values():
                #print(f"Checking experiment {experiment.name}")
                if model_id in [experiment.model_a_id, experiment.model_b_id]:
                    #print(f"Recording metrics for experiment {experiment.name}, model {model_id}")
                    experiment_manager.record_performance(
                        experiment.name,
                        model_id,
                        1 - rmse,  # Already normalized in calculate_rmse
                        latency
                    )
        else:
            #print(f"No ratings found for user {user_id}")
            pass

        return jsonify(recommendations)

    except Exception as e:
        print(f"Error in recommend_movies: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8082)
    finally:
        cleanup_experiments()
