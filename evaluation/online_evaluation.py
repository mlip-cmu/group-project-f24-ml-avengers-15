import traceback
import sys
import itertools
import os
import numpy as np
import time
from datetime import datetime
import psutil
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import recommend_movies
from config import DATA_DIR, MODEL_PATH
import mlflow
import pickle
from dotenv import load_dotenv
from mlflow.models import infer_signature

load_dotenv() 
mlflow.set_tracking_uri("http://127.0.0.1:6001")

experiment_name = "Movie_Recommender_Evaluation"
mlflow.set_experiment(experiment_name)

print("Loading the pre-trained model...")
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)
print("Model loaded successfully.")

hyperparameter_grid = {
    "n_factors": [50, 100, 150],
    "learning_rate": [0.005, 0.01, 0.02],
    "reg_all": [0.01, 0.02, 0.05]
}

def calculate_recall_at_k(user_recommendations, user_relevant_movies, k=10):
    total_recall = 0
    total_users = 0

    for user_id, rec_movies in user_recommendations.items():
        relevant_movies = user_relevant_movies.get(user_id, set())
        if relevant_movies:
            top_k_recommendations = set(rec_movies[:k])
            relevant_count = len(top_k_recommendations.intersection(relevant_movies))
            recall = relevant_count / len(relevant_movies)
            total_recall += recall
            total_users += 1

    return total_recall / total_users if total_users > 0 else 0

def calculate_ndcg_at_k(user_recommendations, user_relevant_movies, k=10):
    total_ndcg = 0
    total_users = 0

    for user_id, rec_movies in user_recommendations.items():
        relevant_movies = user_relevant_movies.get(user_id, set())
        if relevant_movies:
            dcg = 0
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_movies), k)))
            for i, movie in enumerate(rec_movies[:k]):
                if movie in relevant_movies:
                    dcg += 1 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            total_ndcg += ndcg
            total_users += 1

    return total_ndcg / total_users if total_users > 0 else 0

# Calculate Precision@K
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

def calculate_diversity(recommendations):
    unique_movies = len(set(recommendations))
    return unique_movies / len(recommendations) if recommendations else 0

run_count = 1 

def evaluate_snapshot(n_factors, learning_rate, reg_all):
    global run_count
    try:
        
        print("Running online_evaluation.py...")

        snapshot_df = pd.read_csv(DATA_DIR + "/extracted_ratings.csv")
        user_relevant_movies = get_user_relevant_movies(snapshot_df)

        start_time = time.time()
        user_recommendations = {user_id: recommend_movies(user_id) for user_id in user_relevant_movies.keys()}
        end_time = time.time()

        inference_latency_ms = (end_time - start_time) * 1000
        cpu_usage_percent = psutil.cpu_percent(interval=None)
        memory_usage_mb = psutil.virtual_memory().used / (1024 ** 2)

        # Calculate snapshot-level metrics
        precision = calculate_precision_at_k(user_recommendations, user_relevant_movies, 10)
        recall = calculate_recall_at_k(user_recommendations, user_relevant_movies, 10)
        ndcg = calculate_ndcg_at_k(user_recommendations, user_relevant_movies, 10)
        diversity = calculate_diversity([item for sublist in user_recommendations.values() for item in sublist])

        #Update run name for every num
        run_name = f"snapshot_evaluation_run_{run_count}"

        with mlflow.start_run(run_name=run_name) as run:
            run_count += 1
            mlflow.set_tag("type", "snapshot_level")
            mlflow.log_metric("Precision_at_10", precision)
            mlflow.log_metric("Recall_at_10", recall)
            mlflow.log_metric("NDCG_at_10", ndcg)
            mlflow.log_metric("diversity", diversity)
            mlflow.log_metric("Inference_Latency_ms", inference_latency_ms)
            mlflow.log_metric("CPU_Usage_percent", cpu_usage_percent)
            mlflow.log_metric("Memory_Usage_MB", memory_usage_mb)

            # Log hyperparameters
            mlflow.log_param("model_type", "SVD")
            mlflow.log_param("model_path", MODEL_PATH)
            mlflow.log_param("rating_threshold", 4)
            mlflow.log_param("K", 10)
            mlflow.log_param("n_factors", n_factors)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("reg_all", reg_all)

            # Basic data stats
            rating_mean = snapshot_df["rating"].mean()
            rating_std = snapshot_df["rating"].std()
            mlflow.log_metric("rating_mean", rating_mean)
            mlflow.log_metric("rating_std", rating_std)

            # Log input data as artifact
            mlflow.log_artifact(DATA_DIR + "/extracted_ratings.csv")

            sample_input = snapshot_df[["user_id", "movie_id", "rating"]].head(1) 
            signature = infer_signature(sample_input, model.predict(uid=sample_input["user_id"].iloc[0], iid=sample_input["movie_id"].iloc[0]).est)

            print("Logging the model to MLflow...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="SVD_Recommender_new",
                signature=signature,
                input_example=sample_input,
            )
            print("Model logged successfully.")

        with open("online_evaluation_output.txt", 'w') as f:
            f.write(f"Precision@10: {precision:.4f}\n")
    
    except Exception as e:
        print("Error in evaluation:")
        traceback.print_exc()
    
if __name__ == "__main__":
    hyperparameter_combinations = list(itertools.product(
        hyperparameter_grid["n_factors"],
        hyperparameter_grid["learning_rate"],
        hyperparameter_grid["reg_all"]
    ))
    for n_factors, learning_rate, reg_all in hyperparameter_combinations:
        evaluate_snapshot(n_factors, learning_rate, reg_all)

