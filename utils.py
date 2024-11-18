import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
import mlflow
import time
import os
import json
from mlflow.models import infer_signature

# mlflow.set_tracking_uri("./mlruns")
uri = "http://127.0.0.1:6001"

mlflow.set_tracking_uri(uri)
experiment_name = "Movie Recommendation Experiment"
mlflow.set_experiment(experiment_name)

def prepare_data_csv(file_path, split_ratio=0.8):

    df = pd.read_csv(file_path)

    df = df.sort_values(by=['user_id', 'user_time'])

    train_data_list = []
    val_data_list = []

    for user_id, group in df.groupby('user_id'):
        split_index = int(len(group) * split_ratio)
        train_data_list.append(group.iloc[:split_index])
        val_data_list.append(group.iloc[split_index:])

    train_df = pd.concat(train_data_list).reset_index(drop=True)
    val_df = pd.concat(val_data_list).reset_index(drop=True)

    return train_df, val_df


def prepare_data_model(train_df, val_df, rating_range=(1, 5)):

    train_reader = Reader(rating_scale=rating_range)
    val_reader = Reader(rating_scale=rating_range)

    train_data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], train_reader)
    valid_data = Dataset.load_from_df(val_df[["user_id", "movie_id", "rating"]], val_reader)

    return train_data, valid_data


def train_model(train_data, model_version='SVDv1', parameters=None):
    if parameters is None:
        parameters = {'n_factors': 100, 'n_epochs': 20, 'biased': True, 'lr_all': 0.005, 'reg_all': 0.02}

    model = SVD(**parameters)

    start_time = time.time()
    training_set = train_data.build_full_trainset()
    model.fit(training_set)
    training_time = time.time() - start_time

    return model, training_time, parameters


def evaluate(model, data):

    dataset = [(rating[0], rating[1], rating[2]) for rating in data.raw_ratings]

    predictions = model.test(dataset)

    rmse = accuracy.rmse(predictions, verbose=True)
    return rmse

def inference_cost_per_input(model, user_id, movie_id):
    start_time = time.time()
    model.predict(uid=user_id, iid=movie_id)
    inference_time_seconds = time.time() - start_time
    inference_time_ms = inference_time_seconds * 1000  # Convert to milliseconds
    return inference_time_ms

def get_model_size(model_filename):
    # Get the size of the model in bytes
    return os.path.getsize(model_filename)

def predict(model, user_id, movie_list, user_movie_list, model_version, parameters, pipeline_version, train_data, K=20):
    recommendations = []
    scores = []

    for movie in movie_list:
        if user_id in user_movie_list and movie in user_movie_list[user_id]:
            continue
        prediction = model.predict(uid=user_id, iid=movie)
        scores.append((prediction.est, movie))

    scores.sort(reverse=True)
    recommendations = [movie for _, movie in scores[:K]]

    # Log predictions and provenance
    recommendations_file = "recommendations.json"
    with open(recommendations_file, "w") as rec_file:
        json.dump({"user_id": user_id, "recommendations": recommendations}, rec_file)
    mlflow.log_artifact(recommendations_file, artifact_path="predictions")

    provenance_info = {
        "model_version": model_version,
        "parameters": parameters,
        "pipeline_version": pipeline_version,
        "training_data": {
            "file_path": "data/extracted_ratings.csv",
            "split_ratio": 0.8,
            "record_count": len(train_data.raw_ratings),
        },
    }
    provenance_file = "provenance_info.json"
    with open(provenance_file, "w") as prov_file:
        json.dump(provenance_info, prov_file)
    mlflow.log_artifact(provenance_file, artifact_path="provenance")

    return recommendations


if __name__ == "__main__":
    
    svd1_parameters = {'n_factors': 100, 'n_epochs': 20, 'biased': True, 'lr_all': 0.005, 'reg_all': 0.02}
    svd2_parameters = {'n_factors': 50, 'n_epochs': 10, 'biased': True, 'lr_all': 0.001, 'reg_all': 0.5}
    models = {
        "SVDv1": ("models/SVD_movie_recommender.pkl", svd1_parameters),
        "SVDv2": ("models/SVD_movie_recommender_2.pkl", svd2_parameters),
    }
    ratings_file = "data/extracted_ratings.csv"

    train_df, val_df = prepare_data_csv(ratings_file)
    train_data, valid_data = prepare_data_model(train_df, val_df)
    pipeline_version = os.popen("git rev-parse --short HEAD").read().strip()

    all_movies = train_df['movie_id'].unique().tolist()
    user_movies = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()
    test_user_id = 93
    test_movie_id = train_df['movie_id'].iloc[0]

    model_version = "SVDv1" 
    model_path, parameters = models[model_version]

    # Load the model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with mlflow.start_run(run_name=f"Prediction-{model_version}_Pipeline-{pipeline_version}"):

        mlflow.set_tag("Model Type", "SVD")
        mlflow.set_tag("Model Version", model_version)
        mlflow.set_tag("Pipeline Version", pipeline_version)
        mlflow.log_params(parameters)

        mlflow.log_artifact(ratings_file, artifact_path="training_data")

        mlflow.log_artifact(model_path, artifact_path="model")

        recommendations = predict(model, test_user_id, all_movies, user_movies, model_version, parameters, pipeline_version, train_data, K=20)
        print(f"Top 20 recommendations for user {test_user_id}: {recommendations}")

        inference_time_ms = inference_cost_per_input(model, test_user_id, test_movie_id)
        mlflow.log_metric("inference_time_ms", inference_time_ms)

      



