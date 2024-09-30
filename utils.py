import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
import time
import os


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


def train_model(train_data, model_name='SVD'):
    model = SVD(n_factors=100, n_epochs=20, biased=True, lr_all=0.005, reg_all=0.02)
    start_time = time.time()
    training_set = train_data.build_full_trainset()
    model.fit(training_set)
    training_time = time.time() - start_time
    training_time_ms = training_time * 1000



    model_filename = f'{model_name}_movie_recommender.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)

    return model,training_time_ms


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

def predict(model, user_id, movie_list, user_movie_list, K=20):

    recommendations = []
    scores = []

    for movie in movie_list:
        if user_id in user_movie_list and movie in user_movie_list[user_id]:
            continue
        prediction = model.predict(uid=user_id, iid=movie)
        scores.append((prediction.est, movie))

    scores.sort(reverse=True)
    recommendations = [movie for _, movie in scores[:K]]

    return recommendations