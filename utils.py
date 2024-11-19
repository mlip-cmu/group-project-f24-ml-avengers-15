import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
import time
import os
import numpy as np

# Get the base directory once at module level
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_data_csv(file_path, split_ratio=0.8):
    if not os.path.isabs(file_path):
        file_path = os.path.join(BASE_DIR, file_path)
    df = pd.read_csv(file_path)
    
    # Keep original movie IDs/titles, no need to create new numeric IDs
    unique_movies = df['movie_id'].unique()
    
    df = df.sort_values(by=['user_id', 'user_time'])

    train_data_list = []
    val_data_list = []

    for _, group in df.groupby('user_id'):
        split_index = int(len(group) * split_ratio)
        train_data_list.append(group.iloc[:split_index])
        val_data_list.append(group.iloc[split_index:])

    train_df = pd.concat(train_data_list).reset_index(drop=True)
    val_df = pd.concat(val_data_list).reset_index(drop=True)

    # Save movie ID mapping with original IDs
    movie_map_df = pd.DataFrame({'movie_id': unique_movies})
    mapping_file = os.path.join(BASE_DIR, 'data', 'movie_id_mapping.csv')
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    movie_map_df.to_csv(mapping_file, index=False)

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

    model_filename = os.path.join(BASE_DIR, f'{model_name}')
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)

    return model,training_time_ms

def train_model2(train_data, model_name='SVD'):
    model = SVD(n_factors=50, n_epochs=10, biased=True, lr_all=0.001, reg_all=0.5)
    start_time = time.time()
    training_set = train_data.build_full_trainset()
    model.fit(training_set)
    training_time = time.time() - start_time
    training_time_ms = training_time * 1000

    model_filename = os.path.join(BASE_DIR, f'{model_name}')
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
    model_filename = os.path.join(BASE_DIR, model_filename)
    # Get the size of the model in bytes
    return os.path.getsize(model_filename)

def predict(model, user_id, movie_list, user_movie_list, K=20):
    recommendations = []
    scores = []

    try:
        # Convert user_id to int but leave movie_ids as strings
        user_id_int = int(user_id)
        
        for movie in movie_list:
            if user_id in user_movie_list and movie in user_movie_list[user_id]:
                continue
            prediction = model.predict(user_id_int, movie)
            scores.append((prediction.est, movie))

        scores.sort(reverse=True)
        recommended_ids = [movie for _, movie in scores[:K]]
        
        # Format movie titles for output
        recommendations = [movie_id.replace(' ', '+') for movie_id in recommended_ids]
        return recommendations
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return []

def generate_test_ratings(user_id, num_ratings=10):
    """Generate test ratings for a user to simulate real usage"""
    try:
        # Read the movie mapping
        movie_map_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'movie_id_mapping.csv'))
        # Read the movie mapping - now just a list of valid movies
        movie_map_df = pd.read_csv('data/movie_id_mapping.csv')
        
        # Randomly select movies and generate ratings
        selected_movies = movie_map_df.sample(n=min(num_ratings, len(movie_map_df)))
        ratings = np.random.uniform(1, 5, size=len(selected_movies))
        
        # Return (movie_id, rating) tuples using original movie IDs
        return list(zip(selected_movies['movie_id'], ratings))
    except Exception as e:
        print(f"Error generating test ratings: {str(e)}")
        return []

def get_user_ratings(user_id):
    """Get all ratings for a given user"""
    try:
        # Read ratings using absolute path
        ratings_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'extracted_ratings.csv'))
        
        # Filter ratings for the given user
        user_ratings = ratings_df[ratings_df['user_id'] == int(user_id)]
        
        # If no real ratings exist, generate test ratings
        if len(user_ratings) == 0:
            print(f"No real ratings found for user {user_id}, generating test ratings")
            return generate_test_ratings(user_id)
            
        # Return list of (movie_id, rating) tuples directly
        return list(zip(user_ratings['movie_id'], user_ratings['rating']))
    except Exception as e:
        print(f"Error getting user ratings: {str(e)}")
        return []

def get_predicted_ratings(model, user_id, movie_ids):
    """Get predicted ratings for specific movies"""
    try:
        # Only convert user_id to int, leave movie_ids as strings
        return [model.predict(int(user_id), movie_id).est for movie_id in movie_ids]
    except Exception as e:
        print(f"Error getting predicted ratings: {e}")
        return []

def calculate_rmse(predicted, actual):
    """
    Calculate RMSE between actual and predicted ratings
    
    Args:
        predicted: List of (movie_id, rating) tuples for predicted ratings
        actual: List of (movie_id, rating) tuples for actual ratings
        
    Returns:
        float: RMSE value
    """
    try:
        # Convert to dictionaries for easier lookup
        actual_dict = dict(actual)
        pred_dict = dict(predicted)
        
        # Get common movie IDs
        common_movies = set(actual_dict.keys()) & set(pred_dict.keys())
        
        if not common_movies:
            return 1.0  # Return worst case when no common movies
            
        # Calculate squared differences
        squared_diff = [(actual_dict[movie_id] - pred_dict[movie_id])**2 
                       for movie_id in common_movies]
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(squared_diff))
        
        # Normalize to 0-1 scale (assuming ratings are 1-5)
        normalized_rmse = min(rmse / 4.0, 1.0)  # 4.0 is max possible RMSE for 1-5 scale
        
        return normalized_rmse
    except Exception as e:
        print(f"Error calculating RMSE: {str(e)}")
        return 1.0  # Return worst case on error