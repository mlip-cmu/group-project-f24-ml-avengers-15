import traceback
import pandas as pd
from app import recommend_movies
from config import DATA_PATH

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

def evaluate_snapshot(DATA_PATH: str):
    try:
        snapshot_df = pd.read_csv(DATA_PATH)
        user_relevant_movies = get_user_relevant_movies(snapshot_df)
        user_recommendations = {user_id: recommend_movies(user_id) for user_id in user_relevant_movies.keys()}
        
        # Calculate Precision@K
        precision = calculate_precision_at_k(user_recommendations, user_relevant_movies, 10)
        with open("online_evaluation_output.txt", 'w') as f:
            f.write(f"Precision@10: {precision:.4f}\n")
    
    except Exception as e:
        with open("online_evaluation_output.txt", 'w') as f:
            f.write(f"An error occurred while calculating Precision@K: {e}")
