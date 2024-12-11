import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load data
ratings_df = pd.read_csv('../data/extracted_ratings.csv')
users_df = pd.read_csv('../data/user_details.csv')
movies_df = pd.read_csv('../data/movie_details.csv')

# Merge datasets
full_df = ratings_df.merge(users_df, on='user_id', how='left')
full_df = full_df.merge(movies_df[['movie_id', 'genres']], on='movie_id', how='left')
# Convert timestamp to datetime with flexible parsing
full_df['timestamp'] = pd.to_datetime(full_df['user_time'], format='mixed', errors='coerce')

def calculate_rating_intervals(group):
    # Sort by timestamp
    group = group.sort_values('timestamp')
    
    # Calculate time differences between consecutive ratings in seconds
    intervals = group['timestamp'].diff().dt.total_seconds()
    
    return pd.Series({
        'min_interval': intervals.min() if len(intervals) > 0 else np.nan,
        'max_interval': intervals.max() if len(intervals) > 0 else np.nan,
        'mean_interval': intervals.mean() if len(intervals) > 0 else np.nan,
        'std_interval': intervals.std() if len(intervals) > 0 else np.nan
    })

def detect_rating_manipulation():
    print("=== Rating Manipulation Detection ===")
    
    # Calculate basic user statistics
    user_stats = full_df.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std'],
        'timestamp': ['min', 'max']
    })
    
    # Calculate time-based patterns
    time_patterns = full_df.groupby('user_id').apply(calculate_rating_intervals).reset_index()
    time_patterns.columns = ['user_id'] + [f'interval_{col}' for col in time_patterns.columns[1:]]
    
    # Merge statistics
    user_stats = user_stats.reset_index()
    user_stats.columns = ['user_id'] + [f'{col[0]}_{col[1]}' for col in user_stats.columns[1:]]
    user_stats = user_stats.merge(time_patterns, on='user_id', how='left')
    
    # Calculate days active
    user_stats['days_active'] = (user_stats['timestamp_max'] - user_stats['timestamp_min']).dt.total_seconds() / (24*3600)
    
    # Calculate ratings per day
    user_stats['ratings_per_day'] = user_stats['rating_count'] / user_stats['days_active'].clip(lower=0.001)
    
    # Define suspicious behavior criteria
    suspicious_users = user_stats[
        ((user_stats['rating_count'] > 1) & (user_stats['rating_std'] == 0)) |  # Same rating repeatedly
        (user_stats['ratings_per_day'] > 10) |  # Too many ratings per day
        ((user_stats['interval_min_interval'] < 10) & (user_stats['rating_count'] > 1))  # Ratings too close together
    ]
    
    print(f"\nPotential Suspicious Users:")
    print(f"Found {len(suspicious_users)} suspicious users out of {len(user_stats)} total users")
    
    if len(suspicious_users) > 0:
        print("\nSample of suspicious users:")
        cols = ['user_id', 'rating_count', 'rating_mean', 'rating_std', 
                'timestamp_min', 'timestamp_max', 'days_active', 'ratings_per_day']
        print(suspicious_users[cols].head().to_string())
    
    return suspicious_users

def detect_model_poisoning():
    print("\n=== Model Poisoning Detection ===")
    
    # Analyze rating distributions over time
    full_df['week'] = full_df['timestamp'].dt.isocalendar().week
    full_df['year'] = full_df['timestamp'].dt.year
    
    # Calculate weekly rating statistics
    weekly_stats = full_df.groupby(['year', 'week']).agg({
        'rating': ['mean', 'std', 'count'],
        'user_id': 'nunique',
        'movie_id': 'nunique'
    })
    
    # Detect anomalous weeks
    def detect_anomalies(series, threshold=2):
        mean = series.mean()
        std = series.std()
        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold
    
    anomalies = {
        'rating_mean': detect_anomalies(weekly_stats[('rating', 'mean')]),
        'rating_std': detect_anomalies(weekly_stats[('rating', 'std')]),
        'rating_count': detect_anomalies(weekly_stats[('rating', 'count')]),
        'unique_users': detect_anomalies(weekly_stats[('user_id', 'nunique')]),
        'unique_movies': detect_anomalies(weekly_stats[('movie_id', 'nunique')])
    }
    
    print("\nAnomalous weeks detected:")
    for metric, anomaly in anomalies.items():
        anomalous_weeks = weekly_stats[anomaly]
        print(f"\n{metric}: {len(anomalous_weeks)} anomalous weeks")
        if len(anomalous_weeks) > 0:
            print(anomalous_weeks.head())

def analyze_rating_patterns():
    print("\n=== Suspicious Rating Pattern Analysis ===")
    
    # Calculate movie statistics and rating distributions
    movie_stats = full_df.groupby('movie_id').agg({
        'rating': ['count', 'mean', 'std']
    }).round(3)
    
    # Reset index and rename columns
    movie_stats = movie_stats.reset_index()
    movie_stats.columns = ['movie_id'] + [f'{col[0]}_{col[1]}' for col in movie_stats.columns[1:]]
    
    # Calculate rating distributions
    rating_dists = []
    for movie_id in movie_stats['movie_id'].unique():
        movie_ratings = full_df[full_df['movie_id'] == movie_id]['rating']
        dist = movie_ratings.value_counts(normalize=True)
        row = {'movie_id': movie_id}
        for rating in range(1, 6):
            row[f'pct_rating_{rating}'] = dist.get(rating, 0)
        rating_dists.append(row)
    
    rating_dist_df = pd.DataFrame(rating_dists)
    
    # Merge all statistics
    movie_stats = movie_stats.merge(rating_dist_df, on='movie_id', how='left')
    
    # Identify suspicious patterns
    rating_cols = [f'pct_rating_{i}' for i in range(1, 6)]
    suspicious_movies = movie_stats[
        (movie_stats['rating_count'] > 10) &  # Only consider movies with sufficient ratings
        (
            (movie_stats['rating_std'] < 0.5) |  # Very low rating variance
            (movie_stats[rating_cols].max(axis=1) > 0.8)  # One rating value > 80% of all ratings
        )
    ]
    
    print(f"\nFound {len(suspicious_movies)} suspicious movies out of {len(movie_stats)} total movies")
    
    if len(suspicious_movies) > 0:
        print("\nSample of suspicious movies:")
        print(suspicious_movies[['movie_id', 'rating_count', 'rating_mean', 'rating_std']].head().to_string())
        
        print("\nExtreme rating distributions:")
        for _, movie in suspicious_movies.head().iterrows():
            print(f"\nMovie: {movie['movie_id']}")
            print(f"Total ratings: {int(movie['rating_count'])}")
            print(f"Mean rating: {movie['rating_mean']:.2f}")
            print("Rating distribution:")
            for i in range(1, 6):
                pct = movie[f'pct_rating_{i}'] * 100
                if pct > 0:
                    print(f"  {i} stars: {pct:.1f}%")
    
    return suspicious_movies

if __name__ == "__main__":
    suspicious_users = detect_rating_manipulation()
    detect_model_poisoning()
    analyze_rating_patterns()
