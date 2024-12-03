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

# 1. Fairness Analysis
def analyze_rating_distribution():
    print("=== Rating Distribution Analysis ===")
    
    # By Gender
    gender_stats = full_df.groupby('gender')['rating'].agg(['mean', 'std', 'count'])
    print("\nRating Statistics by Gender:")
    print(gender_stats)
    
    # By Age Group
    full_df['age_group'] = pd.cut(full_df['age'], 
                                 bins=[0, 18, 25, 35, 50, 100],
                                 labels=['Under 18', '18-25', '26-35', '36-50', 'Over 50'])
    age_stats = full_df.groupby('age_group')['rating'].agg(['mean', 'std', 'count'])
    print("\nRating Statistics by Age Group:")
    print(age_stats)
    
    # By Occupation
    occ_stats = full_df.groupby('occupation')['rating'].agg(['mean', 'std', 'count'])
    print("\nRating Statistics by Occupation:")
    print(occ_stats)

def analyze_genre_exposure():
    print("\n=== Genre Exposure Analysis ===")
    
    # Split genres and explode to separate rows
    genre_df = full_df.copy()
    genre_df['genres'] = genre_df['genres'].str.split('|')
    genre_df = genre_df.explode('genres')
    
    # Calculate genre exposure by demographic groups
    gender_genre = pd.crosstab(genre_df['gender'], genre_df['genres'], normalize='index')
    age_genre = pd.crosstab(genre_df['age_group'], genre_df['genres'], normalize='index')
    
    print("\nGenre Distribution by Gender:")
    print(gender_genre)
    print("\nGenre Distribution by Age Group:")
    print(age_genre)

# 2. Feedback Loop Analysis
def analyze_temporal_patterns():
    print("\n=== Temporal Pattern Analysis ===")
    
    # Convert timestamp to datetime with flexible parsing
    full_df['timestamp'] = pd.to_datetime(full_df['user_time'], format='mixed', errors='coerce')
    
    # Calculate monthly statistics
    monthly_stats = full_df.groupby(full_df['timestamp'].dt.to_period('M')).agg({
        'rating': ['mean', 'std', 'count'],
        'movie_id': 'nunique'
    })
    
    print("\nMonthly Rating Statistics:")
    print(monthly_stats)
    
    # Calculate genre diversity over time
    genre_df = full_df.copy()
    genre_df['genres'] = genre_df['genres'].str.split('|')
    genre_df = genre_df.explode('genres')
    
    genre_diversity = genre_df.groupby(genre_df['timestamp'].dt.to_period('M'))['genres'].agg(['nunique', 'count'])
    genre_diversity['diversity_ratio'] = genre_diversity['nunique'] / genre_diversity['count']
    
    print("\nGenre Diversity Over Time:")
    print(genre_diversity)

# 3. Security Analysis
def detect_suspicious_patterns():
    print("\n=== Security Analysis ===")
    
    # Analyze rating patterns per user
    user_patterns = full_df.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std'],
        'timestamp': ['min', 'max']
    })
    
    # Calculate rating velocity (ratings per day)
    user_patterns['days_active'] = (
        user_patterns[('timestamp', 'max')] - user_patterns[('timestamp', 'min')]
    ).dt.total_seconds() / (24 * 3600)
    
    user_patterns['ratings_per_day'] = user_patterns[('rating', 'count')] / user_patterns['days_active']
    
    # Flag suspicious users
    suspicious_users = user_patterns[
        (user_patterns['ratings_per_day'] > user_patterns['ratings_per_day'].mean() + 2*user_patterns['ratings_per_day'].std()) |
        (user_patterns[('rating', 'std')] < 0.5) # Users with very low rating variance
    ]
    
    print("\nPotential Suspicious Users:")
    print(f"Found {len(suspicious_users)} suspicious users out of {len(user_patterns)} total users")
    print("\nSample of suspicious users:")
    print(suspicious_users.head())

if __name__ == "__main__":
    analyze_rating_distribution()
    analyze_genre_exposure()
    analyze_temporal_patterns()
    detect_suspicious_patterns()
