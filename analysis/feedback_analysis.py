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

def analyze_popularity_feedback():
    print("=== Popularity Feedback Loop Analysis ===")
    
    # Split data into time periods
    full_df['period'] = pd.qcut(full_df['timestamp'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Calculate movie popularity over time
    popularity_evolution = full_df.groupby(['period', 'movie_id']).size().reset_index(name='ratings_count')
    popularity_evolution = popularity_evolution.pivot(index='movie_id', columns='period', values='ratings_count').fillna(0)
    
    # Calculate correlation between consecutive periods
    correlations = []
    periods = popularity_evolution.columns
    for i in range(len(periods)-1):
        corr = stats.spearmanr(popularity_evolution[periods[i]], popularity_evolution[periods[i+1]])[0]
        correlations.append(corr)
    
    print("\nPopularity Correlation between consecutive periods:")
    for i, corr in enumerate(correlations):
        print(f"{periods[i]} to {periods[i+1]}: {corr:.3f}")
    
    # Calculate popularity concentration
    gini_coefficients = []
    for period in periods:
        ratings = popularity_evolution[period].sort_values()
        n = len(ratings)
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * ratings).sum() / (n * ratings.sum())
        gini_coefficients.append(gini)
    
    print("\nGini Coefficients (Popularity Concentration) by Period:")
    for period, gini in zip(periods, gini_coefficients):
        print(f"{period}: {gini:.3f}")

def analyze_demographic_bubbles():
    print("\n=== Demographic Bubble Analysis ===")
    
    # Split data into time periods
    full_df['period'] = pd.qcut(full_df['timestamp'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Analyze genre preferences by demographic group over time
    genre_df = full_df.copy()
    genre_df['genres'] = genre_df['genres'].str.split('|')
    genre_df = genre_df.explode('genres')
    
    # Calculate genre preferences by gender over time
    gender_genre_evolution = pd.crosstab(
        [genre_df['period'], genre_df['gender']], 
        genre_df['genres'], 
        normalize='index'
    )
    
    print("\nGenre Preference Evolution by Gender:")
    print(gender_genre_evolution)
    
    # Calculate similarity between demographic groups over time
    def calculate_genre_similarity(period_data):
        genres_m = period_data.loc[period_data.index.get_level_values('gender') == 'M']
        genres_f = period_data.loc[period_data.index.get_level_values('gender') == 'F']
        return np.corrcoef(genres_m.values[0], genres_f.values[0])[0,1]
    
    similarities = []
    for period in ['Q1', 'Q2', 'Q3', 'Q4']:
        period_data = gender_genre_evolution.loc[period]
        similarity = calculate_genre_similarity(period_data)
        similarities.append(similarity)
    
    print("\nGender Group Similarity Over Time (correlation):")
    for period, sim in zip(['Q1', 'Q2', 'Q3', 'Q4'], similarities):
        print(f"{period}: {sim:.3f}")

def analyze_rating_patterns():
    print("\n=== Rating Pattern Analysis ===")
    
    # Calculate average rating by movie over time
    full_df['month'] = full_df['timestamp'].dt.to_period('M')
    monthly_ratings = full_df.groupby(['month', 'movie_id'])['rating'].mean().reset_index()
    
    # Calculate rating stability
    rating_stability = monthly_ratings.groupby('movie_id')['rating'].std().describe()
    
    print("\nRating Stability Statistics:")
    print(rating_stability)
    
    # Analyze rating trends
    def calculate_trend(group):
        if len(group) < 2:
            return 0
        x = np.arange(len(group))
        slope, _, _, _, _ = stats.linregress(x, group['rating'])
        return slope
    
    movie_trends = monthly_ratings.groupby('movie_id').apply(calculate_trend)
    
    print("\nRating Trends Summary:")
    print(f"Movies with increasing trend: {(movie_trends > 0.1).sum()}")
    print(f"Movies with decreasing trend: {(movie_trends < -0.1).sum()}")
    print(f"Movies with stable ratings: {((movie_trends >= -0.1) & (movie_trends <= 0.1)).sum()}")

if __name__ == "__main__":
    analyze_popularity_feedback()
    analyze_demographic_bubbles()
    analyze_rating_patterns()
