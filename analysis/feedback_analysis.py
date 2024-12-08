import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import os

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load data
def load_data():
    """Load and prepare the dataset."""
    ratings_df = pd.read_csv(DATA_DIR / "extracted_ratings.csv")
    users_df = pd.read_csv(DATA_DIR / "user_details.csv")
    movies_df = pd.read_csv(DATA_DIR / "movie_details.csv")
    
    # Merge datasets
    full_df = ratings_df.merge(users_df, on='user_id', how='left')
    full_df = full_df.merge(movies_df[['movie_id', 'genres']], on='movie_id', how='left')
    
    # Convert timestamps with mixed format
    full_df['timestamp'] = pd.to_datetime(full_df['user_time'], format='mixed', errors='coerce')
    
    # Remove any rows with invalid timestamps
    full_df = full_df.dropna(subset=['timestamp'])
    print(f"Loaded {len(full_df)} valid ratings after cleaning timestamps")
    
    return full_df

full_df = load_data()

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

def analyze_genre_concentration(df):
    """Analyze genre concentration within the dataset."""
    print("\n=== Genre Concentration Analysis ===")
    
    # Calculate genre diversity per user
    user_genres = df.groupby('user_id')['genres'].nunique()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=user_genres, bins=30)
    plt.title('Genre Diversity Distribution per User')
    plt.xlabel('Number of Unique Genres')
    plt.ylabel('Number of Users')
    plt.savefig(PLOTS_DIR / 'genre_diversity.png')
    plt.close()
    
    print(f"\nGenre Diversity Statistics:")
    print(f"Average genres per user: {user_genres.mean():.2f}")
    print(f"Median genres per user: {user_genres.median():.2f}")
    print(f"Users with single genre: {(user_genres == 1).sum()}")
    
    return user_genres

def analyze_rating_patterns(df):
    """Analyze rating patterns within the dataset."""
    print("\n=== Rating Pattern Analysis ===")
    
    # Calculate rating statistics per movie
    movie_stats = df.groupby('movie_id')['rating'].agg(['count', 'mean', 'std']).reset_index()
    
    # Plot rating distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(data=movie_stats, x='mean', bins=20, ax=ax1)
    ax1.set_title('Distribution of Average Movie Ratings')
    ax1.set_xlabel('Mean Rating')
    
    sns.histplot(data=movie_stats, x='std', bins=20, ax=ax2)
    ax2.set_title('Distribution of Rating Standard Deviations')
    ax2.set_xlabel('Rating Standard Deviation')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'rating_patterns.png')
    plt.close()
    
    print("\nRating Statistics:")
    print(f"Movies with high consensus (std < 0.5): {len(movie_stats[movie_stats['std'] < 0.5])}")
    print(f"Movies with high ratings (mean > 4.0): {len(movie_stats[movie_stats['mean'] > 4.0])}")
    print(f"Movies with low ratings (mean < 2.0): {len(movie_stats[movie_stats['mean'] < 2.0])}")
    
    return movie_stats

def analyze_rating_distribution(df):
    """Analyze detailed rating distribution patterns."""
    print("\n=== Rating Distribution Analysis ===")
    
    # Overall rating distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x='rating', bins=10)
    plt.title('Overall Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # Rating distribution by gender
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='gender', y='rating')
    plt.title('Rating Distribution by Gender')
    
    # Rating distribution by genre
    plt.subplot(1, 3, 3)
    genre_ratings = df.groupby('genres')['rating'].mean().sort_values(ascending=False)
    sns.barplot(x=genre_ratings.head(10).values, y=genre_ratings.head(10).index)
    plt.title('Average Rating by Top 10 Genres')
    plt.xlabel('Average Rating')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'rating_distribution.png')
    plt.close()
    
    print("\nRating Distribution Statistics:")
    print(f"Overall average rating: {df['rating'].mean():.2f}")
    print(f"Rating distribution:")
    print(df['rating'].value_counts().sort_index())
    print(f"\nTop 5 highest-rated genres:")
    print(genre_ratings.head())
    print(f"\nBottom 5 lowest-rated genres:")
    print(genre_ratings.tail())
    
    return genre_ratings

if __name__ == "__main__":
    analyze_popularity_feedback()
    analyze_demographic_bubbles()
    analyze_genre_concentration(full_df)
    analyze_rating_patterns(full_df)
    analyze_rating_distribution(full_df)
    
    print("\n=== Generated Visualizations ===")
    print("The following plots have been generated in the 'plots' directory:")
    print("1. genre_diversity.png")
    print("   - Histogram showing distribution of unique genres per user")
    print("\n2. rating_patterns.png")
    print("   - Left: Distribution of average movie ratings")
    print("   - Right: Distribution of rating standard deviations")
    print("\n3. rating_distribution.png")
    print("   - Left: Overall rating distribution histogram")
    print("   - Middle: Rating distribution by gender (boxplot)")
    print("   - Right: Average rating by top 10 genres (bar plot)")
