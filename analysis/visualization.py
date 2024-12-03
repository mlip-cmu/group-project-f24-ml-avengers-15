import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for all plots
plt.style.use('seaborn-v0_8')  # Updated to use correct style name
sns.set_theme(style="whitegrid")  # Added explicit seaborn theme

# Create output directory for plots
output_dir = Path(os.path.dirname(__file__)) / "plots"
output_dir.mkdir(exist_ok=True)

def clean_timestamp(ts):
    """Clean and parse timestamp strings."""
    try:
        return pd.to_datetime(ts)
    except:
        # Handle malformed timestamps by returning NaT (Not a Time)
        return pd.NaT

def load_data():
    """Load and prepare all necessary data."""
    project_root = Path(os.path.dirname(os.path.dirname(__file__)))
    
    # Load data files from project root
    ratings_df = pd.read_csv(project_root / "data" / "extracted_ratings.csv")
    users_df = pd.read_csv(project_root / "data" / "user_details.csv")
    movies_df = pd.read_csv(project_root / "data" / "movie_details.csv")
    
    # Merge datasets
    full_df = ratings_df.merge(users_df, on='user_id', how='left')
    full_df = full_df.merge(movies_df[['movie_id', 'genres']], on='movie_id', how='left')
    
    # Clean timestamps
    full_df['timestamp'] = full_df['user_time'].apply(clean_timestamp)
    
    # Remove rows with invalid timestamps
    full_df = full_df.dropna(subset=['timestamp'])
    
    print(f"Loaded {len(full_df)} ratings after cleaning timestamps")
    return full_df

def plot_fairness_metrics(df):
    """Generate plots for fairness analysis."""
    print("Generating fairness plots...")
    
    # 1. Rating Distribution by Gender
    if 'gender' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='gender', y='rating', data=df.dropna(subset=['gender', 'rating']))
        plt.title('Rating Distribution by Gender')
        plt.savefig(output_dir / 'gender_rating_dist.png')
        plt.close()
    
    # 2. Rating Distribution by Age Group
    if 'age_group' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='age_group', y='rating', data=df.dropna(subset=['age_group', 'rating']))
        plt.title('Rating Distribution by Age Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'age_rating_dist.png')
        plt.close()
    
    # 3. Rating Distribution by Occupation
    if 'occupation' in df.columns:
        plt.figure(figsize=(15, 6))
        sns.boxplot(x='occupation', y='rating', data=df.dropna(subset=['occupation', 'rating']))
        plt.title('Rating Distribution by Occupation')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'occupation_rating_dist.png')
        plt.close()
    
    # 4. Genre Preference Heatmap by Gender
    if 'gender' in df.columns and 'genres' in df.columns:
        genre_gender = pd.crosstab(df['gender'].fillna('Unknown'), df['genres'].fillna('Unknown'))
        plt.figure(figsize=(15, 8))
        sns.heatmap(genre_gender.T, cmap='YlOrRd', annot=True, fmt='d')
        plt.title('Genre Preferences by Gender')
        plt.tight_layout()
        plt.savefig(output_dir / 'genre_gender_heatmap.png')
        plt.close()

def plot_feedback_analysis(df):
    """Generate plots for feedback loop analysis."""
    print("Generating feedback analysis plots...")
    
    # 1. Rating Evolution Over Time
    daily_ratings = df.groupby(df['timestamp'].dt.date)['rating'].agg(['mean', 'count']).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(daily_ratings['timestamp'], daily_ratings['mean'])
    ax1.set_title('Average Rating Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Rating')
    
    ax2.plot(daily_ratings['timestamp'], daily_ratings['count'])
    ax2.set_title('Number of Ratings Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Ratings')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rating_evolution.png')
    plt.close()
    
    # 2. Movie Popularity Distribution
    movie_counts = df['movie_id'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_counts, bins=50)
    plt.title('Movie Rating Count Distribution')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Movies')
    plt.savefig(output_dir / 'movie_popularity_dist.png')
    plt.close()

def plot_security_analysis(df):
    """Generate plots for security analysis."""
    print("Generating security analysis plots...")
    
    # 1. User Rating Frequency
    user_ratings = df.groupby('user_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_ratings, bins=50)
    plt.title('User Rating Frequency Distribution')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')
    plt.savefig(output_dir / 'user_rating_freq.png')
    plt.close()
    
    # 2. Rating Distribution for Suspicious Movies
    suspicious_movies = df.groupby('movie_id').agg({
        'rating': ['count', 'mean', 'std']
    }).reset_index()
    suspicious_movies.columns = ['movie_id', 'count', 'mean', 'std']
    suspicious_movies = suspicious_movies[
        (suspicious_movies['count'] > 10) & 
        (suspicious_movies['std'] < 0.5)
    ]
    
    if len(suspicious_movies) > 0:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=suspicious_movies, x='mean', y='std', size='count', 
                       sizes=(100, 1000), alpha=0.6)
        plt.title('Rating Patterns of Suspicious Movies')
        plt.xlabel('Mean Rating')
        plt.ylabel('Rating Standard Deviation')
        plt.savefig(output_dir / 'suspicious_movies.png')
        plt.close()
    
    # 3. Rating Time Intervals
    df['rating_interval'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rating_interval'].dropna(), bins=50)
    plt.title('Distribution of Time Intervals Between Ratings')
    plt.xlabel('Seconds Between Ratings')
    plt.ylabel('Frequency')
    plt.savefig(output_dir / 'rating_intervals.png')
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations."""
    print("Loading data...")
    df = load_data()
    
    print("Generating fairness visualizations...")
    plot_fairness_metrics(df)
    
    print("Generating feedback loop visualizations...")
    plot_feedback_analysis(df)
    
    print("Generating security analysis visualizations...")
    plot_security_analysis(df)
    
    print(f"All visualizations have been saved to {output_dir}")

if __name__ == "__main__":
    generate_all_visualizations()
