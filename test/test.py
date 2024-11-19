import pytest
import pandas as pd
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from evaluation.online_evaluation import calculate_precision_at_k, get_user_relevant_movies

# Sample data for testing
sample_data = {
    'user_time': ['2024-10-11T15:07:38', '2024-10-11T15:07:39', '2024-10-11T15:07:40', '2024-10-11T15:07:41'],
    'user_id': [1, 1, 2, 2],
    'movie_id': ['movie1', 'movie2', 'movie1', 'movie3'],
    'movie_title': ['Movie One', 'Movie Two', 'Movie One', 'Movie Three'],
    'year': [2000, 2001, 2000, 2002],
    'rating': [5, 4, 3, 5]
}

test_df = pd.DataFrame(sample_data)

# Test get_user_relevant_movies
def test_get_user_relevant_movies():
    expected_output = {1: {'movie1', 'movie2'}, 2: {'movie3'}}
    relevant_movies = get_user_relevant_movies(test_df, rating_threshold=4)
    assert relevant_movies == expected_output

# Test calculate_precision_at_k
def test_calculate_precision_at_k():
    # Sample data for user recommendations and relevant movies
    user_recommendations = {1: ['movie1', 'movie2', 'movie4'], 2: ['movie3', 'movie5', 'movie6']}
    user_relevant_movies = {1: {'movie1', 'movie2'}, 2: {'movie3'}}
    
    # Calculate precision
    precision = calculate_precision_at_k(user_recommendations, user_relevant_movies, k=10)
    
    # Expected precision: (2/10 + 1/10) / 2 = 0.15
    assert precision == pytest.approx(0.15, rel=1e-2)  
