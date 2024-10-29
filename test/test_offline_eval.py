import pytest
import pandas as pd
import numpy as np
from io import StringIO

# Fixture for sample data
@pytest.fixture
def sample_df():
    sample_data = StringIO("""user_id,age,occupation,gender,movie_id,rating
        1,25,student,M,movie1,4
        2,30,engineer,F,movie2,3
        1,25,student,M,movie3,5
        3,40,teacher,F,movie1,2
    """)
    return pd.read_csv(sample_data)

def test_data_loading(sample_df):
    assert isinstance(sample_df, pd.DataFrame)
    assert len(sample_df) == 4
    assert list(sample_df.columns) == ['user_id', 'age', 'occupation', 'gender', 'movie_id', 'rating']

def test_data_types(sample_df):
    assert sample_df['user_id'].dtype == np.int64
    assert sample_df['age'].dtype == np.int64
    assert sample_df['occupation'].dtype == object
    assert sample_df['gender'].dtype == object
    assert sample_df['movie_id'].dtype == object
    assert sample_df['rating'].dtype == np.int64

def test_unique_users(sample_df):
    unique_users = sample_df['user_id'].nunique()
    assert unique_users == 3

def test_unique_movies(sample_df):
    unique_movies = sample_df['movie_id'].nunique()
    assert unique_movies == 3

def test_rating_range(sample_df):
    min_rating = sample_df['rating'].min()
    max_rating = sample_df['rating'].max()
    assert min_rating >= 1
    assert max_rating <= 5

def test_gender_values(sample_df):
    valid_genders = ['M', 'F']
    assert sample_df['gender'].isin(valid_genders).all()

def test_age_range(sample_df):
    min_age = sample_df['age'].min()
    max_age = sample_df['age'].max()
    assert min_age >= 0
    assert max_age < 100

def test_group_by_user(sample_df):
    user_ratings = sample_df.groupby('user_id')['rating'].mean()
    assert len(user_ratings) == 3
    assert user_ratings[1] == pytest.approx(4.5, rel=1e-1)

def test_group_by_movie(sample_df):
    movie_ratings = sample_df.groupby('movie_id')['rating'].mean()
    assert len(movie_ratings) == 3
    assert movie_ratings['movie1'] == pytest.approx(3.0, rel=1e-1)

def test_occupation_distribution(sample_df):
    occupation_counts = sample_df['occupation'].value_counts()
    assert len(occupation_counts) == 3
    assert occupation_counts['student'] == 2
