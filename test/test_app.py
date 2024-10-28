import pytest
from unittest.mock import patch
import app

# Tests recommend_movies with a valid user ID, expecting a comma-separated list of recommended movies.
@patch('app.utils.predict')
def test_recommend_movies_with_valid_user_id(mock_predict):
    mock_predict.return_value = ['Movie1', 'Movie2', 'Movie3']
    recommendations = app.recommend_movies(999)
    assert recommendations == 'Movie1,Movie2,Movie3'

# Tests recommend_movies with an invalid user ID, expecting an empty response string.
@patch('app.utils.predict')
def test_recommend_movies_with_invalid_user_id(mock_predict):
    mock_predict.return_value = []
    recommendations = app.recommend_movies(-1)
    assert recommendations == '', "Expected empty string for invalid user ID"

# Tests recommend_movies with a very large user ID, expecting a comma-separated list of recommended movies.
@patch('app.utils.predict')
def test_recommend_movies_with_large_user_id(mock_predict):
    mock_predict.return_value = ['Movie1', 'Movie2', 'Movie3']
    recommendations = app.recommend_movies(10**8)
    assert recommendations == 'Movie1,Movie2,Movie3'

# Tests recommend_movies when predict raises an exception, expecting an empty list as a response.
@patch('app.utils.predict')
def test_recommend_movies_exception_handling(mock_predict):
    mock_predict.side_effect = Exception("Prediction error")
    recommendations = app.recommend_movies(999)
    assert recommendations == [], "Expected empty string for exception handling"

# Tests recommend_movies with a None user ID, expecting an empty response string.
@patch('app.utils.predict')
def test_recommend_movies_with_none_user_id(mock_predict):
    mock_predict.return_value = []
    recommendations = app.recommend_movies(None)
    assert recommendations == '', "Expected empty string for None user ID"