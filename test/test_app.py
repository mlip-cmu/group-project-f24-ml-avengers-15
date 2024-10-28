import pytest
import json
from unittest.mock import patch
import app


@pytest.fixture
def client():
    app.app.testing = True
    return app.app.test_client()

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

# Test the `/recommend/<user_id>` route with a valid user ID, expecting a list of recommendations.
@patch('app.recommend_movies')
def test_recommend_route_valid_user(mock_recommend_movies, client):
    mock_recommend_movies.return_value = ['Movie1', 'Movie2', 'Movie3']
    
    response = client.get('/recommend/123')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data == ['Movie1', 'Movie2', 'Movie3'], "Expected list of recommendations for valid user ID"

# Test the `/recommend/<user_id>` route where recommend_movies raises an exception, expecting an empty list.
@patch('app.recommend_movies')
def test_recommend_route_exception_handling(mock_recommend_movies, client):
    mock_recommend_movies.side_effect = Exception("Recommendation error")

    response = client.get('/recommend/999')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data == [], "Expected empty list when an exception occurs in recommend_movies"

# Test that a non-integer user ID (string) results in a 404 response
def test_recommend_route_string_user_id(client):
    response = client.get('/recommend/notanumber')
    assert response.status_code == 404, "Expected 404 for non-integer user ID"

# Test that a float user ID results in a 404 response
def test_recommend_route_float_user_id(client):
    response = client.get('/recommend/123.45')
    assert response.status_code == 404, "Expected 404 for float user ID"

# Test that a None (missing parameter) user ID results in a 404 response
def test_recommend_route_missing_user_id(client):
    response = client.get('/recommend/')
    assert response.status_code == 404, "Expected 404 for missing user ID"
