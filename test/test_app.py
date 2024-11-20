import pytest
import json
from unittest.mock import patch, MagicMock
from flask import Response
import sys
import os

# Add the app directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set testing environment variable
os.environ["TESTING"] = "true"

from app import app, recommend_movies

# Client fixture for testing Flask routes
@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

# Test recommend_movies with a valid user ID
@patch('app.utils.predict')
def test_recommend_movies_with_valid_user_id(mock_predict):
    mock_predict.return_value = ['Movie1', 'Movie2', 'Movie3']
    with app.app_context():
        response = recommend_movies(999)
        assert isinstance(response, str)
        assert response == 'Movie1,Movie2,Movie3'

# Test recommend_movies with an invalid user ID
@patch('app.utils.predict')
def test_recommend_movies_with_invalid_user_id(mock_predict):
    mock_predict.return_value = []
    with app.app_context():
        response = recommend_movies(-1)
        assert isinstance(response, str)
        assert response == ''

# Test recommend_movies with a large user ID
@patch('app.utils.predict')
def test_recommend_movies_with_large_user_id(mock_predict):
    mock_predict.return_value = ['Movie1', 'Movie2', 'Movie3']
    with app.app_context():
        response = recommend_movies(10**8)
        assert isinstance(response, str)
        assert response == 'Movie1,Movie2,Movie3'

# Test recommend_movies when predict raises an exception
@patch('app.utils.predict')
def test_recommend_movies_exception_handling(mock_predict):
    mock_predict.side_effect = Exception("Prediction error")
    with app.app_context():
        response = recommend_movies(999)
        assert isinstance(response, list)
        assert response == []

# Test `/recommend/<user_id>` route with a valid user ID
@patch('app.recommend_movies')
def test_recommend_route_valid_user(mock_recommend_movies, client):
    mock_recommend_movies.return_value = 'Movie1,Movie2,Movie3'
    response = client.get('/recommend/123')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data == 'Movie1,Movie2,Movie3'

# Test movie ID format handling
@patch('app.utils.predict')
def test_movie_id_format_handling(mock_predict):
    test_movies = [
        'Kill Bill Vol. 1 2003',
        'The Matrix 1999',
        'Pulp Fiction 1994'
    ]
    mock_predict.return_value = test_movies
    with app.app_context():
        response = recommend_movies(999)
        assert isinstance(response, str)
        expected = 'Kill Bill Vol. 1 2003,The Matrix 1999,Pulp Fiction 1994'
        assert response == expected

# Test error response format
@patch('app.utils.predict')
def test_error_response_format(mock_predict):
    mock_predict.side_effect = Exception("Movie not found")
    with app.app_context():
        response = recommend_movies(999)
        assert isinstance(response, list)
        assert response == []

# Test `/recommend/<user_id>` route with an exception
@patch('app.recommend_movies')
def test_recommend_route_exception_handling(mock_recommend_movies, client):
    mock_recommend_movies.side_effect = Exception("Recommendation error")
    response = client.get('/recommend/999')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data == []

# Test `/recommend/<user_id>` route with a string user ID
def test_recommend_route_string_user_id(client):
    response = client.get('/recommend/notanumber')
    assert response.status_code == 404

# Test `/recommend/<user_id>` route with a float user ID
def test_recommend_route_float_user_id(client):
    response = client.get('/recommend/123.45')
    assert response.status_code == 404

# Test `/recommend/<user_id>` route with missing user ID
def test_recommend_route_missing_user_id(client):
    response = client.get('/recommend/')
    assert response.status_code == 404
