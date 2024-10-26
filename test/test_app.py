import pytest
from unittest.mock import patch, MagicMock
import app 

@patch('app.utils.predict')  # Mock the predict method from utils
def test_recommend_movies_success(mock_predict):
    # Mock the predict function to return dummy recommendations
    mock_predict.return_value = ['Movie1', 'Movie2', 'Movie3']

    # Call the recommend_movies function with a mock user ID
    recommendations = app.recommend_movies(999)

    # Assert that the recommendations are correctly joined into a string
    assert recommendations == 'Movie1,Movie2,Movie3'

@patch('app.utils.predict')  # Mock the predict method
def test_recommend_movies_exception(mock_predict):
    # Mock the predict function to raise an exception
    mock_predict.side_effect = Exception("Prediction error")

    # Call the recommend_movies function with a mock user ID
    recommendations = app.recommend_movies(999)

    # Assert that an empty list is returned due to the exception
    assert recommendations == []
