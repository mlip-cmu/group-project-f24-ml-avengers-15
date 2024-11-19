# test_data_qualitycheck.py
import pytest
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from evaluation.data_qualitycheck import check_timestamp_format, validate_user_id_format, check_watch_action, validate_rating_value, validate_recommend_result_structure, verify_action_schema, assess_log_entry

# Timestamp validation tests
@pytest.mark.parametrize("timestamp,expected", [
    ("2023-01-01T12:00:00", True),
    ("2023-01-01T12:00", True),
    ("2023-01-01T12:00:00.123456", True),
    ("2023/01/01T12:00:00", False),
    ("01-01-2023T12:00:00", False),
    ("2023-01-01 12:00:00", False),
])
def test_check_timestamp_format(timestamp, expected):
    assert check_timestamp_format(timestamp) == expected

# User ID validation tests
@pytest.mark.parametrize("user_id,expected", [
    ("12345", True),
    ("0", True),
    ("", False),
    ("abc123", False),
])
def test_validate_user_id_format(user_id, expected):
    assert validate_user_id_format(user_id) == expected

# Watch action validation tests
@pytest.mark.parametrize("action,expected", [
    ("GET /media/view/movie/1234.mpg", True),
    ("GET /media/view/movie 1234.mpg", False),
    ("GET /media/view/movie/1234.mpg extra", False),  # extra content not allowed
])
def test_check_watch_action(action, expected):
    assert check_watch_action(action) == expected

# Rating value validation tests
@pytest.mark.parametrize("action,expected", [
    ("GET /rate_movie/movie=5", True),
    ("GET /rate_movie/movie=0", False),
    ("GET /rate_movie/movie=6", False),
    ("GET /rate_movie/movie=abc", False),  # Invalid rating
])
def test_validate_rating_value(action, expected):
    assert validate_rating_value(action) == expected

# Recommendation result validation tests
@pytest.mark.parametrize("action,expected", [
    ("recommend request movie, status 200, result: " + ", ".join([f"Movie{i}" for i in range(20)]) + ", 210 ms", True),  # Valid
    ("recommend request movie, status 200, result: Movie1, Movie1, 210 ms", False),  # Non-unique movies
    ("recommend request movie, status 200, result: Movie1, Movie2, 200.5 ms", False),  # Invalid processing time
    ("recommend request movie, status 200, result: Movie1, Movie2, 210ms", True),  # Incorrect format without space
])
def test_validate_recommend_result_structure(action, expected):
    assert validate_recommend_result_structure(action) == expected


# Action schema validation tests
@pytest.mark.parametrize("action,expected", [
    ("GET /media/view/movie/1234.mpg", True),
    ("GET /rate_movie/movie=5", True),
    ("recommend request movie, status 200, result: " + ", ".join([f"Movie{i}" for i in range(20)]) + ", 210 ms", True),
    ("GET /unknown/action", False),
])
def test_verify_action_schema(action, expected):
    assert verify_action_schema(action) == expected

# Log entry validation tests
@pytest.mark.parametrize("log_entry,expected", [
    ("2023-01-01T12:00:00,12345,GET /media/view/movie/1234.mpg", True),
    ("2023-01-01T12:00:00,12345,GET /rate_movie/movie=5", True),
    ("2023-01-01T12:00:00,12345,GET /rate_movie/movie=6", False),
    ("invalid_timestamp,12345,GET /media/view/movie/1234.mpg", False),
    ("2023-01-01T12:00:00,abc,GET /media/view/movie/1234.mpg", False),
    ("2023-01-01T12:00:00,12345,recommend request movie, status 200, result: " + ", ".join([f"Movie{i}" for i in range(19)]) + ", 210 ms", True),
])
def test_assess_log_entry(log_entry, expected):
    assert assess_log_entry(log_entry) == expected
