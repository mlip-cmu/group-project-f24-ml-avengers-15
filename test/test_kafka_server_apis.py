import pytest
from unittest.mock import patch, MagicMock
from kafka_server_apis import KafkaServerApi

# Fixture for initializing KafkaServerApi with mocked environment variables and SSH tunnel.
@pytest.fixture
@patch('subprocess.Popen')
@patch('os.getenv')
def kafka_api_fixture(mock_getenv, mock_popen):
    mock_getenv.side_effect = lambda key, default=None: {
        'SERVER_IP': '127.0.0.1',
        'SSH_USER': 'test_user',
        'SSH_PASSWORD': 'test_password',
        'KAFKA_PORT': '9092',
        'LOCAL_PORT': '9092'
    }.get(key, default)
    mock_ssh_tunnel = MagicMock()
    mock_popen.return_value = mock_ssh_tunnel
    kafka_api = KafkaServerApi()  
    kafka_api.mock_ssh_tunnel = mock_ssh_tunnel 
    return kafka_api

# Tests get_movie_info by verifying that it returns the correct movie information from the mock API.
def test_get_movie_info(kafka_api_fixture):
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'id': 'dummy_movie_2000',
            'tmdb_id': 99999,
            'imdb_id': 'tt999999',
            'title': 'Dummy Movie',
            'original_title': 'Dummy Original Title',
            'adult': 'False',
            'belongs_to_collection': {},
            'budget': '10000000',
            'genres': [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}],
            'homepage': 'http://dummyhomepage.com',
            'original_language': 'en',
            'overview': 'This is a dummy overview of the movie.',
            'popularity': '10.12345',
            'poster_path': '/dummy_poster_path.jpg',
            'production_companies': [{'name': 'Dummy Productions', 'id': 12345}],
            'production_countries': [{'iso_3166_1': 'US', 'name': 'United States of Dummy'}],
            'release_date': '2000-01-01',
            'revenue': '100000000',
            'runtime': 120,
            'spoken_languages': [{'iso_639_1': 'en', 'name': 'English'}],
            'status': 'Released',
            'vote_average': '8.5',
            'vote_count': '5000'
        }
        mock_get.return_value = mock_response
        movie_info = kafka_api_fixture.get_movie_info('dummy_movie')
        expected_response = {
            'id': 'dummy_movie_2000',
            'tmdb_id': 99999,
            'imdb_id': 'tt999999',
            'title': 'Dummy Movie',
            'original_title': 'Dummy Original Title',
            'adult': 'False',
            'belongs_to_collection': {},
            'budget': '10000000',
            'genres': [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}],
            'homepage': 'http://dummyhomepage.com',
            'original_language': 'en',
            'overview': 'This is a dummy overview of the movie.',
            'popularity': '10.12345',
            'poster_path': '/dummy_poster_path.jpg',
            'production_companies': [{'name': 'Dummy Productions', 'id': 12345}],
            'production_countries': [{'iso_3166_1': 'US', 'name': 'United States of Dummy'}],
            'release_date': '2000-01-01',
            'revenue': '100000000',
            'runtime': 120,
            'spoken_languages': [{'iso_639_1': 'en', 'name': 'English'}],
            'status': 'Released',
            'vote_average': '8.5',
            'vote_count': '5000'
        }
        assert movie_info == expected_response

# Tests get_user_info by verifying that it returns the correct user information from the mock API.
def test_get_user_info(kafka_api_fixture):
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'user_id': 99999,
            'age': 99,
            'occupation': 'tester',
            'gender': 'M'
        }
        mock_get.return_value = mock_response
        user_info = kafka_api_fixture.get_user_info(99999)
        expected_response = {
            'user_id': 99999,
            'age': 99,
            'occupation': 'tester',
            'gender': 'M'
        }
        assert user_info == expected_response

# Tests SSH tunnel setup by verifying the expected SSH command is called with correct environment values.
def test_ssh_tunnel_setup():
    with patch('subprocess.Popen') as mock_popen, patch('os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            'SERVER_IP': '127.0.0.1',
            'SSH_USER': 'test_user',
            'SSH_PASSWORD': 'test_password',
            'KAFKA_PORT': '9092',
            'LOCAL_PORT': '9092'
        }.get(key, default)
        KafkaServerApi()
        expected_command = [
            'ssh', '-L', '9092:localhost:9092',
            'test_user@127.0.0.1', '-NT'
        ]
        mock_popen.assert_called_once_with(expected_command)

# Tests SSH tunnel termination by verifying that the terminate method is called on SSH tunnel close.
def test_ssh_tunnel_close(kafka_api_fixture):
    kafka_api_fixture.close()
    kafka_api_fixture.mock_ssh_tunnel.terminate.assert_called_once()