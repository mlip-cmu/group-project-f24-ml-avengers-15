#data_qualitycheck.py

import re
from datetime import datetime

# Function to check timestamp format and validity
def check_timestamp_format(timestamp):
    formats = ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M')
    for fmt in formats:
        try:
            datetime.strptime(timestamp, fmt)
            return True
        except ValueError:
            continue
    return False

# Function to ensure user ID format is numeric and non-empty
def validate_user_id_format(user_id):
    return user_id.isdigit() and user_id.strip() != ""

# New function to validate watch actions with enhanced structure
def check_watch_action(action):
    pattern = re.compile(r"GET /media/view/.+/\d+\.mpg$")
    return pattern.match(action) is not None and not any(" " in segment for segment in action.split('/')[-1:])

# New function to validate rating values are within range
def validate_rating_value(action):
    if "/rate_movie/" in action:
        try:
            rating = int(action.split('=')[-1])
            return 1 <= rating <= 5
        except ValueError:
            return False
    return True

# New function to validate recommendation structure and result format
def validate_recommend_result_structure(action):
    if "recommend request" in action:
        try:
            parts = action.split("result: ")[1]  # Split to get the relevant part
            movie_list, proc_time = parts.rsplit(", ", 1)  # Separate movie list from processing time
            
            # Check if processing time is valid (with a space or without)
            time_valid = re.match(r"^\d+ ms$|^\d+ms$", proc_time.strip()) is not None
            
            # Check for unique movies
            movies = movie_list.split(", ")
            unique_movies = len(set(movies)) == len(movies)
            
            return time_valid and unique_movies
        except (IndexError, ValueError):
            return False
    return False

# Comprehensive function to validate all action types based on format
def verify_action_schema(action):
    if "GET /media/view/" in action:
        return check_watch_action(action)
    elif "/rate_movie/" in action:
        return validate_rating_value(action)
    elif "recommend request" in action:
        return validate_recommend_result_structure(action)
    return False

# Final log validation function that combines all checks
def assess_log_entry(log_entry):
    parts = log_entry.split(',')
    if len(parts) < 3:
        return False

    timestamp, user_id, action = parts[0], parts[1], ','.join(parts[2:])

    # Validating timestamp and user ID format
    if not check_timestamp_format(timestamp) or not validate_user_id_format(user_id):
        return False

    # Validating action format based on action type
    if not verify_action_schema(action):
        return False

    return True