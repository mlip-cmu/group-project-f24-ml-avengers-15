import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pytest
import pandas as pd
import tempfile
from utils import prepare_data_csv, prepare_data_model

# Tests that `prepare_data_csv` correctly splits data into training and validation sets based on split ratio.
def test_prepare_data_csv():
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, 'test_ratings.csv')
        
        # Create sample data
        sample_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'movie_id': [10, 20, 10, 30],
            'rating': [5, 3, 4, 2],
            'user_time': [1234567890, 1234567891, 1234567892, 1234567893]
        })
        
        # Save to temporary file
        sample_data.to_csv(test_file, index=False)
        
        # Run test
        train_df, val_df = prepare_data_csv(test_file, split_ratio=0.5)
        
        assert len(train_df) == 2, "Expected 2 entries in the training set"
        assert len(val_df) == 2, "Expected 2 entries in the validation set"
        
        # File will be automatically cleaned up when the context manager exits

# Tests that `prepare_data_model` loads data correctly into Surprise Dataset format for training and validation.
def test_prepare_data_model():
    train_df = pd.DataFrame({
        'user_id': [1, 1, 2],
        'movie_id': [10, 20, 10],
        'rating': [5, 3, 4]
    })
    val_df = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': [30, 40],
        'rating': [2, 4]
    })
    train_data, valid_data = prepare_data_model(train_df, val_df, rating_range=(1, 5))

    assert train_data is not None, "Expected train_data to be initialized"
    assert valid_data is not None, "Expected valid_data to be initialized"
