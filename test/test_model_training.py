from utils import train_model, evaluate
from surprise import Dataset, Reader
import pandas as pd

# Tests that `train_model` successfully trains an SVD model and returns a valid model with positive training time.
def test_train_model():
    df = pd.DataFrame({
        'user_id': [1, 1, 2],
        'movie_id': [10, 20, 10],
        'rating': [5, 3, 4]
    })
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
    model, training_time_ms = train_model(data, model_name="TestSVD")
    
    assert model is not None, "Expected a trained model to be returned"
    assert training_time_ms > 0, "Expected positive training time in ms"

# Tests that `evaluate` calculates RMSE for the model on a small dataset and returns a positive RMSE value.
def test_evaluate():
    df = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': [10, 20],
        'rating': [5, 3]
    })
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
    model, _ = train_model(data, model_name="TestSVD")
    rmse = evaluate(model, data)
    
    assert rmse > 0, "Expected RMSE to be greater than zero"
