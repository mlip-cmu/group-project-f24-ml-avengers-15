import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "SVD_movie_recommender.pkl")
MODEL_PATH_2 = os.path.join(ROOT_DIR, "models", "SVD_movie_recommender_2.pkl")
DATA_DIR = os.path.join(ROOT_DIR, "data")
