import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "SVD_movie_recommender.pkl")
BACKUP_MODEL_PATH = os.path.join(MODEL_DIR, "SVD_movie_recommender_previous.pkl")
DATA_DIR = os.path.join(ROOT_DIR, "data")
