import sys
import shutil
import os
import pandas as pd
import logging
from config import MODEL_PATH, BACKUP_MODEL_PATH, MODEL_PATH_2, BACKUP_MODEL_PATH_2
from utils import prepare_data_model, train_model, evaluate, train_model2
from app import log_retraining_provenance 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_csv(file_path="data/extracted_ratings.csv"):
    """
    Loads data from a CSV file into a DataFrame.
    """
    if not os.path.exists(file_path):
        logging.warning(f"The file {file_path} does not exist. No data to load.")
        return pd.DataFrame()

    try:
        logging.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {len(df)} rows from {file_path}.")
        return df
    except Exception as e:
        logging.error(f"An error occurred while reading {file_path}: {str(e)}")
        return pd.DataFrame()
    

def prepare_data_csv(df, split_ratio=0.8):
    """
    Splits the provided DataFrame into training and validation sets based on users.
    """
    logging.info(f"Preparing data with split ratio: {split_ratio}")
    df = df.sort_values(by=['user_id', 'user_time'])

    train_data_list = []
    val_data_list = []

    for user_id, group in df.groupby('user_id'):
        split_index = int(len(group) * split_ratio)
        train_data_list.append(group.iloc[:split_index])
        val_data_list.append(group.iloc[split_index:])

    train_df = pd.concat(train_data_list).reset_index(drop=True)
    val_df = pd.concat(val_data_list).reset_index(drop=True)

    logging.info(f"Training data size: {len(train_df)} rows")
    logging.info(f"Validation data size: {len(val_df)} rows")

    return train_df, val_df

def main():
    try:
        split_ratio = float(os.getenv('SPLIT_RATIO', 0.8))

        recent_data_df = load_data_from_csv(file_path="data/extracted_ratings.csv")


        if recent_data_df.empty:
            logging.warning("No data available for retraining. Exiting.")
            return 1  

        logging.info("Preparing data for training and validation...")
        train_df, val_df = prepare_data_csv(recent_data_df, split_ratio)
        train_data, valid_data = prepare_data_model(train_df, val_df)

        if os.path.exists(MODEL_PATH):
            logging.info(f"Backing up the existing model from {MODEL_PATH} to {BACKUP_MODEL_PATH}...")
            shutil.copyfile(MODEL_PATH, BACKUP_MODEL_PATH)
        
        if os.path.exists(MODEL_PATH_2):
            logging.info(f"Backing up the existing model from {MODEL_PATH_2} to {BACKUP_MODEL_PATH_2}...")
            shutil.copyfile(MODEL_PATH_2, BACKUP_MODEL_PATH_2)

        logging.info("Training the model...")
        model, training_time_ms = train_model(train_data, MODEL_PATH)
        model2, training_time_ms2 = train_model2(train_data, MODEL_PATH_2)

        # log_retraining_provenance(model_id="SVD_movie_recommender.pkl", training_duration=training_time_ms / 1000, training_data_path="data/extracted_ratings.csv")
        # log_retraining_provenance(model_id="SVD_movie_recommender_2.pkl", training_duration=training_time_ms2 / 1000, training_data_path="data/extracted_ratings.csv")

        rmse = evaluate(model, valid_data)
        rmse2 = evaluate(model2, valid_data)
        logging.info(f"Validation RMSE: {rmse:.4f}")
        logging.info(f"Validation RMSE2: {rmse2:.4f}")
        logging.info("Retraining completed successfully.")
        return 0  # Exit code 0 indicates success

    except Exception as e:
        logging.error(f"An error occurred during the retraining process: {str(e)}")
        return 42  # Exit code 42 indicates failure

if __name__ == "__main__":
    sys.exit(main())
