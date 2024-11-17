import shutil
import os
import pandas as pd
import logging
from config import MODEL_PATH, BACKUP_MODEL_PATH
from utils import prepare_data_model, train_model, evaluate
from consume_kafka_logs import load_recent_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data_csv(df, split_ratio=0.8):
    """
    Splits the provided DataFrame into training and validation sets based on users.
    """
    df = df.sort_values(by=['user_id', 'user_time'])

    train_data_list = []
    val_data_list = []

    for user_id, group in df.groupby('user_id'):
        split_index = int(len(group) * split_ratio)
        train_data_list.append(group.iloc[:split_index])
        val_data_list.append(group.iloc[split_index:])

    train_df = pd.concat(train_data_list).reset_index(drop=True)
    val_df = pd.concat(val_data_list).reset_index(drop=True)

    return train_df, val_df

def main():
    try:
        logging.info("Loading recent data...")
        recent_data_df = load_recent_data(days=3)

        if recent_data_df.empty:
            logging.warning("No data available for retraining. Exiting.")
            return

        # Prepare training and validation data
        logging.info("Preparing data for training and validation...")
        train_df, val_df = prepare_data_csv(recent_data_df)
        train_data, valid_data = prepare_data_model(train_df, val_df)

        # Backup the existing model if it exists
        if os.path.exists(MODEL_PATH):
            logging.info(f"Backing up the existing model from {MODEL_PATH} to {BACKUP_MODEL_PATH}...")
            shutil.copyfile(MODEL_PATH, BACKUP_MODEL_PATH)

        # Train the model
        logging.info("Training the model...")
        model, training_time_ms = train_model(train_data, MODEL_PATH)

        # Evaluate the model
        logging.info("Evaluating the model...")
        rmse = evaluate(model, valid_data)
        logging.info(f"Validation RMSE: {rmse:.4f}")
        logging.info("Retraining completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during the retraining process: {str(e)}")

if __name__ == "__main__":
    main()
