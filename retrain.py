import argparse
import pickle
import os
import shutil

from config import MODEL_PATH, BACKUP_MODEL_PATH

from utils import (
    prepare_data_csv,
    prepare_data_model,
    train_model,
    evaluate
)

def main(args):
    train_df, val_df = prepare_data_csv(args.input_data)
    train_data, valid_data = prepare_data_model(train_df, val_df)

    if os.path.exists(MODEL_PATH):
        shutil.copyfile(MODEL_PATH, BACKUP_MODEL_PATH)

    model, training_time_ms = train_model(train_data, MODEL_PATH)
    rmse = evaluate(model, valid_data)
    print(f"Validation RMSE: {rmse:.4f}")
    print("Retraining completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain recommendation model.")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input data CSV file.")
    args = parser.parse_args()
    main(args)