import os
import shutil
from config import MODEL_PATH, BACKUP_MODEL_PATH

def rollback_model():
    """
    Restores the previous model from the backup.
    """
    if os.path.exists(BACKUP_MODEL_PATH):
        print(f"Restoring previous model from {BACKUP_MODEL_PATH} to {MODEL_PATH}...")
        shutil.copyfile(BACKUP_MODEL_PATH, MODEL_PATH)
        print("Rollback completed successfully.")
    else:
        print("No backup model found. Rollback failed.")

if __name__ == "__main__":
    rollback_model()
