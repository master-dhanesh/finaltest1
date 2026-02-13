import csv
import os
from datetime import datetime

LOG_PATH = "/app/prediction_logs.csv"


def log_prediction(filename: str, label: str, confidence: float):
    exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "filename", "label", "confidence"])
        writer.writerow([datetime.utcnow().isoformat(), filename, label, confidence])
