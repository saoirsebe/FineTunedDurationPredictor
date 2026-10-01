import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class TrainingData:
    def __init__(self):
        self.dataset = "MS-LaTTE_split.json"

    def getTrainingSets(self):
        with open(self.dataset, encoding="utf-8") as f:
            records = json.load(f)

        trainingDataTable = pd.DataFrame({
            "sentence": [r["TaskTitle"] for r in records],
            "time": [r["TimeTaken"][0]["EstimatedMinutes"] for r in records],
        })

        # Normalise labels:
        y = np.log(trainingDataTable["time"].values)
        mean, std = float(y.mean()), float(y.std())
        trainingDataTable["labels"] = ((y - mean) / std).astype("float32")
        json.dump({"mean": mean, "std": std}, open("target_stats.json", "w"))

        # Splitting test data:
        train_ds, temp_ds = train_test_split(trainingDataTable, test_size=0.2, random_state=42)
        val_ds, test_ds = train_test_split(temp_ds, test_size=0.5, random_state=42)

        return train_ds, val_ds, test_ds