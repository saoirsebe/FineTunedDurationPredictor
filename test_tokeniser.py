import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split

from TrainSingleModel import TrainSingleModel

model_name = "microsoft/deberta-v3-small"
with open("MS-LaTTE_split.json", encoding="utf-8") as f:
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

candidate_model = TrainSingleModel(model_name=model_name, train_ds=train_ds, test_ds=test_ds, val_ds=val_ds)
candidate_model.train(is_seed_set=True)