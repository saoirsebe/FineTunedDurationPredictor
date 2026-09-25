import numpy as np
import pandas as pd
import json


from sklearn.model_selection import train_test_split

from TrainSingleModel import TrainSingleModel

candidates = [
    "sentence-transformers/all-MiniLM-L6-v2",   # 22M, current baseline
    "distilbert-base-uncased",                   # 66M, general-purpose
    "bert-base-uncased",                          # 110M, general-purpose
    "microsoft/deberta-v3-small",                 # 44M, strong for its size
]

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

# Tokenising:
train_df, val_df = train_test_split(trainingDataTable, test_size=0.2, random_state=42)


results = {}
for name in candidates:
    candidate_model = TrainSingleModel(model_name=name, train_df=train_df, val_df=val_df)
    results[name] = candidate_model.trainModel()

print(results)