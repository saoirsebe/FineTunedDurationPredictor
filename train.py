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

# Splitting test data:
test_ds, temp_ds = train_test_split(trainingDataTable, test_size=0.2, random_state=42)
val_ds, test_ds = train_test_split(temp_ds, test_size=0.5, random_state=42)

results = {}
for name in candidates:
    print(f"\n=== Training {name} ===")
    candidate_model = TrainSingleModel(model_name=name, train_ds=test_ds, val_ds=val_ds)
    results[name] = candidate_model.train_and_evaluate(test_ds=test_ds)

print(results)
best_model = min(results, key=lambda k: results[k]["test_rmse"])
print(f"\nBest candidate on held-out test set: {best_model}")