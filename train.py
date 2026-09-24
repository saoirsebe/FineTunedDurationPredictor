import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split

name = "sentence-transformers/all-MiniLM-L6-v2"

with open("MS-LaTTE_split.json", encoding="utf-8") as f:
    records = json.load(f)

trainingDataTable = pd.DataFrame({
    "sentence": [r["TaskTitle"] for r in records],
    "time": [r["TimeTaken"][0]["EstimatedMinutes"] for r in records],
})

y = np.log(trainingDataTable["time"].values)
mean, std = float(y.mean()), float(y.std())
trainingDataTable["labels"] = ((y - mean) / std).astype("float32")
json.dump({"mean": mean, "std": std}, open("target_stats.json", "w"))

train_df, val_df = train_test_split(trainingDataTable, test_size=0.2, random_state=42)