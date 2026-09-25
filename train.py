import numpy as np
import pandas as pd
import json

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "sentence-transformers/all-MiniLM-L6-v2"

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

tok = AutoTokenizer.from_pretrained(name)

def prep(batch):
    return tok(batch["sentence"], truncation=True, max_length=128)

train_ds = Dataset.from_pandas(train_df[["sentence", "labels"]]).map(prep, batched=True)
val_ds = Dataset.from_pandas(val_df[["sentence", "labels"]]).map(prep, batched=True)


# Loads pre-trained encoder and randomly initialises regression head
model = AutoModelForSequenceClassification.from_pretrained(
    name, num_labels=1, problem_type="regression")

def metrics(p):
    predictions, labels = p.predictions.squeeze(), p.label_ids
    return {"rmse": float(np.sqrt(((predictions - labels) ** 2).mean()))}