import numpy as np
from TrainSingleModel import TrainSingleModel
from TrainingData import TrainingData

training_data = TrainingData()
train_ds, val_ds, test_ds = training_data.getTrainingSets()

candidates = [
    1e-5,
    2e-5,
    3e-5,
]

results = {}
for learning_rate in candidates:
    print(f"\n=== Training with learning_rate = {learning_rate} ===")
    candidate_model = TrainSingleModel(model_name="bert-base-uncased", train_ds=train_ds, test_ds=test_ds, val_ds=val_ds)
    candidate_model.set_learning_rate(learning_rate)
    candidate_model.train(is_seed_set=True)
    results[learning_rate] = candidate_model.evaluate()

print(results)
best_model = min(results, key=lambda lr: results[lr])
print(f"\nBest candidate on held-out test set: {best_model}")

