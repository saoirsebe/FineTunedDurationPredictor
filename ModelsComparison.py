from TrainingData import TrainingData
from TrainSingleModel import TrainSingleModel

candidates = [
    "microsoft/deberta-v3-small",               # 44M, strong for its size
    "sentence-transformers/all-MiniLM-L6-v2",   # 22M, current baseline
    "distilbert-base-uncased",                   # 66M, general-purpose
    "bert-base-uncased",                         # 110M, general-purpose
]

training_data = TrainingData()
train_ds, val_ds, test_ds = training_data.getTrainingSets()

results = {}
for name in candidates:
    print(f"\n=== Training {name} ===")
    candidate_model = TrainSingleModel(model_name=name, train_ds=train_ds, test_ds=test_ds, val_ds=val_ds)
    candidate_model.train(is_seed_set=True)
    results[name] = candidate_model.evaluate()


print(results)
best_model = min(results, key=lambda k: results[k]["test_rmse"])
print(f"\nBest candidate on held-out test set: {best_model}")

"""
{'microsoft/deberta-v3-small': {'val_rmse': 0.6223149299621582, 'val_spearman': 0.7454823043831983, 'test_rmse': 0.6186169385910034, 'test_spearman': 0.7248710046888187}, 
'sentence-transformers/all-MiniLM-L6-v2': {'val_rmse': 0.6213993430137634, 'val_spearman': 0.7360901255417569, 'test_rmse': 0.6133740544319153, 'test_spearman': 0.7247540804594248},
'distilbert-base-uncased': {'val_rmse': 0.6152892708778381, 'val_spearman': 0.7335545824871792, 'test_rmse': 0.5942729115486145, 'test_spearman': 0.7333104906953662}, 
'bert-base-uncased': {'val_rmse': 0.6057919859886169, 'val_spearman': 0.7427528056035912, 'test_rmse': 0.5770533680915833, 'test_spearman': 0.7486673127489775}}

Best candidate on held-out test set: bert-base-uncased
"""
