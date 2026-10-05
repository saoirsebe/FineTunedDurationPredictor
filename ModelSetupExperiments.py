import numpy as np
from TrainSingleModel import TrainSingleModel
from TrainingData import TrainingData

training_data = TrainingData()
train_ds, val_ds, test_ds = training_data.getTrainingSets()

def best_learning_rate():
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
        best_model = min(results, key=lambda lr: results[lr]["test_rmse"])
        print(f"\nBest candidate on held-out test set: {best_model}")

        """
        {1e-05: {'val_rmse': 0.6203073859214783, 'val_spearman': 0.7280660716531929, 'test_rmse': 0.6129282116889954, 'test_spearman': 0.7163101337264025}, 
        2e-05: {'val_rmse': 0.5994768142700195, 'val_spearman': 0.7439444829792587, 'test_rmse': 0.5779285430908203, 'test_spearman': 0.7476010620608693}, 
        3e-05: {'val_rmse': 0.5961213111877441, 'val_spearman': 0.7487205218932426, 'test_rmse': 0.5832621455192566, 'test_spearman': 0.7370050542143901}}
        """


def warmup_ratio():
    print(f"\n=== Training with warmup_ratio = 0.1 ===")
    candidate_model = TrainSingleModel(model_name="bert-base-uncased", train_ds=train_ds, test_ds=test_ds,val_ds=val_ds)
    candidate_model.train(is_seed_set=True)
    results = candidate_model.evaluate()
    print(results)

    """
    Without warmup_ratio: 2e-05: {'val_rmse': 0.5994768142700195, 'val_spearman': 0.7439444829792587, 'test_rmse': 0.5779285430908203, 'test_spearman': 0.7476010620608693} 
    With: 
    """

warmup_ratio()

def weight_decay():
    print(f"\n=== Training with weight_decay = 0.01 ===")
    candidate_model = TrainSingleModel(model_name="bert-base-uncased", train_ds=train_ds, test_ds=test_ds, val_ds=val_ds)
    candidate_model.train(is_seed_set=True)
    results = candidate_model.evaluate()
    print(results)