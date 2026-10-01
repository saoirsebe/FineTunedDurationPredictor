import numpy as np
from TrainSingleModel import TrainSingleModel
from TrainingData import TrainingData

training_data = TrainingData()
train_ds, val_ds, test_ds = training_data.getTrainingSets()

candidate_model = TrainSingleModel(model_name="bert-base-uncased", train_ds=train_ds, test_ds=test_ds, val_ds=val_ds)
candidate_model.train(is_seed_set=True)
results = candidate_model.evaluate()

