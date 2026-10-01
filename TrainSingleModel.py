from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from datasets import Dataset
from FreezeThenUnfreeze import FreezeThenUnfreeze
from scipy.stats import spearmanr
from transformers import EarlyStoppingCallback

class TrainSingleModel:
    def __init__(self, model_name, train_ds, val_ds, test_ds, load_final = False, max_length=128):
        self.learning_rate = 2e-5
        self.model_name = model_name
        self.max_length = max_length
        self.frozen_epochs = 3
        self.training_epochs = 8

        self.output_dir = Path("out") / model_name.replace("/", "__") # Unique, filesystem-safe output dir per candidate
        self.final_dir = self.output_dir / "final"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False
        )

        self.train_ds = self._prepare_dataset(train_ds)
        self.val_ds = self._prepare_dataset(val_ds)
        self.test_ds = self._prepare_dataset(test_ds)
        
        self.model = None
        self._model_initialisation(load_final)

        self.trainer = self._create_trainer(initial_training=not load_final)

    def _model_initialisation(self, load_final = False):
        if load_final:
            self._load_final_model()
        else:
            # Loads pre-trained encoder and randomly initialises regression head
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=1, problem_type="regression")



    def _prepare_dataset(self, dataframe):
        """
        Convert a pandas DataFrame into a tokenised Hugging Face Dataset.
        """

        required_columns = {"sentence", "labels"}
        missing = required_columns - set(dataframe.columns)
        if missing: raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

        dataset = Dataset.from_pandas(
            dataframe[["sentence", "labels"]],
            preserve_index=False,
        )

        def tokenize(batch):
            return self.tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=self.max_length,
            )

        return dataset.map(
            tokenize,
            batched=True,
        )


    def train(self, is_seed_set = False, seed = 42):
        if is_seed_set:
            set_seed(seed)

        self.trainer.train()


    def evaluate(self):
        if self.trainer is None:
            raise RuntimeError("No Trainer exists. Train or load the model before evaluating.")

        val_metrics = self.trainer.evaluate(
            eval_dataset=self.val_ds, metric_key_prefix="val",
        )
        test_metrics = self.trainer.evaluate(
            eval_dataset=self.test_ds, metric_key_prefix="test",
        )
        return {"val_rmse": val_metrics["val_rmse"], "val_spearman": val_metrics["val_spearman"],
                "test_rmse": test_metrics["test_rmse"], "test_spearman": test_metrics["test_spearman"], }


    def save_final_model(self):
        """
        Save model + tokenizer to:
            output_dir/final/
        """

        if self.trainer is None:
            raise RuntimeError("No Trainer exists. Train or load the model before saving.")

        self.final_dir.mkdir(parents=True, exist_ok=True)

        self.trainer.save_model(str(self.final_dir))
        self.tokenizer.save_pretrained(str(self.final_dir))

        print(f"Saved final model to: {self.final_dir}")

    def _load_final_model(self):
        """
        Load the final saved model.
        """

        if not self.final_dir.exists():
            raise FileNotFoundError(
                f"No final model found at: {self.final_dir}"
            )

        print(f"Loading final model from: {self.final_dir}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.final_dir),
            num_labels=1,
            problem_type="regression",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.final_dir),
            use_fast=False,
        )


    def _create_trainer(self, initial_training=True):
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.training_epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            report_to="none",  # skip wandb/tensorboard prompts
            save_total_limit=1,  # don't keep every epoch's checkpoint on disk
        )
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

        # Only use the freeze/unfreeze callback when training newly initialised model.
        if initial_training:
            callbacks.append(
                FreezeThenUnfreeze(freeze_epochs=self.frozen_epochs)
            )

        def _metrics(p):
            predictions, labels = p.predictions.squeeze(), p.label_ids
            rmse = np.sqrt(np.mean((predictions - labels) ** 2))
            spearman = spearmanr(predictions, labels).statistic

            return {
                "rmse": float(rmse),
                "spearman": float(spearman),
            }


        return Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            processing_class=self.tokenizer,
            compute_metrics=_metrics,
            callbacks=callbacks,
        )

    def set_frozen_epochs(self, frozen_epochs, initial_training = True):
        self.frozen_epochs = frozen_epochs
        self.trainer = self._create_trainer(initial_training=initial_training)

    def set_training_epochs(self, training_epochs, initial_training = True):
        self.training_epochs = training_epochs
        self.trainer = self._create_trainer(initial_training=initial_training)

    def set_learning_rate(self, learning_rate, initial_training = True):
        self.learning_rate = learning_rate
        self.trainer = self._create_trainer(initial_training=initial_training)
