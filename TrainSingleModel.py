from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from datasets import Dataset
from FreezeThenUnfreeze import FreezeThenUnfreeze
from scipy.stats import spearmanr

class TrainSingleModel:
    def __init__(self, model_name, train_ds, val_ds, test_ds, load_final = False):
        self.model_name = model_name
        self.train_ds = self._prepare_dataset(train_ds)
        self.val_ds =  self._prepare_dataset(val_ds)
        self.test_ds = self._prepare_dataset(test_ds)

        self.output_dir = Path("out") / model_name.replace("/", "__") # Unique, filesystem-safe output dir per candidate
        self.final_dir = self.output_dir / "final"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False
        )
        self.model = None
        self._model_initialisation(load_final)

        self.trainer = None

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

        dataset = Dataset.from_pandas(
            dataframe[["sentence", "labels"]],
            preserve_index=False,
        )

        def tokenize(batch):
            return self.tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=128,
            )

        return dataset.map(
            tokenize,
            batched=True,
        )


    def train(self, is_seed_set = False, seed = 42):
        if is_seed_set:
            set_seed(seed)

        def metrics(p):
            predictions, labels = p.predictions.squeeze(), p.label_ids
            return {"rmse": float(np.sqrt(((predictions - labels) ** 2).mean()))}

        args = TrainingArguments(
            output_dir=self.output_dir, num_train_epochs=8, learning_rate=2e-5,
            per_device_train_batch_size=16, eval_strategy="epoch",
            save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="rmse", greater_is_better=False,
            report_to="none",  # skip wandb/tensorboard prompts
            save_total_limit=1,  # don't keep every epoch's checkpoint on disk
        )

        self.trainer = Trainer(model=self.model, args=args, train_dataset=self.train_ds,
                          eval_dataset=self.val_ds, processing_class=self.tokenizer, compute_metrics=metrics,
                          callbacks=[FreezeThenUnfreeze(freeze_epochs=3)])

        self.trainer.train()


    def evaluate(self):
        val_rmse = self.trainer.evaluate(self.val_ds)["eval_rmse"]
        test_rmse = self.trainer.evaluate(self.test_ds)["eval_rmse"]  # unbiased comparison metric

        return {"val_rmse": val_rmse, "test_rmse": test_rmse}


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

