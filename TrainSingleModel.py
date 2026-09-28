import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from datasets import Dataset
from FreezeThenUnfreeze import FreezeThenUnfreeze
from scipy.stats import spearmanr

class TrainSingleModel:
    def __init__(self, model_name, train_ds, val_ds, test_ds, seed=42):
        self.model_name = model_name
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.seed = seed
        # Unique, filesystem-safe output dir per candidate
        self.output_dir = f"out/{model_name.replace('/', '__')}"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False
        )
        self.model = None
        self.trainer = None

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

        dataset = dataset.map(
            tokenize,
            batched=True,
        )

        return dataset

    def train_and_evaluate(self):
        set_seed(self.seed)

        train_ds = self._prepare_dataset(self.train_ds)
        val_ds = self._prepare_dataset(self.val_ds)
        test_ds = self._prepare_dataset(self.test_ds)

        # Loads pre-trained encoder and randomly initialises regression head
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1, problem_type="regression")

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

        trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                          eval_dataset=val_ds, processing_class=self.tokenizer, compute_metrics=metrics,
                          callbacks=[FreezeThenUnfreeze(freeze_epochs=3)])

        trainer.train()

        val_rmse = trainer.evaluate(val_ds)["eval_rmse"]
        test_rmse = trainer.evaluate(test_ds)["eval_rmse"]  # unbiased comparison metric


        return {"val_rmse": val_rmse, "test_rmse": test_rmse}

    def train_and_save(self):


        train_ds = self._prepare_dataset(self.train_ds)
        val_ds = self._prepare_dataset(self.val_ds)
        test_ds = self._prepare_dataset(self.test_ds)

        # Loads pre-trained encoder and randomly initialises regression head
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1, problem_type="regression")

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

        trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                          eval_dataset=val_ds, processing_class=self.tokenizer, compute_metrics=metrics,
                          callbacks=[FreezeThenUnfreeze(freeze_epochs=3)])

        trainer.train()

        val_rmse = trainer.evaluate(val_ds)["eval_rmse"]
        test_rmse = trainer.evaluate(test_ds)["eval_rmse"]  # unbiased comparison metric

        print(f"\nValidation eval: {val_rmse}")
        print(f"\nTest eval: {test_rmse}")

        trainer.save_model("final_model")
        self.tokenizer.save_pretrained("final_model")

    def report(self, pred_z, true_minutes, mean, std):
        pred = np.exp(pred_z * std + mean)
        err = pred - true_minutes
        ratio = np.maximum(pred / true_minutes, true_minutes / pred)
        return {
            "MAE_min": np.abs(err).mean(),
            "MedAE_min": np.median(np.abs(err)),
            "RMSE_min": np.sqrt((err ** 2).mean()),
            "MAE_log": np.abs(np.log(pred) - np.log(true_minutes)).mean(),
            "within_1.5x": (ratio <= 1.5).mean(),
            "within_2x": (ratio <= 2).mean(),
            "spearman": spearmanr(pred, true_minutes).correlation,
        }
