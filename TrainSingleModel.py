import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from datasets import Dataset
from FreezeThenUnfreeze import FreezeThenUnfreeze


class TrainSingleModel():
    def __init__(self, model_name, train_ds, val_ds, seed=42):
        self.model_name = model_name
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.seed = seed
        # Unique, filesystem-safe output dir per candidate
        self.output_dir = f"out/{model_name.replace('/', '__')}"

    def train_and_evaluate(self, test_ds):
        set_seed(self.seed)
        tok = AutoTokenizer.from_pretrained(self.model_name)

        def prep(batch):
            return tok(batch["sentence"], truncation=True, max_length=128)

        train_ds = Dataset.from_pandas(self.train_ds[["sentence", "labels"]]).map(prep, batched=True)
        val_ds = Dataset.from_pandas(self.val_ds[["sentence", "labels"]]).map(prep, batched=True)

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
                          eval_dataset=val_ds, processing_class=tok, compute_metrics=metrics,
                          callbacks=[FreezeThenUnfreeze(freeze_epochs=3)])

        trainer.train()

        val_rmse = trainer.evaluate(val_ds)["eval_rmse"]
        test_rmse = trainer.evaluate(test_ds)["eval_rmse"]  # unbiased comparison metric

        return {"val_rmse": val_rmse, "test_rmse": test_rmse}

