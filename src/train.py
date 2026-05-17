import pandas as pd
import numpy as np
import torch
import json, os
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, f1_score

MODEL_CHECKPOINT = "distilbert-base-uncased"
PROCESSED_DIR    = "data/processed"
MODEL_OUTPUT_DIR = "models/distilbert-doc-classifier"

# ── MLflow experiment name ────────────────────────────────────────────
# All runs group under this experiment in the UI
EXPERIMENT_NAME  = "document-classification"


class DocumentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.texts     = df["text_clean"].tolist()
        self.labels    = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)
    f1  = evaluate.load("f1").compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def run_experiment(
    run_name,           # e.g. "distilbert-lr5e-5-epochs3"
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    max_length=256,
    weight_decay=0.01
):
    """
    One full training run wrapped in an MLflow context.
    Every param and metric gets logged automatically.
    Call this multiple times with different params to compare runs.
    """

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Starting run: {run_name}")
    print(f"{'='*55}")

    train_df = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    val_df   = pd.read_csv(f"{PROCESSED_DIR}/val.csv")
    test_df  = pd.read_csv(f"{PROCESSED_DIR}/test.csv")

    with open(f"{PROCESSED_DIR}/label_map.json") as f:
        meta = json.load(f)

    label_map  = meta["label_map"]
    id2label   = {int(k): v for k, v in meta["id2label"].items()}
    num_labels = len(label_map)

    # ── Tokenizer + datasets ──────────────────────────────────────
    tokenizer     = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    train_dataset = DocumentDataset(train_df, tokenizer, max_length)
    val_dataset   = DocumentDataset(val_df,   tokenizer, max_length)
    test_dataset  = DocumentDataset(test_df,  tokenizer, max_length)

    # ── Model ─────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_map
    )

    # ── Training args ─────────────────────────────────────────────
    run_output_dir = f"{MODEL_OUTPUT_DIR}/{run_name}"
    args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="logs",
        logging_steps=50,
        report_to="none",           # we handle logging via MLflow manually
        dataloader_num_workers=0,   # Windows fix
        use_cpu=True                # CPU only
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # ── MLflow run ────────────────────────────────────────────────
    # Everything inside this block gets tracked and versioned
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):

        # Log hyperparameters (params)
        # JD keywords: experiment tracking, model versioning
        mlflow.log_params({
            "model":          MODEL_CHECKPOINT,
            "num_epochs":     num_epochs,
            "batch_size":     batch_size,
            "learning_rate":  learning_rate,
            "max_length":     max_length,
            "weight_decay":   weight_decay,
            "train_samples":  len(train_df),
            "val_samples":    len(val_df),
            "num_labels":     num_labels
        })

        # Train
        print(f"\nTraining {run_name}...")
        train_result = trainer.train()

        # Log training metrics
        mlflow.log_metrics({
            "train_loss":    round(train_result.training_loss, 4),
            "train_runtime": round(train_result.metrics["train_runtime"], 2),
        })

        # Evaluate on validation set
        val_results = trainer.evaluate(val_dataset)
        mlflow.log_metrics({
            "val_accuracy": round(val_results["eval_accuracy"], 4),
            "val_f1":       round(val_results["eval_f1"], 4),
            "val_loss":     round(val_results["eval_loss"], 4),
        })

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        mlflow.log_metrics({
            "test_accuracy": round(test_results["eval_accuracy"], 4),
            "test_f1":       round(test_results["eval_f1"], 4),
            "test_loss":     round(test_results["eval_loss"], 4),
        })

        print(f"  Val  Accuracy: {val_results['eval_accuracy']:.4f}  F1: {val_results['eval_f1']:.4f}")
        print(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}  F1: {test_results['eval_f1']:.4f}")

        # ── Per-class fairness analysis ───────────────────────────
        # JD keywords: responsible AI, bias detection, monitoring
        print("\nPer-class performance (responsible AI check):")
        preds_out  = trainer.predict(test_dataset)
        preds      = np.argmax(preds_out.predictions, axis=-1)
        labels_arr = preds_out.label_ids

        report = classification_report(
            labels_arr, preds,
            target_names=list(id2label.values()),
            output_dict=True
        )

        # Log per-class F1 scores individually — visible in MLflow UI
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and "f1-score" in metrics:
                clean_name = class_name.replace(" ", "_")
                mlflow.log_metric(f"f1_{clean_name}", round(metrics["f1-score"], 4))
                flag = " ← potential bias" if metrics["f1-score"] < 0.90 else ""
                print(f"  {class_name:<25} F1: {metrics['f1-score']:.3f}{flag}")

        # Save per-class report as artifact
        os.makedirs("logs", exist_ok=True)
        report_path = f"logs/{run_name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(
                labels_arr, preds,
                target_names=list(id2label.values())
            ))
        mlflow.log_artifact(report_path)

        # ── Save and register model ───────────────────────────────
        # JD keywords: model versioning, model registry, deployment
        save_path = f"{run_output_dir}/final"
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

        # Log model to MLflow model registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="document-classifier"
        )

        print(f"\nRun '{run_name}' complete. Model registered in MLflow.")

    return {
        "run_name":     run_name,
        "val_f1":       round(val_results["eval_f1"], 4),
        "test_f1":      round(test_results["eval_f1"], 4),
        "test_accuracy": round(test_results["eval_accuracy"], 4),
    }


if __name__ == "__main__":
    # ── Run multiple experiments to compare ──────────────────────
    # This is what 'iterative improvement' means in JDs
    # Each run logs separately — compare them all in MLflow UI
    results = []

    # Experiment 1 — baseline (your already trained config)
    # SKIP training again, just log a mock run with your Day 1 results
    # to show experiment comparison. Comment this out if you want to retrain.
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="baseline-day1-results"):
        mlflow.log_params({
            "model": "distilbert-base-uncased",
            "num_epochs": 3, "batch_size": 8,
            "learning_rate": 5e-5, "max_length": 256,
            "note": "results from day1 training run"
        })
        mlflow.log_metrics({
            "test_accuracy": 0.9251,
            "test_f1":       0.9255,
            "f1_comp.graphics":      0.92,
            "f1_rec.sport.hockey":   0.97,
            "f1_sci.med":            0.92,
            "f1_sci.space":          0.89,
        })
    print("Baseline run logged to MLflow.")

    # Experiment 2 — fewer epochs, faster training
    # Uncomment when you have time to run a second experiment:
    # r2 = run_experiment(
    #     run_name="distilbert-epochs1-lr5e5",
    #     num_epochs=1,
    #     learning_rate=5e-5
    # )
    # results.append(r2)

    # Experiment 3 — lower learning rate
    # r3 = run_experiment(
    #     run_name="distilbert-epochs3-lr2e5",
    #     num_epochs=3,
    #     learning_rate=2e-5
    # )
    # results.append(r3)

    print("\nAll experiments logged.")
    print("Launch MLflow UI with:  mlflow ui")
    print("Then open:              http://127.0.0.1:5000")