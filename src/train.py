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

MODEL_CHECKPOINT  = "distilbert-base-uncased"
PROCESSED_DIR     = "data/processed"
MODEL_OUTPUT_DIR  = "models/distilbert-doc-classifier"


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


def run_training():
    # ── Load processed data ───────────────────────────────────────
    print("Loading processed data...")
    train_df = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    val_df   = pd.read_csv(f"{PROCESSED_DIR}/val.csv")
    test_df  = pd.read_csv(f"{PROCESSED_DIR}/test.csv")

    with open(f"{PROCESSED_DIR}/label_map.json") as f:
        meta = json.load(f)

    label_map = meta["label_map"]
    id2label  = {int(k): v for k, v in meta["id2label"].items()}
    num_labels = len(label_map)

    # ── Tokenizer + Datasets ──────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer     = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    train_dataset = DocumentDataset(train_df, tokenizer)
    val_dataset   = DocumentDataset(val_df,   tokenizer)
    test_dataset  = DocumentDataset(test_df,  tokenizer)

    # ── Model ─────────────────────────────────────────────────────
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_map
    )

    # ── Training arguments (CPU + Windows safe) ───────────────────
    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="logs",
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,   # Windows fix
        use_cpu=True               # CPU only
    )

    # ── Train ─────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("\nStarting fine-tuning (this takes 25-45 min on CPU)...")
    print("You will see progress bars for each epoch.\n")
    trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_dataset)
    print(f"  Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  Test F1:       {results['eval_f1']:.4f}")

    # ── Per-class fairness check ──────────────────────────────────
    print("\nPer-class performance (responsible AI check):")
    from sklearn.metrics import classification_report, f1_score
    preds_out = trainer.predict(test_dataset)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    labels    = preds_out.label_ids
    print(classification_report(labels, preds, target_names=list(id2label.values())))

    # ── Save model ────────────────────────────────────────────────
    save_path = f"{MODEL_OUTPUT_DIR}/final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")
    print("\nTraining complete. Ready for Day 3-4: MLflow experiment tracking.")


if __name__ == "__main__":
    run_training()