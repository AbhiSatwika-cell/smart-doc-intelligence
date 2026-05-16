from datasets import load_dataset
import pandas as pd
import os, json

TARGET_LABELS = [
    "sci.med",
    "sci.space",
    "comp.graphics",
    "rec.sport.hockey"
]

RAW_SAVE_DIR       = "data/raw"
PROCESSED_SAVE_DIR = "data/processed"


def load_raw_data(save_locally=True):
    print("=" * 50)
    print("STAGE 1: Raw Data Ingestion")
    print("=" * 50)

    print("\n[1/4] Downloading from Hugging Face Hub...")
    dataset = load_dataset("SetFit/20_newsgroups")
    print(f"      Train (full): {len(dataset['train'])} samples")
    print(f"      Test  (full): {len(dataset['test'])} samples")

    print("\n[2/4] Converting to Pandas...")
    train_raw = pd.DataFrame(dataset["train"])
    test_raw  = pd.DataFrame(dataset["test"])

    print(f"\n[3/4] Filtering to {len(TARGET_LABELS)} classes...")
    train_df = train_raw.query("label_text in @TARGET_LABELS").reset_index(drop=True).copy()
    test_df  = test_raw.query("label_text in @TARGET_LABELS").reset_index(drop=True).copy()
    print(f"      Train: {len(train_df)} samples")
    print(f"      Test:  {len(test_df)} samples")

    print("\n      Class distribution (train):")
    for label, count in train_df["label_text"].value_counts().items():
        print(f"        {label:<25} {count}")

    meta = {
        "dataset":       "SetFit/20_newsgroups",
        "target_labels": TARGET_LABELS,
        "train_size":    len(train_df),
        "test_size":     len(test_df),
    }

    if save_locally:
        os.makedirs(RAW_SAVE_DIR, exist_ok=True)
        train_df.to_csv(f"{RAW_SAVE_DIR}/train_raw.csv", index=False)
        test_df.to_csv(f"{RAW_SAVE_DIR}/test_raw.csv",   index=False)
        with open(f"{RAW_SAVE_DIR}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n[4/4] Saved raw data to {RAW_SAVE_DIR}/")

    print("\nStage 1 complete.")
    return train_df, test_df, meta


def validate_raw_data(train_df, test_df):
    print("\n── Validation ───────────────────────────────────")
    issues = []

    assert len(train_df) > 0, "Train set is empty!"
    assert len(test_df)  > 0, "Test set is empty!"

    for col in ["text", "label", "label_text"]:
        assert col in train_df.columns, f"Missing column: {col}"

    nulls = train_df[["text", "label_text"]].isnull().sum()
    if nulls.any():
        issues.append(f"Nulls found: {nulls[nulls>0].to_dict()}")

    dist  = train_df["label_text"].value_counts()
    ratio = dist.max() / dist.min()
    if ratio > 3.0:
        issues.append(f"Class imbalance ratio: {ratio:.1f}x — consider oversampling")

    short = (train_df["text"].str.len() < 20).sum()
    if short > 0:
        issues.append(f"{short} texts shorter than 20 chars")

    if issues:
        print("  WARNINGS:")
        for i in issues:
            print(f"    - {i}")
    else:
        print("  All checks passed.")

    sample = train_df.iloc[0]
    print(f"\n  Sample — Label: {sample['label_text']}")
    print(f"  Text preview:   {sample['text'][:150]}...")
    return len(issues) == 0


def clean_text(text):
    import re
    text = re.sub(r'^.*?:\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:512]


def clean_dataset(df):
    print("\n── Cleaning ──────────────────────────────────────")
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 20].reset_index(drop=True)

    label_map = {label: idx for idx, label in enumerate(sorted(df["label_text"].unique()))}
    id2label  = {str(v): k for k, v in label_map.items()}
    df["label_id"] = df["label_text"].map(label_map)

    print(f"  Label mapping: {label_map}")
    print(f"  Clean samples: {len(df)}")
    return df, label_map, id2label


def split_data(train_df, test_df):
    from sklearn.model_selection import train_test_split
    print("\n── Splitting ─────────────────────────────────────")
    train_final, val_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df["label_id"],
        random_state=42
    )
    train_final = train_final.reset_index(drop=True)
    val_df      = val_df.reset_index(drop=True)
    print(f"  Train: {len(train_final)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_final, val_df, test_df


def save_processed(train_df, val_df, test_df, label_map, id2label):
    os.makedirs(PROCESSED_SAVE_DIR, exist_ok=True)
    train_df.to_csv(f"{PROCESSED_SAVE_DIR}/train.csv", index=False)
    val_df.to_csv(f"{PROCESSED_SAVE_DIR}/val.csv",     index=False)
    test_df.to_csv(f"{PROCESSED_SAVE_DIR}/test.csv",   index=False)
    with open(f"{PROCESSED_SAVE_DIR}/label_map.json", "w") as f:
        json.dump({"label_map": label_map, "id2label": id2label}, f, indent=2)
    print(f"\n  Saved processed data to {PROCESSED_SAVE_DIR}/")


if __name__ == "__main__":
    # Stage 1 — ingest
    train_raw, test_raw, meta = load_raw_data(save_locally=True)
    validate_raw_data(train_raw, test_raw)

    # Stage 2 — clean
    train_clean, label_map, id2label = clean_dataset(train_raw)
    test_clean,  _,         _        = clean_dataset(test_raw)

    # Stage 3 — split
    train_df, val_df, test_df = split_data(train_clean, test_clean)

    # Stage 4 — save
    save_processed(train_df, val_df, test_df, label_map, id2label)

    print("\nPipeline complete. Ready for fine-tuning (src/train.py)")