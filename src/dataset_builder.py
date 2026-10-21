"""
Dataset Builder for ASTMA.
Constructs training examples combining symptom mentions and surrounding context,
creates leakage-free grouped splits (Train/Val/Test), and saves manifests.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DATA_DIR,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
from src.data_generator import generate_telehealth_corpus
from src.data_validation import run_full_validation_suite, validate_split_integrity

def build_feature_text(statement_text: str, start_char: int, end_char: int, window_chars: int = 50) -> str:
    """
    Constructs an ML input feature by highlighting the symptom mention
    along with its surrounding sentence context.
    """
    start_char = max(0, int(start_char))
    end_char = min(len(statement_text), int(end_char))
    mention_span = statement_text[start_char:end_char]

    ctx_start = max(0, start_char - window_chars)
    ctx_end = min(len(statement_text), end_char + window_chars)

    left_ctx = statement_text[ctx_start:start_char].strip()
    right_ctx = statement_text[end_char:ctx_end].strip()

    # Form enriched feature string: "symptom_span [SEP] left_ctx right_ctx"
    feature_text = f"{mention_span} | context: {left_ctx} {mention_span} {right_ctx}".strip()
    return feature_text

def create_grouped_splits(df_examples: pd.DataFrame, group_col: str = "conversation_id",
                          seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits examples into Train (70%), Val (15%), and Test (15%)
    strictly by group_col to guarantee zero conversation leakage.
    """
    unique_groups = df_examples[group_col].drop_duplicates().sample(frac=1.0, random_state=seed).values
    n_groups = len(unique_groups)

    n_train = int(n_groups * TRAIN_RATIO)
    n_val = int(n_groups * VAL_RATIO)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    train_df = df_examples[df_examples[group_col].isin(train_groups)].copy()
    val_df = df_examples[df_examples[group_col].isin(val_groups)].copy()
    test_df = df_examples[df_examples[group_col].isin(test_groups)].copy()

    # Validate zero leakage
    leak_ok, leak_errs = validate_split_integrity(train_df, val_df, test_df, group_col=group_col)
    if not leak_ok:
        raise ValueError(f"Split leakage error: {leak_errs}")

    return train_df, val_df, test_df

def generate_and_save_dataset(num_conversations: int = 4000, seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Full pipeline: Generates ~8000 statements, validates, prepares ML examples,
    splits by conversation, and writes all artifacts to disk.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate Raw Corpus
    df_statements, df_mentions = generate_telehealth_corpus(num_conversations=num_conversations, seed=seed)

    # 2. Validation
    val_results = run_full_validation_suite(df_statements, df_mentions)
    if not val_results["is_valid"]:
        raise ValueError(f"Dataset generation failed validation: {val_results['errors']}")

    # 3. Save Raw Data
    df_statements.to_csv(RAW_DATA_DIR / "telehealth_statements.csv", index=False)
    df_mentions.to_csv(RAW_DATA_DIR / "telehealth_mentions.csv", index=False)

    # 4. Prepare Processed ML Examples
    stmt_lookup = dict(zip(df_statements["record_id"], df_statements["text"]))
    examples = []

    for _, row in df_mentions.iterrows():
        rec_id = row["record_id"]
        stmt_text = stmt_lookup.get(rec_id, "")
        feat_text = build_feature_text(stmt_text, row["start_char"], row["end_char"])

        examples.append({
            "mention_id": row["mention_id"],
            "conversation_id": row["conversation_id"],
            "record_id": rec_id,
            "symptom_text": row["symptom_text"],
            "normalized_symptom": row["normalized_symptom"],
            "feature_text": feat_text,
            "full_statement": stmt_text,
            "primary_category": row["primary_category"],
            "is_negated": row["is_negated"],
            "severity": row["severity"],
            "duration_text": row["duration_text"]
        })

    df_examples = pd.DataFrame(examples)
    df_examples.to_csv(PROCESSED_DATA_DIR / "symptom_classification_examples.csv", index=False)

    # 5. Create Grouped Splits
    train_df, val_df, test_df = create_grouped_splits(df_examples, group_col="conversation_id", seed=seed)

    train_df.to_csv(SPLITS_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DATA_DIR / "test.csv", index=False)

    # 6. Save Manifest
    manifest = {
        "dataset_name": "ASTMA Synthetic Telehealth Symptom Corpus",
        "generator_version": "2.0.0",
        "random_seed": seed,
        "num_conversations": num_conversations,
        "total_statements": len(df_statements),
        "total_mentions": len(df_mentions),
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "train_category_distribution": train_df["primary_category"].value_counts().to_dict(),
        "val_category_distribution": val_df["primary_category"].value_counts().to_dict(),
        "test_category_distribution": test_df["primary_category"].value_counts().to_dict(),
        "validation_passed": True
    }

    with open(SPLITS_DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
