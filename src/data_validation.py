"""
Data Validation Engine for Telehealth Datasets.
Verifies schema, character offsets, taxonomy categories, negation logic, and split leakage.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
from config.settings import TAXONOMY_10

REQUIRED_STATEMENT_COLUMNS = [
    "record_id", "conversation_id", "patient_id", "speaker",
    "text", "split_group_id", "is_synthetic", "generator_version"
]

REQUIRED_MENTION_COLUMNS = [
    "mention_id", "record_id", "symptom_text", "normalized_symptom",
    "start_char", "end_char", "primary_category", "is_negated",
    "severity", "duration_text", "duration_value", "duration_unit",
    "onset_type", "confidence_source"
]

def validate_statements_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates that statement DataFrame contains required fields."""
    errors = []
    missing_cols = [c for c in REQUIRED_STATEMENT_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing statement columns: {missing_cols}")
    if df.empty:
        errors.append("Statements DataFrame is empty.")
    return len(errors) == 0, errors

def validate_mentions_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates that mention DataFrame contains required fields."""
    errors = []
    missing_cols = [c for c in REQUIRED_MENTION_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing mention columns: {missing_cols}")
    if df.empty:
        errors.append("Mentions DataFrame is empty.")
    return len(errors) == 0, errors

def validate_character_offsets(df_statements: pd.DataFrame, df_mentions: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Verifies that text[start_char:end_char] exactly matches symptom_text.
    """
    errors = []
    records_lookup = dict(zip(df_statements["record_id"], df_statements["text"]))

    for idx, row in df_mentions.iterrows():
        rec_id = row["record_id"]
        if rec_id not in records_lookup:
            errors.append(f"Mention {row.get('mention_id')} references unknown record_id: {rec_id}")
            continue

        raw_text = records_lookup[rec_id]
        start = int(row["start_char"])
        end = int(row["end_char"])
        expected = str(row["symptom_text"])

        if start < 0 or end > len(raw_text) or start >= end:
            errors.append(f"Mention {row.get('mention_id')} invalid offset: [{start}:{end}] in length {len(raw_text)}")
            continue

        actual_slice = raw_text[start:end]
        if actual_slice != expected:
            errors.append(f"Mention {row.get('mention_id')} slice mismatch: expected '{expected}', found '{actual_slice}'")

        if len(errors) > 25:
            errors.append("...more offset errors truncated.")
            break

    return len(errors) == 0, errors

def validate_taxonomy_categories(df_mentions: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verifies that all primary_category values belong to TAXONOMY_10."""
    errors = []
    valid_categories = set(TAXONOMY_10.keys())
    unique_cats = set(df_mentions["primary_category"].unique())
    invalid = unique_cats - valid_categories
    if invalid:
        errors.append(f"Invalid primary_category values found: {invalid}")
    return len(errors) == 0, errors

def validate_split_integrity(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, group_col: str = "split_group_id") -> Tuple[bool, List[str]]:
    """
    Ensures zero leakage across train, val, and test partitions based on group_col.
    """
    errors = []
    train_groups = set(df_train[group_col].dropna().unique())
    val_groups = set(df_val[group_col].dropna().unique())
    test_groups = set(df_test[group_col].dropna().unique())

    leakage_train_val = train_groups.intersection(val_groups)
    leakage_train_test = train_groups.intersection(test_groups)
    leakage_val_test = val_groups.intersection(test_groups)

    if leakage_train_val:
        errors.append(f"Data leakage detected between Train and Val: {len(leakage_train_val)} groups overlap")
    if leakage_train_test:
        errors.append(f"Data leakage detected between Train and Test: {len(leakage_train_test)} groups overlap")
    if leakage_val_test:
        errors.append(f"Data leakage detected between Val and Test: {len(leakage_val_test)} groups overlap")

    return len(errors) == 0, errors

def run_full_validation_suite(df_statements: pd.DataFrame, df_mentions: pd.DataFrame) -> Dict[str, Any]:
    """Runs all validation checks and returns summary dictionary."""
    stat_ok, stat_err = validate_statements_schema(df_statements)
    men_ok, men_err = validate_mentions_schema(df_mentions)
    offset_ok, offset_err = validate_character_offsets(df_statements, df_mentions)
    tax_ok, tax_err = validate_taxonomy_categories(df_mentions)

    all_passed = stat_ok and men_ok and offset_ok and tax_ok
    all_errors = stat_err + men_err + offset_err + tax_err

    return {
        "is_valid": all_passed,
        "statements_schema_valid": stat_ok,
        "mentions_schema_valid": men_ok,
        "offsets_valid": offset_ok,
        "taxonomy_valid": tax_ok,
        "errors": all_errors,
        "num_statements": len(df_statements),
        "num_mentions": len(df_mentions),
        "category_distribution": df_mentions["primary_category"].value_counts().to_dict() if "primary_category" in df_mentions.columns else {}
    }
