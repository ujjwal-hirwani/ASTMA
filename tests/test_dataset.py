"""
Tests for Dataset Generation, Schemas, Character Offsets, and Split Integrity.
"""

import unittest
from pathlib import Path
import pandas as pd
from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DATA_DIR, TAXONOMY_10
)
from src.data_generator import generate_telehealth_corpus
from src.data_validation import (
    validate_statements_schema, validate_mentions_schema,
    validate_character_offsets, validate_split_integrity,
    validate_taxonomy_categories
)

class TestDataset(unittest.TestCase):
    def test_small_corpus_generation_and_validation(self):
        df_stat, df_men = generate_telehealth_corpus(num_conversations=50, seed=123)
        self.assertEqual(len(df_stat), 100) # 50 patient + 50 clinician
        self.assertGreater(len(df_men), 30)

        # Validate schemas
        stat_ok, stat_err = validate_statements_schema(df_stat)
        self.assertTrue(stat_ok, f"Statement schema errors: {stat_err}")

        men_ok, men_err = validate_mentions_schema(df_men)
        self.assertTrue(men_ok, f"Mention schema errors: {men_err}")

        # Validate offsets
        off_ok, off_err = validate_character_offsets(df_stat, df_men)
        self.assertTrue(off_ok, f"Offset errors: {off_err}")

        # Validate taxonomy
        tax_ok, tax_err = validate_taxonomy_categories(df_men)
        self.assertTrue(tax_ok, f"Taxonomy errors: {tax_err}")

    def test_persisted_splits_integrity(self):
        train_path = SPLITS_DATA_DIR / "train.csv"
        val_path = SPLITS_DATA_DIR / "val.csv"
        test_path = SPLITS_DATA_DIR / "test.csv"

        if not (train_path.exists() and val_path.exists() and test_path.exists()):
            self.skipTest("Splits have not been generated yet.")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        # Zero leakage check
        leak_ok, leak_err = validate_split_integrity(df_train, df_val, df_test, group_col="conversation_id")
        self.assertTrue(leak_ok, f"Leakage detected: {leak_err}")

        # Class coverage
        valid_cats = set(TAXONOMY_10.keys())
        self.assertEqual(set(df_train["primary_category"].unique()), valid_cats)
        self.assertEqual(set(df_test["primary_category"].unique()), valid_cats)

if __name__ == "__main__":
    unittest.main()
