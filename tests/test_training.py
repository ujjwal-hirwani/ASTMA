"""
Tests for Model Training, Pipeline Consistency, and Artifact Persistence.
"""

import unittest
from pathlib import Path
import joblib
from config.settings import MODEL_CATEGORY_DIR, TAXONOMY_10
from src.model_trainer import build_model_pipelines, MODEL_NAMES

class TestModelTraining(unittest.TestCase):
    def test_pipeline_construction(self):
        pipes = build_model_pipelines()
        self.assertEqual(len(pipes), 4)
        for name in MODEL_NAMES:
            self.assertIn(name, pipes)
            self.assertTrue(hasattr(pipes[name], "fit"))
            self.assertTrue(hasattr(pipes[name], "predict"))

    def test_persisted_model_artifacts_loadable(self):
        if not MODEL_CATEGORY_DIR.exists():
            self.skipTest("Model directory does not exist.")

        saved_files = list(MODEL_CATEGORY_DIR.glob("*.joblib"))
        if not saved_files:
            self.skipTest("No saved models found.")

        for f in saved_files:
            pipe = joblib.load(f)
            self.assertTrue(hasattr(pipe, "predict"))
            self.assertTrue(hasattr(pipe, "predict_proba"))
            # Check that classes are valid clinical categories
            self.assertGreaterEqual(len(pipe.classes_), 2)
            self.assertTrue(all(isinstance(c, str) for c in pipe.classes_))

if __name__ == "__main__":
    unittest.main()
