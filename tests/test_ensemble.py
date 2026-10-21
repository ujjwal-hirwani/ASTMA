"""
Tests for Weighted Probability Ensemble Classifier.
"""

import unittest
import numpy as np
from src.ensemble import WeightedEnsembleClassifier, derive_ensemble_weights_from_val

class DummyModel:
    def __init__(self, classes, probs):
        self.classes_ = np.array(classes)
        self.probs = np.array(probs)

    def predict_proba(self, X):
        return self.probs

    def predict(self, X):
        idx = np.argmax(self.probs, axis=1)
        return self.classes_[idx]

class TestEnsemble(unittest.TestCase):
    def test_weights_sum_to_one(self):
        val_metrics = {
            "m1": {"val_macro_f1": 0.85},
            "m2": {"val_macro_f1": 0.90},
            "m3": {"val_macro_f1": 0.95}
        }
        weights = derive_ensemble_weights_from_val(val_metrics)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
        self.assertGreater(weights["m3"], weights["m1"])

    def test_probability_combination_and_prediction(self):
        classes = ["CatA", "CatB", "CatC"]
        m1 = DummyModel(classes, [[0.7, 0.2, 0.1]])
        m2 = DummyModel(classes, [[0.5, 0.4, 0.1]])

        weights = {"m1": 0.5, "m2": 0.5}
        ensemble = WeightedEnsembleClassifier(models={"m1": m1, "m2": m2}, weights=weights)

        probs = ensemble.predict_proba(["sample text"])
        self.assertEqual(probs.shape, (1, 3))
        # Expected: 0.5*[0.7, 0.2, 0.1] + 0.5*[0.5, 0.4, 0.1] = [0.6, 0.3, 0.1]
        self.assertAlmostEqual(probs[0, 0], 0.6)
        self.assertAlmostEqual(probs[0, 1], 0.3)
        self.assertAlmostEqual(probs[0, 2], 0.1)

        pred = ensemble.predict(["sample text"])
        self.assertEqual(pred[0], "CatA")

    def test_class_alignment(self):
        # Different class orders in models
        m1 = DummyModel(["CatA", "CatB"], [[0.8, 0.2]])
        m2 = DummyModel(["CatB", "CatA"], [[0.3, 0.7]]) # CatA has 0.7, CatB has 0.3

        ensemble = WeightedEnsembleClassifier(models={"m1": m1, "m2": m2}, weights={"m1": 0.5, "m2": 0.5})
        probs = ensemble.predict_proba(["sample text"])
        # Both models assign roughly 0.75 to CatA, 0.25 to CatB
        idx_a = list(ensemble.classes_).index("CatA")
        self.assertAlmostEqual(probs[0, idx_a], 0.75, places=2)

if __name__ == "__main__":
    unittest.main()
