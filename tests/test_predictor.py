"""
Tests for SymptomPredictor inference orchestrator and custom queries.
"""

import unittest
from src.predictor import SymptomPredictor

class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = SymptomPredictor()

    def test_empty_query(self):
        res = self.predictor.predict_query("")
        self.assertFalse(res["success"])
        self.assertIn("empty", res["error"])

    def test_no_symptom_query(self):
        res = self.predictor.predict_query("Good morning, what are your clinic hours tomorrow?")
        self.assertTrue(res["success"])
        self.assertFalse(res["has_symptoms"])
        self.assertEqual(len(res["mentions"]), 0)

    def test_single_symptom_prediction(self):
        query = "I've been getting severe headaches for the past week."
        res = self.predictor.predict_query(query)
        self.assertTrue(res["success"])
        self.assertTrue(res["has_symptoms"])
        self.assertGreaterEqual(res["total_mentions"], 1)

        first_m = res["mentions"][0]
        self.assertIn("symptom_text", first_m)
        self.assertEqual(first_m["status"], "Affirmative")
        if first_m.get("ml_available"):
            self.assertEqual(first_m["ensemble_prediction"], "Neurological")
            self.assertIn("calibrated_linear_svm", first_m["model_predictions"])

    def test_compound_affirmative_negated_prediction(self):
        query = "I have a bad cough, but no chest pain."
        res = self.predictor.predict_query(query)
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["total_mentions"], 2)

        cough = next((m for m in res["mentions"] if "cough" in m["symptom_text"].lower()), None)
        chest = next((m for m in res["mentions"] if "chest" in m["symptom_text"].lower()), None)

        self.assertIsNotNone(cough)
        self.assertIsNotNone(chest)
        self.assertFalse(cough["is_negated"])
        self.assertTrue(chest["is_negated"])

if __name__ == "__main__":
    unittest.main()
