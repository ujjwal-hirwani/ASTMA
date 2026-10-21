"""
Unit tests for spaCy NLP Engine: extraction, negation scope, modifiers, and edge cases.
"""

import unittest
from src.nlp_engine import analyze_chat_message, normalize_text

class TestNLPEngine(unittest.TestCase):
    def test_single_symptom_extraction(self):
        text = "I have had a severe migraine for 3 days."
        records = analyze_chat_message(text)
        self.assertGreaterEqual(len(records), 1)
        migraine = next((r for r in records if "Migraine" in r["symptom"] or "Headache" in r["symptom"]), None)
        self.assertIsNotNone(migraine)
        self.assertEqual(migraine["status"], "Affirmative")
        self.assertEqual(migraine["category"], "Neurological")
        self.assertEqual(migraine["severity"], "Severe")
        self.assertIn("3 days", migraine["duration"].lower())

    def test_multiple_symptoms_extraction(self):
        text = "Patient reports dry cough and high fever since yesterday."
        records = analyze_chat_message(text)
        self.assertGreaterEqual(len(records), 2)
        symptoms = [r["symptom"].lower() for r in records]
        self.assertTrue(any("cough" in s for s in symptoms))
        self.assertTrue(any("fever" in s for s in symptoms))

    def test_negation_scope_independence(self):
        # Fever is negated, but headache is affirmative
        text = "No fever, but I have a headache."
        records = analyze_chat_message(text)
        self.assertGreaterEqual(len(records), 2)

        fever_rec = next((r for r in records if "fever" in r["symptom"].lower()), None)
        headache_rec = next((r for r in records if "headache" in r["symptom"].lower()), None)

        self.assertIsNotNone(fever_rec)
        self.assertIsNotNone(headache_rec)

        self.assertTrue(fever_rec["is_negated"], "Fever should be detected as negated")
        self.assertFalse(headache_rec["is_negated"], "Headache should remain affirmative")

    def test_denial_negation_phrase(self):
        text = "Patient denies chest pain and denies nausea."
        records = analyze_chat_message(text)
        self.assertGreaterEqual(len(records), 1)
        for r in records:
            self.assertTrue(r["is_negated"])

    def test_spelling_variants_and_informal(self):
        text = "I have a bad tummy ache and head ache."
        records = analyze_chat_message(text)
        self.assertGreaterEqual(len(records), 1)

    def test_empty_and_malformed_input(self):
        self.assertEqual(analyze_chat_message(""), [])
        self.assertEqual(analyze_chat_message("   "), [])
        self.assertEqual(analyze_chat_message("12345 !@#$%^&*()"), [])

    def test_character_offset_accuracy(self):
        text = "Hello doctor, I have an intense sore throat today."
        records = analyze_chat_message(text)
        self.assertGreaterEqual(len(records), 1)
        for r in records:
            start = r["char_start"]
            end = r["char_end"]
            expected = r["symptom"].lower()
            actual = text[start:end].lower()
            self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()
