"""
Tests for Streamlit dashboard visualization helpers, reporting, and core utility logic.
"""

import unittest
import pandas as pd
from src.visualizer import (
    plot_category_distribution, plot_top_symptoms, plot_negation_status,
    plot_urgency_breakdown, plot_model_comparison_bar, generate_cooccurrence_network_html
)
from src.reporting import generate_text_summary, generate_pdf_summary

class TestAppLogic(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame([
            {"chat_id": "CHAT-1", "symptom": "Headache", "category": "Neurological", "status": "Affirmative", "is_negated": False, "urgency": "Low"},
            {"chat_id": "CHAT-1", "symptom": "Fever", "category": "General / Systemic", "status": "Negated", "is_negated": True, "urgency": "High"},
            {"chat_id": "CHAT-2", "symptom": "Cough", "category": "Respiratory", "status": "Affirmative", "is_negated": False, "urgency": "Low"},
            {"chat_id": "CHAT-2", "symptom": "Chest Pain", "category": "Cardiovascular", "status": "Affirmative", "is_negated": False, "urgency": "High"}
        ])

    def test_charts_creation(self):
        fig_cat = plot_category_distribution(self.sample_df)
        self.assertIsNotNone(fig_cat)

        fig_top = plot_top_symptoms(self.sample_df, top_n=5)
        self.assertIsNotNone(fig_top)

        fig_neg = plot_negation_status(self.sample_df)
        self.assertIsNotNone(fig_neg)

        fig_urg = plot_urgency_breakdown(self.sample_df)
        self.assertIsNotNone(fig_urg)

    def test_model_comparison_plot(self):
        dummy_metrics = {
            "model_a": {"accuracy": 0.95, "macro_f1": 0.94, "weighted_f1": 0.95, "macro_precision": 0.94, "macro_recall": 0.94},
            "model_b": {"accuracy": 0.92, "macro_f1": 0.91, "weighted_f1": 0.92, "macro_precision": 0.91, "macro_recall": 0.91}
        }
        fig_comp = plot_model_comparison_bar(dummy_metrics)
        self.assertIsNotNone(fig_comp)

    def test_network_generation(self):
        html_code = generate_cooccurrence_network_html(self.sample_df)
        self.assertIsInstance(html_code, str)
        self.assertIn("html", html_code.lower())

    def test_report_generation(self):
        txt = generate_text_summary(self.sample_df, total_chats=2)
        self.assertIn("TELEHEALTH TEXT MINING EXECUTIVE SUMMARY", txt)
        self.assertIn("Neurological", txt)

        pdf_bytes = generate_pdf_summary(self.sample_df, total_chats=2)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 100)

if __name__ == "__main__":
    unittest.main()
