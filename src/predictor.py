"""
Unified Predictor Module for ASTMA.
Integrates spaCy extraction pipeline with trained classifiers and the weighted ensemble
to produce multi-model categorization, probability distributions, and uncertainty scores.
"""

from typing import Dict, List, Any, Optional
import json
import joblib
import numpy as np

from config.settings import (
    MODEL_CATEGORY_DIR, MODEL_METADATA_DIR, MODEL_EVAL_DIR
)
from src.nlp_engine import analyze_chat_message, normalize_text
from src.dataset_builder import build_feature_text
from src.ensemble import WeightedEnsembleClassifier

class SymptomPredictor:
    """
    Inference orchestrator for free-text patient statements.
    Extracts clinical mentions and generates predictions from each classifier
    and the validation-weighted ensemble.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.ensemble: Optional[WeightedEnsembleClassifier] = None
        self.classes: List[str] = []
        self.is_loaded: bool = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads trained pipelines and instantiates the ensemble."""
        if not MODEL_CATEGORY_DIR.exists():
            return

        loaded_models = {}
        for p in MODEL_CATEGORY_DIR.glob("*.joblib"):
            try:
                loaded_models[p.stem] = joblib.load(p)
            except Exception:
                pass

        if not loaded_models:
            return

        self.models = loaded_models

        # Load ensemble config or fallback to uniform
        config_path = MODEL_METADATA_DIR / "ensemble_config.json"
        weights = None
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    weights = cfg.get("weights")
            except Exception:
                pass

        try:
            self.ensemble = WeightedEnsembleClassifier(models=self.models, weights=weights)
            self.classes = list(self.ensemble.classes_)
            self.is_loaded = True
        except Exception:
            self.is_loaded = False

    def reload(self):
        """Reloads active model artifacts from disk."""
        self._load_artifacts()

    def update_models(self, models: Dict[str, Any], ensemble: Any):
        """Directly injects newly trained models and ensemble in-memory."""
        if models and ensemble:
            self.models = models
            self.ensemble = ensemble
            self.classes = list(ensemble.classes_)
            self.is_loaded = True

    def predict_query(self, user_text: str) -> Dict[str, Any]:
        """
        Processes arbitrary patient free-form text.
        Returns:
            - Validated query text
            - Extracted symptom mentions with attributes (negation, duration, severity, urgency)
            - Multi-model and ensemble classifications for each mention
            - Confidence breakdown & uncertainty notices
        """
        clean_text = normalize_text(user_text)
        if not clean_text:
            return {
                "success": False,
                "error": "Query text cannot be empty.",
                "mentions": []
            }

        if len(clean_text) > 4000:
            return {
                "success": False,
                "error": "Query text exceeds maximum allowed length of 4000 characters.",
                "mentions": []
            }

        # 1. NLP Extraction
        extracted_mentions = analyze_chat_message(clean_text)

        if not extracted_mentions:
            return {
                "success": True,
                "raw_text": clean_text,
                "has_symptoms": False,
                "message": "No recognized clinical symptom mentions were detected in the input.",
                "mentions": []
            }

        # 2. ML Inference for each mention
        results_mentions = []
        for m in extracted_mentions:
            start = m["char_start"]
            end = m["char_end"]
            feat_text = build_feature_text(clean_text, start, end)

            mention_summary = {
                "symptom_text": m["symptom"],
                "normalized_symptom": m["normalized_symptom"],
                "status": m["status"],
                "is_negated": m["is_negated"],
                "severity": m["severity"],
                "duration": m["duration"],
                "urgency": m["urgency"],
                "char_span": [start, end],
                "nlp_rule_category": m["category"]
            }

            if self.is_loaded and self.ensemble is not None:
                # Individual model predictions
                model_preds = {}
                for m_name, model in self.models.items():
                    pred_cat = model.predict([feat_text])[0]
                    proba = None
                    if hasattr(model, "predict_proba"):
                        raw_prob = model.predict_proba([feat_text])[0]
                        cls_idx = list(model.classes_).index(pred_cat)
                        proba = round(float(raw_prob[cls_idx]), 4)
                    model_preds[m_name] = {
                        "predicted_category": pred_cat,
                        "confidence": proba
                    }

                # Ensemble prediction and complete probability distribution
                ens_pred = self.ensemble.predict([feat_text])[0]
                ens_proba_dist = self.ensemble.get_confidence_scores([feat_text])[0]
                top_confidence = ens_proba_dist.get(ens_pred, 0.0)

                # Uncertainty assessment
                is_low_confidence = top_confidence < 0.40

                mention_summary["ml_available"] = True
                mention_summary["model_predictions"] = model_preds
                mention_summary["ensemble_prediction"] = ens_pred
                mention_summary["ensemble_confidence"] = top_confidence
                mention_summary["probability_distribution"] = ens_proba_dist
                mention_summary["is_low_confidence"] = is_low_confidence
                mention_summary["uncertainty_notice"] = (
                    "Low model confidence detected; prediction may be ambiguous." if is_low_confidence else "Confidence within normal range."
                )
            else:
                mention_summary["ml_available"] = False
                mention_summary["ensemble_prediction"] = m["category"]
                mention_summary["notice"] = "Trained ML models are currently not loaded. Showing NLP rule-based category."

            results_mentions.append(mention_summary)

        return {
            "success": True,
            "raw_text": clean_text,
            "has_symptoms": True,
            "total_mentions": len(results_mentions),
            "mentions": results_mentions,
            "disclaimer": (
                "Educational and Text Mining Analytics Demo only. "
                "This application does not provide medical advice, diagnosis, or clinical triage."
            )
        }
