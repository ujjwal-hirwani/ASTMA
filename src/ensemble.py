"""
Weighted Probability Ensemble Classifier for ASTMA.
Combines prediction probabilities from multiple supervised classifiers using
weights derived strictly from validation set macro-F1 performance.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from config.settings import MODEL_CATEGORY_DIR, MODEL_METADATA_DIR

class WeightedEnsembleClassifier:
    """
    Combines class probability distributions from multiple classifiers
    with normalized validation-performance weights:
        P_ensemble(c | x) = sum(w_i * P_i(c | x))
    """
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {}
        self.classes_ = None

        self._validate_and_initialize()

    def _validate_and_initialize(self):
        if not self.models:
            raise ValueError("No models provided to WeightedEnsembleClassifier.")

        # Determine reference classes from first model
        first_model = next(iter(self.models.values()))
        self.classes_ = np.array(list(first_model.classes_))

        # Check weights
        if not self.weights:
            # Default to uniform weights if not provided
            n = len(self.models)
            self.weights = {m: 1.0 / n for m in self.models.keys()}
        else:
            # Normalize weights to sum to 1.0
            total_w = sum(self.weights.values())
            if total_w <= 0:
                raise ValueError("Sum of ensemble weights must be positive.")
            self.weights = {m: w / total_w for m, w in self.weights.items()}

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Computes aligned weighted probability distribution across all models.
        Returns 2D numpy array of shape (n_samples, n_classes).
        """
        if isinstance(X, str):
            X = [X]

        n_samples = len(X)
        n_classes = len(self.classes_)
        ensemble_probs = np.zeros((n_samples, n_classes), dtype=float)

        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w <= 0:
                continue

            if not hasattr(model, "predict_proba"):
                raise AttributeError(f"Model {name} does not support predict_proba.")

            raw_probs = model.predict_proba(X)
            model_classes = list(model.classes_)

            # Explicit column alignment mapping model classes to self.classes_
            aligned_probs = np.zeros((n_samples, n_classes), dtype=float)
            for col_idx, cls_name in enumerate(model_classes):
                target_idx = np.where(self.classes_ == cls_name)[0]
                if len(target_idx) > 0:
                    aligned_probs[:, target_idx[0]] = raw_probs[:, col_idx]

            ensemble_probs += w * aligned_probs

        # Ensure normalized rows
        row_sums = ensemble_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        ensemble_probs = ensemble_probs / row_sums

        return ensemble_probs

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class with highest ensemble probability."""
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]

    def get_confidence_scores(self, X: Any) -> List[Dict[str, float]]:
        """Returns list of dictionaries mapping class -> probability for each sample."""
        probs = self.predict_proba(X)
        results = []
        for row in probs:
            results.append({cls_name: round(float(p), 4) for cls_name, p in zip(self.classes_, row)})
        return results

    def save_configuration(self, out_path: Optional[Path] = None) -> None:
        """Saves ensemble weights and class mapping to JSON."""
        if out_path is None:
            out_path = MODEL_METADATA_DIR / "ensemble_config.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "weights": self.weights,
            "classes": list(self.classes_),
            "model_names": list(self.models.keys())
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

def derive_ensemble_weights_from_val(val_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes normalized weights from validation macro-F1 scores:
        w_i = val_macro_f1_i / sum(val_macro_f1_j)
    """
    f1_scores = {}
    for name, metrics in val_metrics.items():
        f1 = float(metrics.get("val_macro_f1", 0.0))
        f1_scores[name] = max(0.001, f1)

    total_f1 = sum(f1_scores.values())
    weights = {name: round(score / total_f1, 4) for name, score in f1_scores.items()}

    # Ensure exact sum = 1.0
    diff = 1.0 - sum(weights.values())
    first_key = next(iter(weights.keys()))
    weights[first_key] = round(weights[first_key] + diff, 4)

    return weights
