"""
Evaluation Engine for ASTMA Supervised Classifiers and Ensemble.
Computes comprehensive classification metrics on held-out test data,
confusion matrices, inference timings, and error analysis.
"""

import time
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config.settings import (
    SPLITS_DATA_DIR, MODEL_CATEGORY_DIR, MODEL_EVAL_DIR, MODEL_METADATA_DIR
)
from src.ensemble import WeightedEnsembleClassifier, derive_ensemble_weights_from_val

def load_trained_models() -> Dict[str, Any]:
    """Loads all saved model pipelines from MODEL_CATEGORY_DIR."""
    models = {}
    for p in MODEL_CATEGORY_DIR.glob("*.joblib"):
        name = p.stem
        models[name] = joblib.load(p)
    return models

def evaluate_classifier(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                        class_labels: List[str]) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates a single model or ensemble on X_test and y_test.
    Measures prediction speed, overall accuracy, macro/weighted precision/recall/F1,
    per-class metrics, and confusion matrix.
    """
    n_samples = len(X_test)

    # Measure inference time
    t0 = time.time()
    y_pred = model.predict(X_test)
    inf_time_total = time.time() - t0
    inf_time_per_sample_ms = (inf_time_total / n_samples) * 1000.0

    acc = float(accuracy_score(y_test, y_pred))
    macro_p = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_report = classification_report(y_test, y_pred, labels=class_labels,
                                             output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "inference_time_total_s": round(inf_time_total, 4),
        "inference_time_ms_per_sample": round(inf_time_per_sample_ms, 3),
        "per_class": {
            cls: {
                "precision": round(per_class_report[cls]["precision"], 4),
                "recall": round(per_class_report[cls]["recall"], 4),
                "f1_score": round(per_class_report[cls]["f1-score"], 4),
                "support": int(per_class_report[cls]["support"])
            }
            for cls in class_labels if cls in per_class_report
        }
    }

    return metrics, cm, y_pred

def run_evaluation_suite() -> Dict[str, Any]:
    """
    Loads test split and all trained models + ensemble, computes test metrics,
    performs error analysis, and writes outputs to MODEL_EVAL_DIR.
    """
    MODEL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    test_path = SPLITS_DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split {test_path} not found. Run dataset generation first.")

    test_df = pd.read_csv(test_path)
    X_test = test_df["feature_text"].values
    y_test = test_df["primary_category"].values

    models = load_trained_models()
    if not models:
        raise FileNotFoundError("No trained models found in models/symptom_category/. Run training first.")

    # Load validation metrics to determine ensemble weights
    train_summary_file = MODEL_METADATA_DIR / "training_summary.json"
    val_metrics = {}
    if train_summary_file.exists():
        with open(train_summary_file, "r", encoding="utf-8") as f:
            train_summary = json.load(f)
            val_metrics = train_summary.get("models", {})

    ensemble_weights = derive_ensemble_weights_from_val(val_metrics)
    ensemble = WeightedEnsembleClassifier(models=models, weights=ensemble_weights)
    ensemble.save_configuration()

    # Shared class labels
    class_labels = list(ensemble.classes_)

    all_metrics = {}
    all_cms = {}
    predictions_dict = {}

    # Evaluate individual models
    for name, model in models.items():
        m_dict, cm, y_pred = evaluate_classifier(model, X_test, y_test, class_labels)
        # Add training time from metadata if available
        if name in val_metrics:
            m_dict["training_time_seconds"] = val_metrics[name].get("training_time_seconds", 0.0)
        all_metrics[name] = m_dict
        all_cms[name] = cm.tolist()
        predictions_dict[name] = y_pred

    # Evaluate ensemble
    ens_dict, ens_cm, ens_pred = evaluate_classifier(ensemble, X_test, y_test, class_labels)
    ens_dict["ensemble_weights"] = ensemble_weights
    ens_dict["training_time_seconds"] = sum(m.get("training_time_seconds", 0.0) for m in all_metrics.values())
    all_metrics["weighted_ensemble"] = ens_dict
    all_cms["weighted_ensemble"] = ens_cm.tolist()
    predictions_dict["weighted_ensemble"] = ens_pred

    # Error analysis: Find instances where models disagreed or ensemble misclassified
    error_cases = []
    for idx, (true_label, feat) in enumerate(zip(y_test, X_test)):
        ens_p = ens_pred[idx]
        if ens_p != true_label or any(predictions_dict[m][idx] != true_label for m in models):
            case = {
                "sample_index": idx,
                "feature_text": feat,
                "ground_truth": true_label,
                "ensemble_prediction": ens_p,
                "model_predictions": {m: predictions_dict[m][idx] for m in models}
            }
            error_cases.append(case)
            if len(error_cases) >= 50:
                break

    # Determine best model based on macro_f1
    best_model_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]["macro_f1"])

    results_payload = {
        "evaluation_dataset": "ASTMA Test Split (Zero-Leakage Held-Out)",
        "num_test_samples": len(test_df),
        "class_labels": class_labels,
        "best_model_by_macro_f1": best_model_name,
        "metrics": all_metrics
    }

    # Save to disk
    with open(MODEL_EVAL_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    with open(MODEL_EVAL_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_labels, "matrices": all_cms}, f, indent=2)

    with open(MODEL_EVAL_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"total_disagreement_or_error_cases": len(error_cases), "cases": error_cases}, f, indent=2)

    return results_payload
