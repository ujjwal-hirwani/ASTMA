"""
CLI Script to evaluate individual classifiers and weighted ensemble on held-out test split.
Usage:
    python scripts/evaluate_models.py
"""

import sys
from pathlib import Path

# Safe stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluator import run_evaluation_suite

def main():
    print("[INFO] Initiating ASTMA Model Evaluation Pipeline on Held-out Test Split...")

    results = run_evaluation_suite()
    metrics = results["metrics"]

    print("\n[SUCCESS] Test Split Evaluation Complete!")
    print(f"       Dataset: {results['evaluation_dataset']} (N={results['num_test_samples']} test samples)")
    print(f"       Best Model by Macro-F1: {results['best_model_by_macro_f1']}")
    print("-" * 92)
    print(f"{'Model / Ensemble':<25} | {'Accuracy':<9} | {'Macro P':<9} | {'Macro R':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Inf/samp (ms)':<12}")
    print("-" * 92)
    for name, m in metrics.items():
        print(f"{name:<25} | {m['accuracy']:<9.4f} | {m['macro_precision']:<9.4f} | {m['macro_recall']:<9.4f} | {m['macro_f1']:<9.4f} | {m['weighted_f1']:<11.4f} | {m['inference_time_ms_per_sample']:<12.3f}")
    print("-" * 92)

    ens_metrics = metrics.get("weighted_ensemble", {})
    weights = ens_metrics.get("ensemble_weights", {})
    if weights:
        print("\n[INFO] Ensemble Weights (Derived from Validation Macro-F1):")
        for m_name, w in weights.items():
            print(f"       - {m_name:<25}: {w:.4f}")

if __name__ == "__main__":
    main()
