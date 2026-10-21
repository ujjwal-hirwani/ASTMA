"""
CLI Script to train all 4 supervised classifiers on training split and save models.
Usage:
    python scripts/train_models.py
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

from src.model_trainer import run_training_pipeline

def main():
    print("[INFO] Initiating ASTMA Model Training Pipeline...")
    print("       Models to train: Multinomial Naive Bayes, Logistic Regression, Calibrated Linear SVM, Random Forest")

    trained_models, training_metrics = run_training_pipeline()

    print("\n[SUCCESS] Model Training Complete! Saved to models/symptom_category/")
    print("-" * 75)
    print(f"{'Model Name':<25} | {'Fit Time (s)':<12} | {'Val Accuracy':<12} | {'Val Macro-F1':<12}")
    print("-" * 75)
    for name, metrics in training_metrics.items():
        print(f"{name:<25} | {metrics['training_time_seconds']:<12.3f} | {metrics['val_accuracy']:<12.4f} | {metrics['val_macro_f1']:<12.4f}")
    print("-" * 75)

if __name__ == "__main__":
    main()
