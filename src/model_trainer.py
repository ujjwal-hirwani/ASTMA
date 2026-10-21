"""
Model Training Module for ASTMA.
Trains 4 supervised classifiers (Naive Bayes, Logistic Regression, Linear SVM with calibration, Random Forest)
using scikit-learn Pipelines on training data, validates on validation data, and persists artifacts.
Supports automatic training on arbitrary chosen or uploaded datasets.
"""

import time
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from config.settings import (
    SPLITS_DATA_DIR, MODEL_CATEGORY_DIR, MODEL_METADATA_DIR,
    MODEL_EVAL_DIR, RANDOM_SEED
)
from src.dataset_builder import build_feature_text
from src.ensemble import WeightedEnsembleClassifier, derive_ensemble_weights_from_val

MODEL_NAMES = [
    "multinomial_nb",
    "logistic_regression",
    "calibrated_linear_svm",
    "random_forest"
]

def build_model_pipelines(seed: int = RANDOM_SEED, min_df: int = 1) -> Dict[str, Any]:
    """
    Constructs scikit-learn Pipelines with TF-IDF text feature extraction
    and classifiers configured for multi-class clinical text categorization.
    """
    # 1. Multinomial Naive Bayes Pipeline
    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.5))
    ])

    # 2. Logistic Regression Pipeline
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=seed))
    ])

    # 3. Calibrated Linear SVM Pipeline
    base_svm = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("clf", calibrated_svm)
    ])

    # 4. Random Forest Pipeline
    rf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=4000)),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=25, class_weight="balanced", random_state=seed, n_jobs=-1))
    ])

    return {
        "multinomial_nb": nb_pipe,
        "logistic_regression": lr_pipe,
        "calibrated_linear_svm": svm_pipe,
        "random_forest": rf_pipe
    }

def train_all_models(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trains all four classifiers on train_df, evaluates validation metrics on val_df,
    records training durations, and returns fitted pipelines and training metrics.
    """
    X_train = train_df["feature_text"].values
    y_train = train_df["primary_category"].values

    X_val = val_df["feature_text"].values
    y_val = val_df["primary_category"].values

    min_df = 1 if len(train_df) < 80 else 2
    pipelines = build_model_pipelines(seed=RANDOM_SEED, min_df=min_df)
    trained_models = {}
    training_metrics = {}

    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0

        val_preds = pipe.predict(X_val)
        val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
        val_acc = float(accuracy_score(y_val, val_preds))

        trained_models[name] = pipe
        training_metrics[name] = {
            "training_time_seconds": round(fit_time, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "classes": list(pipe.classes_)
        }

    return trained_models, training_metrics

def save_model_artifacts(trained_models: Dict[str, Any], training_metrics: Dict[str, Any]) -> None:
    """Persists fitted model pipelines and metadata JSON files to disk."""
    MODEL_CATEGORY_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in trained_models.items():
        model_path = MODEL_CATEGORY_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)

    metadata = {
        "models": training_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED
    }
    with open(MODEL_METADATA_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def train_and_evaluate_on_dataframe(df_extracted: pd.DataFrame, test_ratio: float = 0.20, seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Trains the 4 classifiers and ensemble directly on the currently active / uploaded dataset.
    Automatically prepares features, creates a train/test split, fits pipelines, derives weights,
    computes test metrics, and persists artifacts.
    """
    if df_extracted.empty:
        raise ValueError("Cannot train models on empty extracted dataset.")

    # Prepare feature text and target category
    feature_texts = []
    categories = []

    for _, row in df_extracted.iterrows():
        cat = str(row.get("category", row.get("primary_category", ""))).strip()
        if not cat or cat == "N/A":
            continue

        if "feature_text" in row and pd.notna(row["feature_text"]):
            feat = str(row["feature_text"])
        else:
            raw_text = str(row.get("raw_text", row.get("message_text", row.get("text", ""))))
            start = int(row.get("char_start", 0))
            end = int(row.get("char_end", len(raw_text)))
            feat = build_feature_text(raw_text, start, end)

        feature_texts.append(feat)
        categories.append(cat)

    if len(feature_texts) < 6:
        raise ValueError(f"Insufficient training examples (found {len(feature_texts)}, need at least 6).")

    X = np.array(feature_texts)
    y = np.array(categories)

    # Train / Test Split
    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=seed, stratify=stratify
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=seed, stratify=None
        )

    # Build and fit pipelines
    min_df = 1 if len(X_train) < 60 else 2
    pipelines = build_model_pipelines(seed=seed, min_df=min_df)

    trained_models = {}
    training_metrics = {}

    for name, pipe in pipelines.items():
        t0 = time.time()
        # Fallback for SVM calibration if class frequencies are too small for 3-fold CV
        if name == "calibrated_linear_svm" and min(pd.Series(y_train).value_counts()) < 3:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
                ("clf", CalibratedClassifierCV(estimator=LinearSVC(C=1.0, max_iter=2000, random_state=seed), cv=2))
            ])

        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0

        test_preds = pipe.predict(X_test)
        test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))
        test_acc = float(accuracy_score(y_test, test_preds))

        trained_models[name] = pipe
        training_metrics[name] = {
            "training_time_seconds": round(fit_time, 4),
            "val_accuracy": round(test_acc, 4),
            "val_macro_f1": round(test_macro_f1, 4),
            "classes": list(pipe.classes_)
        }

    # Derive ensemble weights and instantiate ensemble
    ens_weights = derive_ensemble_weights_from_val(training_metrics)
    ensemble = WeightedEnsembleClassifier(models=trained_models, weights=ens_weights)

    class_labels = list(ensemble.classes_)
    all_metrics = {}
    all_cms = {}

    # Evaluate individual models on X_test
    for name, model in trained_models.items():
        t0 = time.time()
        y_pred = model.predict(X_test)
        inf_time = time.time() - t0

        acc = float(accuracy_score(y_test, y_pred))
        macro_p = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=class_labels).tolist()
        per_class_rep = classification_report(y_test, y_pred, labels=class_labels, output_dict=True, zero_division=0)

        all_metrics[name] = {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "training_time_seconds": training_metrics[name]["training_time_seconds"],
            "inference_time_ms_per_sample": round((inf_time / len(X_test)) * 1000.0, 3) if len(X_test) > 0 else 0.0,
            "per_class": {
                cls: {
                    "precision": round(per_class_rep[cls]["precision"], 4),
                    "recall": round(per_class_rep[cls]["recall"], 4),
                    "f1_score": round(per_class_rep[cls]["f1-score"], 4),
                    "support": int(per_class_rep[cls]["support"])
                }
                for cls in class_labels if cls in per_class_rep
            }
        }
        all_cms[name] = cm

    # Evaluate ensemble on X_test
    t0 = time.time()
    ens_pred = ensemble.predict(X_test)
    ens_inf_time = time.time() - t0

    ens_acc = float(accuracy_score(y_test, ens_pred))
    ens_mp = float(precision_score(y_test, ens_pred, average="macro", zero_division=0))
    ens_mr = float(recall_score(y_test, ens_pred, average="macro", zero_division=0))
    ens_mf1 = float(f1_score(y_test, ens_pred, average="macro", zero_division=0))
    ens_wf1 = float(f1_score(y_test, ens_pred, average="weighted", zero_division=0))
    ens_cm = confusion_matrix(y_test, ens_pred, labels=class_labels).tolist()

    all_metrics["weighted_ensemble"] = {
        "accuracy": round(ens_acc, 4),
        "macro_precision": round(ens_mp, 4),
        "macro_recall": round(ens_mr, 4),
        "macro_f1": round(ens_mf1, 4),
        "weighted_f1": round(ens_wf1, 4),
        "training_time_seconds": round(sum(m["training_time_seconds"] for m in all_metrics.values()), 4),
        "inference_time_ms_per_sample": round((ens_inf_time / len(X_test)) * 1000.0, 3) if len(X_test) > 0 else 0.0,
        "ensemble_weights": ens_weights
    }
    all_cms["weighted_ensemble"] = ens_cm

    # Persist active models and evaluation files
    save_model_artifacts(trained_models, training_metrics)
    ensemble.save_configuration()

    best_model_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]["macro_f1"])

    results_payload = {
        "evaluation_dataset": "Active Telehealth Dataset Split",
        "num_test_samples": len(X_test),
        "class_labels": class_labels,
        "best_model_by_macro_f1": best_model_name,
        "metrics": all_metrics
    }

    MODEL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_EVAL_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    with open(MODEL_EVAL_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_labels, "matrices": all_cms}, f, indent=2)

    return {
        "models": trained_models,
        "ensemble": ensemble,
        "metrics_payload": results_payload,
        "cms_payload": {"classes": class_labels, "matrices": all_cms},
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

def run_training_pipeline() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads split data, trains all models, and saves artifacts."""
    train_path = SPLITS_DATA_DIR / "train.csv"
    val_path = SPLITS_DATA_DIR / "val.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing splits in {SPLITS_DATA_DIR}. Run dataset generation first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    trained_models, training_metrics = train_all_models(train_df, val_df)
    save_model_artifacts(trained_models, training_metrics)
    return trained_models, training_metrics
