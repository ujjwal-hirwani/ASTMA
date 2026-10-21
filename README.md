# 🩺 ASTMA: Advanced Symptom Text Mining and Machine Learning Analytics

> **ASTMA (Advanced Social & Text Media Analytics) Project**  
> **Topic**: *A Hybrid Text Mining and Supervised Machine Learning System for Categorizing Patient Symptoms from Telehealth Chat Logs using spaCy, Scikit-Learn, and Streamlit*

---

## 📌 1. Project Overview

Telehealth and virtual consultation platforms process massive volumes of unstructured, informal patient conversations. Accurately extracting symptom mentions, identifying negated conditions (e.g., distinguishing *"patient has fever"* from *"no fever, denies chest pain"*), and routing symptoms to standard medical body systems is essential for healthcare operations and clinical text intelligence.

**ASTMA** transforms telehealth conversation mining into an end-to-end, reproducible NLP + Machine Learning system:
1. **Clinical NLP Pipeline**: Extracts clinical entities across 10 body systems, verifies exact character offsets, detects per-mention negation using dependency parsing, and extracts duration and severity modifiers.
2. **Supervised ML Classification**: Trains and compares 4 scikit-learn classifiers (**Multinomial Naive Bayes**, **Logistic Regression**, **Calibrated Linear Support Vector Machine**, and **Random Forest**) on TF-IDF representation of symptom mentions and surrounding conversational context.
3. **Validation-Weighted Ensemble**: Combines posterior probability distributions using weights strictly derived from held-out validation Macro-F1 scores:
   $$P_{\text{ensemble}}(c \mid x) = \sum_{i=1}^{M} w_i \cdot P_i(c \mid x) \quad \text{where} \quad w_i = \frac{\text{ValMacroF1}_i}{\sum_j \text{ValMacroF1}_j}$$
4. **Interactive Streamlit Dashboard**: Provides multi-view analytics:
   - **Executive Telehealth Analytics**: Dynamic filtering, KPI cards, Plotly charts, Pyvis co-occurrence network graphs, conversation timeline inspector, and CSV/JSON/PDF exports.
   - **Patient Query Analyzer**: Arbitrary free-text query input with multi-model prediction, confidence distributions, and uncertainty flags.
   - **ML Model Comparison**: Actual test-split metrics, confusion matrix heatmaps, per-class metrics, and error analysis.
   - **Dataset & Model Admin**: Local dataset generation, annotation validation, and one-click model retraining.

---

## 📂 2. System Architecture

```text
ASTMA/
├── app.py                         # Streamlit multi-view dashboard application
├── requirements.txt               # Project dependencies (Streamlit, spaCy, scikit-learn, etc.)
├── README.md                      # Comprehensive project documentation
├── config/
│   ├── __init__.py
│   └── settings.py                # 10-category taxonomy, paths, seeds, thresholds
├── data/
│   ├── raw/
│   │   ├── telehealth_statements.csv  # 8,000 multi-turn synthetic statements
│   │   └── telehealth_mentions.csv    # 5,771 verified clinical mentions
│   ├── processed/
│   │   └── symptom_classification_examples.csv # ML feature-text examples
│   ├── splits/
│   │   ├── train.csv              # 4,065 training samples (70% split)
│   │   ├── val.csv                # 830 validation samples (15% split)
│   │   ├── test.csv               # 876 test samples (15% split, held-out)
│   │   └── manifest.json          # Split statistics and class balance metadata
│   ├── sample_telehealth_chats.csv # Existing legacy sample data
│   ├── test_telehealth_chats.csv   # Existing legacy test data
│   └── test_telehealth_chats.json  # Existing legacy JSON data
├── models/
│   ├── symptom_category/          # Persisted joblib model pipelines
│   │   ├── multinomial_nb.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── calibrated_linear_svm.joblib
│   │   └── random_forest.joblib
│   ├── evaluation/                # Machine-readable test evaluation outputs
│   │   ├── test_metrics.json
│   │   ├── confusion_matrices.json
│   │   └── error_analysis.json
│   └── metadata/                  # Training metadata and ensemble weights
│       ├── training_summary.json
│       └── ensemble_config.json
├── src/
│   ├── __init__.py
│   ├── nlp_engine.py              # spaCy NER pipeline, negation parser & modifier extractor
│   ├── categorizer.py             # 10-category taxonomy rules & triage urgency
│   ├── data_generator.py          # Synthetic telehealth generator with offset tracking
│   ├── dataset_builder.py         # Leakage-free grouped train/val/test split builder
│   ├── data_validation.py         # Schemas, character offsets, and leakage validators
│   ├── model_trainer.py           # Pipeline definitions and multi-model trainer
│   ├── ensemble.py                # Validation-weighted probability ensemble
│   ├── predictor.py               # Unified inference engine for free-text queries
│   ├── evaluator.py               # Comprehensive test evaluator and error analyzer
│   ├── visualizer.py              # Plotly charts, confusion matrix heatmaps, Pyvis graph
│   └── reporting.py               # ASCII summary and ReportLab PDF generator
├── scripts/
│   ├── __init__.py
│   ├── generate_dataset.py        # CLI dataset generator
│   ├── train_models.py            # CLI model training pipeline
│   └── evaluate_models.py         # CLI test split evaluator
├── tests/
│   ├── __init__.py
│   ├── test_nlp_engine.py         # NLP extraction, negation scope, and offset tests
│   ├── test_dataset.py            # Dataset schemas, offsets, and split integrity tests
│   ├── test_training.py           # Pipeline construction and persistence tests
│   ├── test_predictor.py          # Query prediction, uncertainty, and fallback tests
│   ├── test_ensemble.py           # Ensemble weights and probability alignment tests
│   └── test_app_logic.py          # Visualizations and report generator tests
└── lib/                           # Offline Pyvis JS/CSS dependencies
```

---

## 🩺 3. 10-Category Clinical Taxonomy

| # | Clinical Body System | Key Symptoms and Clinical Synonyms |
| :--- | :--- | :--- |
| 1 | **Neurological** | Headache, migraine, dizziness, vertigo, numbness, tingling, confusion, seizure, blurred vision, tremor |
| 2 | **Respiratory** | Cough, dry cough, wet cough, wheezing, shortness of breath, dyspnea, chest congestion, phlegm |
| 3 | **Cardiovascular** | Chest pain, chest tightness, palpitations, rapid heartbeat, racing heart, irregular heartbeat, chest pressure |
| 4 | **Gastrointestinal** | Nausea, vomiting, diarrhea, abdominal pain, stomach ache, heartburn, acid reflux, constipation, bloating |
| 5 | **Musculoskeletal** | Joint pain, back pain, knee pain, shoulder pain, neck pain, joint stiffness, muscle cramps, muscle weakness |
| 6 | **Dermatological** | Rash, skin itching, hives, redness, eczema, blister, skin peeling, erythema |
| 7 | **Urinary** | Painful urination, dysuria, frequent urination, urinary urgency, blood in urine, cloudy urine, urinary retention |
| 8 | **Reproductive** | Menstrual cramps, dysmenorrhea, pelvic pain, abnormal bleeding, heavy periods, testicular pain, vaginal discharge |
| 9 | **ENT** | Sore throat, earache, nasal congestion, runny nose, tinnitus, sinus pressure, hoarseness |
| 10 | **General / Systemic** | Fever, high fever, chills, fatigue, body ache, night sweats, malaise, lethargy |

---

## 📊 4. Dataset Generation & Split Integrity

The dataset generator produces realistic, multi-turn clinical chat sessions with full linguistic variability (openers, severity descriptors, duration phrases, single-symptom, multi-symptom, colloquial language, and negations).

- **Total Statements**: 8,000 (4,000 patient turns + 4,000 clinician responses).
- **Total Clinical Mentions**: 5,771 verified mentions.
- **Split Strategy**: Grouped by `conversation_id` / `split_group_id` to guarantee **zero conversational leakage** across partitions:
  - **Train**: 4,065 mentions (70%)
  - **Validation**: 830 mentions (15%)
  - **Held-Out Test**: 876 mentions (15%)
- **Validation Suite**: 
  - Exact character offset matching: `text[start_char:end_char] == symptom_text`.
  - Schema consistency and non-null constraints.
  - Zero group overlap across partitions.

---

## 🤖 5. Machine Learning Models & Methodology

Each classifier is implemented as a self-contained scikit-learn `Pipeline`:
1. **Feature Extraction**: `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)` preserving mention text and local context.
2. **Supervised Classifiers**:
   - **Multinomial Naive Bayes** (`alpha=0.5`)
   - **Logistic Regression** (`C=2.0, max_iter=1000, class_weight='balanced'`)
   - **Calibrated Linear SVM** (`LinearSVC` calibrated with 3-fold cross-validation via `CalibratedClassifierCV`)
   - **Random Forest** (`n_estimators=150, max_depth=25, class_weight='balanced'`)
3. **Weighted Probability Ensemble**:
   - Validated weights derived from validation Macro-F1:
     - `Calibrated Linear SVM`: **0.2591**
     - `Logistic Regression`: **0.2585**
     - `Multinomial Naive Bayes`: **0.2433**
     - `Random Forest`: **0.2391**

---

## 📈 6. Quantitative Evaluation Results (Held-Out Test Split)

Evaluated strictly on the unseen held-out test split ($N = 876$ samples). Results are loaded dynamically from `models/evaluation/test_metrics.json`:

| Model / Ensemble | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Inference Time (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Calibrated Linear SVM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.042 ms** |
| 🥈 **Logistic Regression** | 0.9909 | 0.9911 | 0.9910 | 0.9910 | 0.9909 | 0.030 ms |
| 🥉 **Weighted Ensemble** | 0.9897 | 0.9903 | 0.9901 | 0.9901 | 0.9897 | 0.228 ms |
| **Multinomial Naive Bayes** | 0.9235 | 0.9267 | 0.9247 | 0.9243 | 0.9234 | 0.041 ms |
| **Random Forest** | 0.8950 | 0.9389 | 0.8904 | 0.9035 | 0.9047 | 0.113 ms |

> **Key Evaluation Insight**: The **Calibrated Linear SVM** achieved the highest generalization performance on the held-out test split with a Macro-F1 of 1.0000, closely followed by Logistic Regression and the Weighted Ensemble. Multinomial Naive Bayes and Random Forest perform well but exhibit minor overlap on short or ambiguous context tokens.

---

## 🚀 7. Installation & Setup Guide

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Generate Dataset, Train, and Evaluate
Run the automated pipeline CLI scripts:
```bash
# Step 1: Generate 8,000 synthetic telehealth records and leakage-free splits
python scripts/generate_dataset.py

# Step 2: Train all 4 classifiers and save models
python scripts/train_models.py

# Step 3: Evaluate models & ensemble on held-out test split
python scripts/evaluate_models.py
```

### 4. Run Automated Tests
```bash
python -m pytest -v
```
All 22 unit and integration tests will execute and verify the pipeline.

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 🧪 8. Custom Query Examples

You can test arbitrary free-form queries in the **Patient Query Analyzer** page:

1. **Compound Affirmative + Negated**:
   - *Query*: `"I've been getting severe headaches for the past week, but no fever."`
   - *Extraction*:
     - `Headaches`: Affirmative, Severity: Severe, Duration: for the past week, Predicted Category: **Neurological**
     - `Fever`: Negated, Severity: Unknown, Duration: Unspecified, Predicted Category: **General / Systemic**
2. **Cardiovascular & Respiratory Triage**:
   - *Query*: `"Sharp chest tightness and shortness of breath when climbing stairs. Denies chest pain."`
   - *Extraction*:
     - `Chest Tightness`: Affirmative, Urgency: High, Category: **Cardiovascular**
     - `Shortness of Breath`: Affirmative, Urgency: High, Category: **Respiratory**
     - `Chest Pain`: Negated, Urgency: High, Category: **Cardiovascular**
3. **Urinary Symptoms**:
   - *Query*: `"Experiencing painful urination and frequent urination for 2 days. No blood in urine."`
   - *Extraction*: All mapped to **Urinary** with per-mention negation detection.

---

## ⚠️ 9. Limitations & Responsible Use Disclaimer

- **Academic & Educational Text Mining**: This software is designed for academic research, text mining benchmarking, and operational dashboard demonstrations in telehealth analytics.
- **Synthetic Data Disclaimer**: All benchmark conversations were synthesized locally for reproducible NLP/ML training. Synthetic data verifies system mechanics, but does not substitute for real-world clinical validation.
- **No Diagnostic Claims**: This system does **NOT** provide medical advice, establish diagnoses, or replace professional clinician judgement. Model probabilities reflect statistical token distributions, not clinical disease probabilities.
