"""
Configuration and settings for the ASTMA project.
Defines medical taxonomies, severity and urgency keywords, dataset schema, and file paths.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DATA_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
MODEL_CATEGORY_DIR = MODELS_DIR / "symptom_category"
MODEL_EVAL_DIR = MODELS_DIR / "evaluation"
MODEL_METADATA_DIR = MODELS_DIR / "metadata"

# Reproducibility
RANDOM_SEED = 42

# 10-Category Clinical Taxonomy
# Maps each clinical category to representative normalized symptoms and synonyms
TAXONOMY_10 = {
    "Neurological": [
        "headache", "migraine", "dizziness", "vertigo", "lightheadedness",
        "numbness", "tingling", "confusion", "disorientation", "seizure",
        "blurred vision", "memory loss", "fainting", "syncope", "tremor"
    ],
    "Respiratory": [
        "cough", "dry cough", "wet cough", "productive cough", "wheezing",
        "shortness of breath", "dyspnea", "breathing difficulty", "sore throat",
        "chest congestion", "phlegm", "stridor", "shallow breathing"
    ],
    "Cardiovascular": [
        "chest pain", "chest tightness", "palpitations", "rapid heartbeat",
        "racing heart", "irregular heartbeat", "chest pressure",
        "shortness of breath on exertion", "ankle swelling", "fluttering in chest"
    ],
    "Gastrointestinal": [
        "nausea", "vomiting", "diarrhea", "abdominal pain", "stomach ache",
        "stomach pain", "stomach cramps", "heartburn", "acid reflux",
        "constipation", "bloating", "loss of appetite", "indigestion", "flatulence"
    ],
    "Musculoskeletal": [
        "joint pain", "back pain", "lower back pain", "muscle pain", "muscle ache",
        "knee pain", "shoulder pain", "neck pain", "joint stiffness", "swelling",
        "muscle cramps", "muscle weakness", "joint inflammation", "wrist pain"
    ],
    "Dermatological": [
        "rash", "skin itching", "hives", "redness", "skin redness", "eczema",
        "skin lesion", "blister", "skin peeling", "boil", "itching", "dry skin",
        "urticaria", "flaking skin"
    ],
    "Urinary": [
        "painful urination", "dysuria", "frequent urination", "burning urination",
        "blood in urine", "hematuria", "urinary urgency", "cloudy urine",
        "urinary retention", "incontinence"
    ],
    "Reproductive": [
        "menstrual cramps", "dysmenorrhea", "pelvic pain", "abnormal bleeding",
        "vaginal discharge", "testicular pain", "heavy periods", "scrotal swelling",
        "breast tenderness", "irregular period"
    ],
    "ENT": [
        "sore throat", "throat pain", "earache", "ear pain", "nasal congestion",
        "runny nose", "sneezing", "tinnitus", "ringing in ears", "hearing loss",
        "loss of smell", "anosmia", "sinus pressure", "hoarseness"
    ],
    "General / Systemic": [
        "fever", "high fever", "chills", "fatigue", "extreme tiredness",
        "lethargy", "body ache", "generalized weakness", "night sweats",
        "sweating", "malaise", "unexplained weight loss", "exhaustion"
    ]
}

# High-Urgency Symptoms Flag Matrix
HIGH_URGENCY_SYMPTOMS = {
    "chest pain", "chest pressure", "shortness of breath", "difficulty breathing",
    "seizure", "syncope", "fainting", "high fever", "sudden numbness",
    "confusion", "disorientation", "coughing blood", "blood in urine",
    "unbearable pain", "stridor", "hematuria"
}

# Negation Triggers (syntactic & lexicon)
NEGATION_TRIGGERS = {
    "no", "not", "none", "neither", "never", "without", "denies", "deny",
    "denied", "haven't", "hasn't", "don't", "doesn't", "didn't", "free of",
    "negative for", "ruled out", "no signs of", "resolved", "no longer"
}

# Severity Keywords
SEVERITY_KEYWORDS = {
    "mild": "Mild",
    "slight": "Mild",
    "minor": "Mild",
    "moderate": "Moderate",
    "medium": "Moderate",
    "severe": "Severe",
    "intense": "Severe",
    "unbearable": "Severe",
    "extreme": "Severe",
    "sharp": "Severe",
    "acute": "Severe",
    "dull": "Mild"
}

# Category Color Mapping for Visualizations
CATEGORY_COLORS = {
    "Neurological": "#8b5cf6",
    "Respiratory": "#3b82f6",
    "Cardiovascular": "#ef4444",
    "Gastrointestinal": "#10b981",
    "Musculoskeletal": "#f59e0b",
    "Dermatological": "#ec4899",
    "Urinary": "#06b6d4",
    "Reproductive": "#a855f7",
    "ENT": "#14b8a6",
    "General / Systemic": "#6366f1"
}

# Generator & ML Defaults
DEFAULT_NUM_RECORDS = 8000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
