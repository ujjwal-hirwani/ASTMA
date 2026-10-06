"""
spaCy NLP Text Mining Engine for Telehealth Symptom Extraction & Negation Analysis.
Handles entity recognition (Symptoms, Body Parts, Severity, Duration) and dependency-based negation parsing.
"""

import spacy
from spacy.pipeline import EntityRuler
import re
from typing import List, Dict, Any
from src.categorizer import categorize_symptom, determine_urgency

# Global NLP pipeline cache
_nlp_model = None

NEGATION_TRIGGERS = {
    "no", "not", "none", "neither", "never", "without", "denies", "deny",
    "denied", "haven't", "hasn't", "don't", "doesn't", "didn't", "free of",
    "negative for", "ruled out", "no signs of"
}

SEVERITY_KEYWORDS = ["mild", "slight", "moderate", "severe", "intense", "unbearable", "extreme", "sharp", "dull"]
DURATION_PATTERNS = [r"\b\d+\s*(?:day|days|week|weeks|month|months|hour|hours)\b", r"\bsince\s+\w+\b"]

def load_nlp_pipeline():
    """
    Loads spaCy pipeline and injects custom clinical EntityRuler rules.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Fallback to blank model if en_core_web_sm fails
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    # Add entity ruler if not present
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
    else:
        ruler = nlp.get_pipe("entity_ruler")

    # Add clinical patterns
    patterns = [
        # Symptoms
        {"label": "SYMPTOM", "pattern": "headache"},
        {"label": "SYMPTOM", "pattern": "fever"},
        {"label": "SYMPTOM", "pattern": "high fever"},
        {"label": "SYMPTOM", "pattern": "cough"},
        {"label": "SYMPTOM", "pattern": "dry cough"},
        {"label": "SYMPTOM", "pattern": "wet cough"},
        {"label": "SYMPTOM", "pattern": "shortness of breath"},
        {"label": "SYMPTOM", "pattern": "breathing difficulty"},
        {"label": "SYMPTOM", "pattern": "chest tightness"},
        {"label": "SYMPTOM", "pattern": "chest pain"},
        {"label": "SYMPTOM", "pattern": "nausea"},
        {"label": "SYMPTOM", "pattern": "vomiting"},
        {"label": "SYMPTOM", "pattern": "diarrhea"},
        {"label": "SYMPTOM", "pattern": "abdominal pain"},
        {"label": "SYMPTOM", "pattern": "stomach pain"},
        {"label": "SYMPTOM", "pattern": "dizziness"},
        {"label": "SYMPTOM", "pattern": "vertigo"},
        {"label": "SYMPTOM", "pattern": "rash"},
        {"label": "SYMPTOM", "pattern": "skin itching"},
        {"label": "SYMPTOM", "pattern": "joint pain"},
        {"label": "SYMPTOM", "pattern": "back pain"},
        {"label": "SYMPTOM", "pattern": "muscle ache"},
        {"label": "SYMPTOM", "pattern": "fatigue"},
        {"label": "SYMPTOM", "pattern": "chills"},
        {"label": "SYMPTOM", "pattern": "sore throat"},
        {"label": "SYMPTOM", "pattern": "runny nose"},
        {"label": "SYMPTOM", "pattern": "sneezing"},
        {"label": "SYMPTOM", "pattern": "heartburn"},
        {"label": "SYMPTOM", "pattern": "constipation"},
        {"label": "SYMPTOM", "pattern": "bloating"},
        {"label": "SYMPTOM", "pattern": "palpitations"},

        # Severity
        {"label": "SEVERITY", "pattern": "mild"},
        {"label": "SEVERITY", "pattern": "moderate"},
        {"label": "SEVERITY", "pattern": "severe"},
        {"label": "SEVERITY", "pattern": "unbearable"},
        {"label": "SEVERITY", "pattern": "sharp"},

        # Body Parts
        {"label": "BODY_PART", "pattern": "head"},
        {"label": "BODY_PART", "pattern": "chest"},
        {"label": "BODY_PART", "pattern": "stomach"},
        {"label": "BODY_PART", "pattern": "abdomen"},
        {"label": "BODY_PART", "pattern": "throat"},
        {"label": "BODY_PART", "pattern": "knee"},
        {"label": "BODY_PART", "pattern": "back"},
        {"label": "BODY_PART", "pattern": "skin"},
        {"label": "BODY_PART", "pattern": "arm"},
        {"label": "BODY_PART", "pattern": "leg"}
    ]

    ruler.add_patterns(patterns)
    _nlp_model = nlp
    return _nlp_model

def is_symptom_negated(doc, ent) -> bool:
    """
    Determines if a symptom entity is negated using dependency parsing & surrounding window text.
    """
    # 1. Check dependency tree children/ancestors for negation modifier
    for token in ent:
        if token.dep_ == "neg" or token.lower_ in NEGATION_TRIGGERS:
            return True
        for child in token.children:
            if child.dep_ == "neg" or child.lower_ in NEGATION_TRIGGERS:
                return True
        for ancestor in token.ancestors:
            if ancestor.lower_ in NEGATION_TRIGGERS:
                return True

    # 2. Window-based fallback check (within 4 words preceding entity)
    start_idx = max(0, ent.start - 4)
    preceding_tokens = [t.lower_ for t in doc[start_idx:ent.start]]
    if any(neg in preceding_tokens for neg in NEGATION_TRIGGERS):
        return True

    return False

def extract_modifiers(doc, ent):
    """
    Extracts associated severity and duration for a given symptom entity.
    """
    severity = "Normal / Unspecified"
    duration = "Unspecified"

    # Search window around entity
    window_start = max(0, ent.start - 3)
    window_end = min(len(doc), ent.end + 4)
    window_text = doc[window_start:window_end].text.lower()

    # Match severity
    for sev in SEVERITY_KEYWORDS:
        if sev in window_text:
            severity = sev.capitalize()
            break

    # Match duration using regex pattern
    for pat in DURATION_PATTERNS:
        match = re.search(pat, window_text)
        if match:
            duration = match.group(0)
            break

    return severity, duration

def analyze_chat_message(text: str, chat_id: str = "N/A", timestamp: str = "N/A", speaker: str = "patient") -> List[Dict[str, Any]]:
    """
    Processes a single chat message and extracts structured symptom records.
    """
    nlp = load_nlp_pipeline()
    doc = nlp(text)
    
    extracted_records = []
    
    # Process entities
    for ent in doc.ents:
        if ent.label_ == "SYMPTOM":
            is_neg = is_symptom_negated(doc, ent)
            severity, duration = extract_modifiers(doc, ent)
            category = categorize_symptom(ent.text)
            urgency = determine_urgency(ent.text, severity)

            record = {
                "chat_id": chat_id,
                "timestamp": timestamp,
                "speaker": speaker,
                "raw_text": text,
                "symptom": ent.text.title(),
                "category": category,
                "is_negated": is_neg,
                "status": "Negated" if is_neg else "Affirmative",
                "severity": severity,
                "duration": duration,
                "urgency": urgency,
                "char_start": ent.start_char,
                "char_end": ent.end_char
            }
            extracted_records.append(record)

    # Fallback heuristic if no spaCy entity matched but keywords exist
    if not extracted_records:
        from src.categorizer import TAXONOMY
        text_lower = text.lower()
        for cat, keywords in TAXONOMY.items():
            for kw in keywords:
                if kw in text_lower:
                    is_neg = any(neg in text_lower for neg in ["no ", "not ", "denies ", "without ", "haven't "])
                    urgency = determine_urgency(kw, "Normal")
                    record = {
                        "chat_id": chat_id,
                        "timestamp": timestamp,
                        "speaker": speaker,
                        "raw_text": text,
                        "symptom": kw.title(),
                        "category": cat,
                        "is_negated": is_neg,
                        "status": "Negated" if is_neg else "Affirmative",
                        "severity": "Unspecified",
                        "duration": "Unspecified",
                        "urgency": urgency,
                        "char_start": text_lower.find(kw),
                        "char_end": text_lower.find(kw) + len(kw)
                    }
                    extracted_records.append(record)
                    break

    return extracted_records
