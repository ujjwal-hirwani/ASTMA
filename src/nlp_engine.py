"""
spaCy NLP Text Mining Engine for Telehealth Symptom Extraction & Negation Analysis.
Handles entity recognition across 10 clinical body systems, dependency and context-based
negation parsing per mention, and symptom-modifier (duration, severity) association.
Optimized with batch processing via nlp.pipe for high throughput on large datasets.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
import spacy
from spacy.tokens import Doc, Span
from config.settings import (
    TAXONOMY_10, NEGATION_TRIGGERS, SEVERITY_KEYWORDS, HIGH_URGENCY_SYMPTOMS
)
from src.categorizer import categorize_symptom, determine_urgency

# Global cached spaCy model
_nlp_model = None

# Regex patterns for duration extraction and normalization
DURATION_REGEXES = [
    (r"\b(\d+)\s*(?:hour|hours|hr|hrs)\b", "hours"),
    (r"\b(\d+)\s*(?:day|days)\b", "days"),
    (r"\b(\d+)\s*(?:week|weeks|wk|wks)\b", "weeks"),
    (r"\b(\d+)\s*(?:month|months|mo|mos)\b", "months"),
    (r"\b(since\s+(?:yesterday|morning|last\s+night|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", "reference_point"),
    (r"\b(for\s+the\s+past\s+(?:\d+\s+)?(?:days|hours|weeks|months))\b", "relative_window")
]

def load_nlp_pipeline():
    """
    Loads spaCy pipeline and injects comprehensive clinical EntityRuler patterns
    covering all 10 categories and multiword expressions.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    # Add or update entity ruler
    if "entity_ruler" in nlp.pipe_names:
        ruler = nlp.get_pipe("entity_ruler")
    else:
        ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)

    # Build patterns from TAXONOMY_10 and clinical synonyms
    patterns = []
    seen_patterns = set()

    for category, keywords in TAXONOMY_10.items():
        for kw in keywords:
            kw_clean = kw.strip().lower()
            if kw_clean not in seen_patterns:
                seen_patterns.add(kw_clean)
                patterns.append({"label": "SYMPTOM", "pattern": kw_clean, "id": category})

    # Additional clinical multiword and informal variants
    extra_symptoms = [
        ("head ache", "Neurological"),
        ("throwing up", "Gastrointestinal"),
        ("tummy ache", "Gastrointestinal"),
        ("belly ache", "Gastrointestinal"),
        ("burning when peeing", "Urinary"),
        ("peeing hurts", "Urinary"),
        ("trouble breathing", "Respiratory"),
        ("room spinning", "Neurological"),
        ("period cramps", "Reproductive"),
        ("ringing in ears", "ENT"),
        ("pins and needles", "Neurological"),
        ("scratchy throat", "ENT"),
        ("stuffy nose", "ENT"),
        ("chills and shaking", "General / Systemic")
    ]
    for sym, cat in extra_symptoms:
        if sym not in seen_patterns:
            seen_patterns.add(sym)
            patterns.append({"label": "SYMPTOM", "pattern": sym, "id": cat})

    ruler.add_patterns(patterns)
    _nlp_model = nlp
    return _nlp_model

def normalize_text(text: str) -> str:
    """Cleans excess whitespace and normalizes basic punctuation without destroying syntax."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text

def is_mention_negated(doc: Doc, ent: Span) -> Tuple[bool, str]:
    """
    Determines if a specific symptom mention span is negated independently.
    """
    # 1. Dependency parse inspection (if parser is active)
    if doc.has_annotation("DEP"):
        for token in ent:
            if token.dep_ == "neg" or token.lower_ in NEGATION_TRIGGERS:
                return True, f"dependency_{token.lower_}"
            for child in token.children:
                if child.dep_ == "neg" or child.lower_ in NEGATION_TRIGGERS:
                    return True, f"dep_child_{child.lower_}"
            for ancestor in token.ancestors:
                if ancestor.lower_ in NEGATION_TRIGGERS:
                    return True, f"dep_ancestor_{ancestor.lower_}"

    # 2. Preceding clause window check (stop at clause boundaries)
    start_idx = max(0, ent.start - 4)
    preceding_tokens = []
    for t in reversed(doc[start_idx:ent.start]):
        if t.text in [".", ";", ",", "but", "however", "although", "except"]:
            break
        preceding_tokens.append(t.lower_)

    for neg in NEGATION_TRIGGERS:
        if neg in preceding_tokens:
            return True, f"clause_preceding_{neg}"

    # 3. Trailing negation check
    end_idx = min(len(doc), ent.end + 3)
    trailing_text = " ".join([t.lower_ for t in doc[ent.end:end_idx]])
    if any(phrase in trailing_text for phrase in ["resolved", "ruled out", "no longer", "is gone", "subsided"]):
        return True, "clause_trailing_resolution"

    return False, "none"

def extract_mention_modifiers(doc: Doc, ent: Span) -> Dict[str, Any]:
    """
    Extracts associated severity and duration for a specific symptom mention span.
    """
    severity = "unknown"
    duration_text = "unknown"
    duration_val = None
    duration_unit = "unknown"

    # Window around entity
    win_start = max(0, ent.start - 4)
    win_end = min(len(doc), ent.end + 5)
    window_span = doc[win_start:win_end]
    window_text = window_span.text.lower()

    # Severity matching
    for kw, norm_sev in SEVERITY_KEYWORDS.items():
        if re.search(rf"\b{kw}\b", window_text):
            severity = norm_sev.lower()
            break

    # Duration matching
    for regex_pat, unit in DURATION_REGEXES:
        m = re.search(regex_pat, window_text, re.IGNORECASE)
        if m:
            duration_text = m.group(0).strip()
            if unit in ["hours", "days", "weeks", "months"]:
                try:
                    duration_val = int(m.group(1))
                    duration_unit = unit
                except (IndexError, ValueError):
                    duration_unit = unit
            else:
                duration_unit = "reference_point"
            break

    return {
        "severity": severity,
        "duration_text": duration_text,
        "duration_value": duration_val,
        "duration_unit": duration_unit
    }

def _extract_records_from_doc(doc: Doc, raw_text: str, chat_id: str, timestamp: str, speaker: str) -> List[Dict[str, Any]]:
    """Helper to extract structured records from an annotated spaCy Doc."""
    records = []
    seen_spans = set()

    for ent in doc.ents:
        if ent.label_ == "SYMPTOM":
            span_key = (ent.start_char, ent.end_char)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            is_neg, neg_reason = is_mention_negated(doc, ent)
            mods = extract_mention_modifiers(doc, ent)
            category = ent.ent_id_ if ent.ent_id_ else categorize_symptom(ent.text)
            urgency = determine_urgency(ent.text, mods["severity"])

            norm_name = ent.text.strip().lower()

            record = {
                "chat_id": chat_id,
                "timestamp": timestamp,
                "speaker": speaker,
                "raw_text": raw_text,
                "symptom": ent.text.title(),
                "normalized_symptom": norm_name,
                "category": category,
                "is_negated": is_neg,
                "status": "Negated" if is_neg else "Affirmative",
                "negation_reason": neg_reason,
                "severity": mods["severity"].capitalize() if mods["severity"] != "unknown" else "Unknown",
                "duration": mods["duration_text"] if mods["duration_text"] != "unknown" else "Unspecified",
                "duration_value": mods["duration_value"],
                "duration_unit": mods["duration_unit"],
                "urgency": urgency,
                "char_start": ent.start_char,
                "char_end": ent.end_char
            }
            records.append(record)

    # Fallback heuristic if no entity matched
    if not records:
        text_lower = raw_text.lower()
        for cat, keywords in TAXONOMY_10.items():
            for kw in keywords:
                idx = text_lower.find(kw)
                if idx != -1:
                    span_key = (idx, idx + len(kw))
                    if span_key in seen_spans:
                        continue
                    seen_spans.add(span_key)

                    window_left = text_lower[max(0, idx - 25):idx]
                    is_neg = any(t in window_left for t in ["no ", "not ", "denies ", "without ", "never "])
                    urgency = determine_urgency(kw, "Normal")

                    record = {
                        "chat_id": chat_id,
                        "timestamp": timestamp,
                        "speaker": speaker,
                        "raw_text": raw_text,
                        "symptom": kw.title(),
                        "normalized_symptom": kw,
                        "category": cat,
                        "is_negated": is_neg,
                        "status": "Negated" if is_neg else "Affirmative",
                        "negation_reason": "fallback_lexicon",
                        "severity": "Unknown",
                        "duration": "Unspecified",
                        "duration_value": None,
                        "duration_unit": "unknown",
                        "urgency": urgency,
                        "char_start": idx,
                        "char_end": idx + len(kw)
                    }
                    records.append(record)
                    break

    return records

def analyze_chat_message(text: str, chat_id: str = "N/A", timestamp: str = "N/A", speaker: str = "patient") -> List[Dict[str, Any]]:
    """Analyzes a single chat message and extracts structured symptom records."""
    text = normalize_text(text)
    if not text:
        return []

    nlp = load_nlp_pipeline()
    doc = nlp(text)
    return _extract_records_from_doc(doc, text, chat_id, timestamp, speaker)

def analyze_chat_batch(items: List[Dict[str, str]], batch_size: int = 256) -> List[Dict[str, Any]]:
    """
    High-throughput batch processor using nlp.pipe.
    items: list of dicts with keys: 'text', 'chat_id', 'timestamp', 'speaker'.
    """
    if not items:
        return []

    nlp = load_nlp_pipeline()
    cleaned_texts = [normalize_text(it.get("text", "")) for it in items]

    records = []
    # Process with nlp.pipe
    for doc, item in zip(nlp.pipe(cleaned_texts, batch_size=batch_size), items):
        if not doc.text:
            continue
        recs = _extract_records_from_doc(
            doc,
            doc.text,
            chat_id=item.get("chat_id", "N/A"),
            timestamp=item.get("timestamp", "N/A"),
            speaker=item.get("speaker", "patient")
        )
        records.extend(recs)

    return records
