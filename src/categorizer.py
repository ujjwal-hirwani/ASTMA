"""
Clinical Taxonomy Categorizer for Telehealth Symptoms.
Maps extracted symptom terms to standard medical body systems and urgency levels.
Supports 10-category clinical taxonomy with backward-compatibility for legacy categories.
"""

from typing import Optional
from config.settings import TAXONOMY_10, HIGH_URGENCY_SYMPTOMS

# Maintain TAXONOMY reference for backward compatibility
TAXONOMY = TAXONOMY_10

def categorize_symptom(symptom_text: str) -> str:
    """
    Categorizes a symptom string into a primary clinical body system.
    Searches across TAXONOMY_10 and returns the matching category.
    Defaults to 'General / Systemic' if no specific category matches.
    """
    if not symptom_text:
        return "General / Systemic"

    symptom_lower = symptom_text.lower().strip()

    # Exact or substring keyword search against taxonomy
    # Prioritize longer keyword matches to avoid sub-token confusion (e.g. 'ear pain' before 'pain')
    matched_cat = None
    max_len = 0

    for category, keywords in TAXONOMY_10.items():
        for kw in keywords:
            if kw == symptom_lower:
                return category
            if kw in symptom_lower or symptom_lower in kw:
                if len(kw) > max_len:
                    max_len = len(kw)
                    matched_cat = category

    if matched_cat:
        return matched_cat

    return "General / Systemic"

def determine_urgency(symptom_text: str, severity: str = "Normal") -> str:
    """
    Determines triage urgency level (High, Medium, Low) based on symptom & severity.
    """
    if not symptom_text:
        return "Low"

    symptom_lower = symptom_text.lower().strip()
    sev_lower = (severity or "Normal").lower().strip()

    if any(h in symptom_lower for h in HIGH_URGENCY_SYMPTOMS) or "severe" in sev_lower or "unbearable" in sev_lower:
        return "High"
    elif "moderate" in sev_lower or "medium" in sev_lower or "intense" in sev_lower:
        return "Medium"
    else:
        return "Low"
