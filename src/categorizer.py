"""
Clinical Taxonomy Categorizer for Telehealth Symptoms.
Maps extracted symptom terms to standard medical body systems and urgency levels.
"""

# Major Clinical Body Systems and associated symptom keywords
TAXONOMY = {
    "Respiratory": [
        "cough", "dry cough", "wet cough", "productive cough", "wheezing", 
        "shortness of breath", "dyspnea", "breathing difficulty", "sore throat", 
        "throat pain", "nasal congestion", "runny nose", "sneezing", 
        "chest congestion", "phlegm", "stridor"
    ],
    "Gastrointestinal": [
        "nausea", "vomiting", "diarrhea", "abdominal pain", "stomach ache", 
        "stomach pain", "stomach cramps", "heartburn", "acid reflux", 
        "constipation", "bloating", "loss of appetite", "indigestion", 
        "cramping", "flatulence"
    ],
    "Neurological": [
        "headache", "migraine", "dizziness", "vertigo", "lightheadedness", 
        "numbness", "tingling", "confusion", "disorientation", "seizure", 
        "blurred vision", "memory loss", "fainting", "syncope", "tremor"
    ],
    "Cardiovascular": [
        "chest tightness", "chest pain", "palpitations", "rapid heartbeat", 
        "racing heart", "irregular heartbeat", "shortness of breath on exertion", 
        "ankle swelling", "leg swelling", "chest pressure"
    ],
    "Musculoskeletal": [
        "joint pain", "back pain", "lower back pain", "muscle pain", "muscle ache", 
        "knee pain", "shoulder pain", "neck pain", "joint stiffness", "swelling", 
        "muscle cramps", "muscle weakness", "joint inflammation"
    ],
    "Dermatological": [
        "rash", "skin itching", "hives", "redness", "skin redness", "eczema", 
        "lesion", "skin lesion", "blister", "skin peeling", "boil", "itching", 
        "dry skin", "hives", "swelling"
    ],
    "General / Systemic": [
        "fever", "high fever", "chills", "fatigue", "extreme tiredness", 
        "lethargy", "body ache", "generalized weakness", "night sweats", 
        "unexplained weight loss", "sweating", "malaise"
    ]
}

# High-Urgency Symptoms Flag Matrix
HIGH_URGENCY_SYMPTOMS = {
    "chest pain", "chest pressure", "shortness of breath", "difficulty breathing",
    "seizure", "syncope", "fainting", "high fever", "sudden numbness",
    "confusion", "disorientation", "coughing blood", "unbearable pain"
}

def categorize_symptom(symptom_text: str) -> str:
    """
    Categorizes a symptom string into a primary clinical body system.
    Returns 'General / Systemic' or 'Other / Uncategorized' if no match.
    """
    symptom_lower = symptom_text.lower().strip()
    
    # Direct keyword search against taxonomy
    for category, keywords in TAXONOMY.items():
        for kw in keywords:
            if kw in symptom_lower or symptom_lower in kw:
                return category
                
    return "General / Systemic"

def determine_urgency(symptom_text: str, severity: str = "Normal") -> str:
    """
    Determines triage urgency level (High, Medium, Low) based on symptom & severity.
    """
    symptom_lower = symptom_text.lower().strip()
    sev_lower = severity.lower().strip()
    
    if any(h in symptom_lower for h in HIGH_URGENCY_SYMPTOMS) or "severe" in sev_lower or "unbearable" in sev_lower:
        return "High"
    elif "moderate" in sev_lower:
        return "Medium"
    else:
        return "Low"
