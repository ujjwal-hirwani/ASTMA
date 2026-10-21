"""
Synthetic Telehealth Chat and Clinical Dataset Generator.
Generates realistic, diverse multi-turn patient-clinician conversations with
ground-truth character offsets, negation, severity, duration, and clinical taxonomy labels.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import pandas as pd
from config.settings import TAXONOMY_10, RANDOM_SEED

GENERATOR_VERSION = "2.0.0"

# Category-specific symptoms, synonyms, and conversational variations
CATEGORY_SYMPTOM_VARIANTS = {
    "Neurological": [
        ("headache", "headache", ["headache", "head ache", "severe throbbing in my head", "bad head pain", "migraine headache"]),
        ("migraine", "migraine", ["migraine", "blinding migraine", "migraine aura", "pounding migraine"]),
        ("dizziness", "dizziness", ["dizziness", "feeling dizzy", "lightheadedness", "woozy feeling"]),
        ("vertigo", "vertigo", ["vertigo", "room spinning sensation", "loss of balance"]),
        ("numbness", "numbness", ["numbness", "loss of feeling", "numbness in my fingers", "numbness in toes"]),
        ("tingling", "tingling", ["tingling", "pins and needles", "tingling sensation"]),
        ("confusion", "confusion", ["confusion", "mental fog", "trouble concentrating", "disorientation"]),
        ("seizure", "seizure", ["seizure", "convulsions", "episodes of shaking"]),
        ("blurred vision", "blurred vision", ["blurred vision", "fuzzy vision", "blurry eyesight"]),
        ("tremor", "tremor", ["tremor", "shaking hands", "muscle tremor"])
    ],
    "Respiratory": [
        ("cough", "cough", ["cough", "persistent cough", "coughing fits", "hacking cough"]),
        ("dry cough", "dry cough", ["dry cough", "ticklish dry cough", "non-productive cough"]),
        ("wet cough", "wet cough", ["wet cough", "productive cough", "cough with phlegm", "chest cough"]),
        ("wheezing", "wheezing", ["wheezing", "whistling sound when breathing", "wheezy chest"]),
        ("shortness of breath", "shortness of breath", ["shortness of breath", "difficulty breathing", "trouble breathing", "winded easily", "breathlessness"]),
        ("chest congestion", "chest congestion", ["chest congestion", "heavy feeling in chest", "clogged chest"]),
        ("phlegm", "phlegm", ["excessive phlegm", "mucus buildup", "thick phlegm"]),
        ("shallow breathing", "shallow breathing", ["shallow breathing", "rapid shallow breathing"])
    ],
    "Cardiovascular": [
        ("chest pain", "chest pain", ["chest pain", "pain across my chest", "aching chest", "sharp chest pain"]),
        ("chest tightness", "chest tightness", ["chest tightness", "tight band around chest", "heavy chest pressure"]),
        ("palpitations", "palpitations", ["palpitations", "heart fluttering", "fluttering in chest", "skipped heartbeats"]),
        ("rapid heartbeat", "rapid heartbeat", ["rapid heartbeat", "racing heart", "heart pounding in chest", "fast pulse"]),
        ("irregular heartbeat", "irregular heartbeat", ["irregular heartbeat", "erratic heart rate", "irregular pulse"]),
        ("ankle swelling", "ankle swelling", ["ankle swelling", "swollen ankles", "puffy lower legs"]),
        ("chest pressure", "chest pressure", ["chest pressure", "crushing chest feeling", "intense pressure in chest"])
    ],
    "Gastrointestinal": [
        ("nausea", "nausea", ["nausea", "feeling nauseous", "queasy stomach", "sick to my stomach"]),
        ("vomiting", "vomiting", ["vomiting", "throwing up", "episodes of throwing up"]),
        ("diarrhea", "diarrhea", ["diarrhea", "loose watery stools", "frequent watery bowel movements"]),
        ("abdominal pain", "abdominal pain", ["abdominal pain", "belly ache", "stomach cramps", "sharp stomach pain", "tummy ache"]),
        ("heartburn", "heartburn", ["heartburn", "burning in my esophagus", "acid reflux", "sour taste and reflux"]),
        ("constipation", "constipation", ["constipation", "difficulty passing stool", "hard infrequent stools"]),
        ("bloating", "bloating", ["bloating", "swollen tight stomach", "distended abdomen"]),
        ("indigestion", "indigestion", ["indigestion", "upset stomach after eating", "dyspepsia"])
    ],
    "Musculoskeletal": [
        ("joint pain", "joint pain", ["joint pain", "aching joints", "tender joints"]),
        ("back pain", "back pain", ["back pain", "lower back pain", "achy lower back", "stiff lumbar spine"]),
        ("knee pain", "knee pain", ["knee pain", "aching in my right knee", "sharp pain in both knees"]),
        ("shoulder pain", "shoulder pain", ["shoulder pain", "stiff aching shoulder", "rotator cuff soreness"]),
        ("neck pain", "neck pain", ["neck pain", "stiff neck", "cervical spine stiffness"]),
        ("joint stiffness", "joint stiffness", ["joint stiffness", "morning stiffness in joints", "rigid joints"]),
        ("muscle cramps", "muscle cramps", ["muscle cramps", "painful spasms in legs", "charley horse cramps"]),
        ("muscle weakness", "muscle weakness", ["muscle weakness", "weak limbs", "loss of muscle strength"])
    ],
    "Dermatological": [
        ("rash", "rash", ["rash", "red bumpy rash", "skin breakout", "raised rash"]),
        ("skin itching", "skin itching", ["skin itching", "intense itching", "pruritus", "scratching constantly"]),
        ("hives", "hives", ["hives", "welts on skin", "urticaria flare-up"]),
        ("skin redness", "skin redness", ["skin redness", "inflamed red skin", "erythema"]),
        ("eczema", "eczema", ["eczema", "dry itchy eczema patches", "eczema flare"]),
        ("blister", "blister", ["blister", "fluid-filled blisters", "skin blisters"]),
        ("skin peeling", "skin peeling", ["skin peeling", "flaking skin", "peeling epidermis"])
    ],
    "Urinary": [
        ("painful urination", "painful urination", ["painful urination", "burning when peeing", "dysuria", "stinging during urination"]),
        ("frequent urination", "frequent urination", ["frequent urination", "peeing constantly", "urinating every hour"]),
        ("urinary urgency", "urinary urgency", ["urinary urgency", "sudden intense urge to pee", "bladder urgency"]),
        ("blood in urine", "blood in urine", ["blood in urine", "pink tinged urine", "hematuria"]),
        ("cloudy urine", "cloudy urine", ["cloudy urine", "murky dark urine", "foul smelling urine"]),
        ("urinary retention", "urinary retention", ["urinary retention", "trouble emptying bladder", "weak urine stream"])
    ],
    "Reproductive": [
        ("menstrual cramps", "menstrual cramps", ["menstrual cramps", "severe period cramps", "dysmenorrhea", "lower uterine cramping"]),
        ("pelvic pain", "pelvic pain", ["pelvic pain", "deep pelvic ache", "lower pelvic pressure"]),
        ("abnormal bleeding", "abnormal bleeding", ["abnormal bleeding", "spotting between cycles", "unusual vaginal bleeding"]),
        ("heavy periods", "heavy periods", ["heavy periods", "excessive menstrual bleeding", "heavy flow"]),
        ("testicular pain", "testicular pain", ["testicular pain", "scrotal tenderness", "aching in testicles"]),
        ("vaginal discharge", "vaginal discharge", ["vaginal discharge", "unusual discharge with odor"])
    ],
    "ENT": [
        ("sore throat", "sore throat", ["sore throat", "painful scratchy throat", "raw throat", "throat irritation"]),
        ("earache", "earache", ["earache", "ear pain", "sharp pain in left ear", "throbbing ear pain"]),
        ("nasal congestion", "nasal congestion", ["nasal congestion", "stuffy nose", "blocked nasal passages"]),
        ("runny nose", "runny nose", ["runny nose", "constant nasal drip", "rhinorrhea"]),
        ("tinnitus", "tinnitus", ["tinnitus", "ringing in ears", "high pitched buzzing in ears"]),
        ("sinus pressure", "sinus pressure", ["sinus pressure", "facial fullness around forehead", "pain behind eyes"]),
        ("hoarseness", "hoarseness", ["hoarseness", "raspy voice", "lost my voice"])
    ],
    "General / Systemic": [
        ("fever", "fever", ["fever", "running a temperature", "high body heat", "febrile feeling"]),
        ("high fever", "high fever", ["high fever", "fever of 102 degrees", "burning up with fever"]),
        ("chills", "chills", ["chills", "shivering and shaking", "severe rigors"]),
        ("fatigue", "fatigue", ["fatigue", "constant exhaustion", "feeling completely drained", "extreme tiredness"]),
        ("body ache", "body ache", ["body ache", "generalized body aches", "all-over aching", "flu-like body aches"]),
        ("night sweats", "night sweats", ["night sweats", "waking up drenched in sweat", "profuse sweating"]),
        ("malaise", "malaise", ["malaise", "general feeling of being unwell", "feeling sluggish and sick"])
    ]
}

# Linguistic frames and patterns
OPENERS = [
    "Hello doctor,", "Hi,", "Good morning,", "I'm contacting you because",
    "For the past few days,", "Over the last few days,", "I am writing because",
    "I've been experiencing", "Lately,", "Doctor, I wanted to report that",
    "I'm worried because", "I have noticed that", "I need advice regarding",
    "Since earlier this week,"
]

SEVERITIES = [
    ("mild", "mild"),
    ("slight", "mild"),
    ("moderate", "moderate"),
    ("severe", "severe"),
    ("intense", "severe"),
    ("unbearable", "severe"),
    ("sharp", "severe"),
    ("", "unknown")
]

DURATIONS = [
    ("for 2 days", 2, "days"),
    ("for 3 days", 3, "days"),
    ("for 5 days", 5, "days"),
    ("for 1 week", 1, "weeks"),
    ("for 2 weeks", 2, "weeks"),
    ("since yesterday", 1, "days"),
    ("since Monday", 3, "days"),
    ("for the past 24 hours", 24, "hours"),
    ("for about 4 hours", 4, "hours"),
    ("for over a month", 1, "months"),
    ("", None, "unknown")
]

ONSET_TYPES = ["acute", "gradual", "chronic", "unknown"]

CLINICIAN_REPLIES = [
    "Thank you for reaching out. How long have you had these symptoms?",
    "Understood. Are you currently taking any regular medications or supplements?",
    "I see. Let us review your health background and see if any other issues are present.",
    "Thank you for explaining. Please hydrate and avoid exertion while we evaluate.",
    "Got it. Have you noticed any other associated symptoms or changes recently?",
    "I appreciate the detailed notes. We will proceed with a structured assessment."
]

def _build_mention(norm_sym: str, raw_sym: str, cat: str, is_neg: bool,
                   sev_desc: str, sev_norm: str, dur_text: str, dur_val: Any,
                   dur_unit: str, onset: str) -> Dict[str, Any]:
    return {
        "normalized_symptom": norm_sym,
        "symptom_text": raw_sym,
        "primary_category": cat,
        "is_negated": is_neg,
        "severity": sev_norm,
        "severity_desc": sev_desc,
        "duration_text": dur_text,
        "duration_value": dur_val,
        "duration_unit": dur_unit,
        "onset_type": onset,
        "confidence_source": "generator_annotation"
    }

def generate_patient_statement(rng: random.Random) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Synthesizes a realistic patient text statement with 1 to 3 annotated symptoms,
    carefully tracking and verifying exact character offsets.
    Returns: (text, mentions, primary_category)
    """
    categories = list(CATEGORY_SYMPTOM_VARIANTS.keys())
    primary_category = rng.choice(categories)
    
    # Decide pattern type:
    # 0: Single symptom affirmative
    # 1: Single symptom with explicit duration and severity
    # 2: Single symptom negated (denial/absence)
    # 3: Compound: Affirmative symptom + Negated symptom
    # 4: Multi-symptom (2 affirmative symptoms from same or different categories)
    # 5: Multi-symptom (3 symptoms with varied severity/durations)
    # 6: Colloquial / inverted sentence structure
    pattern_type = rng.choices([0, 1, 2, 3, 4, 5, 6], weights=[15, 25, 15, 20, 15, 5, 5])[0]

    # Helper to pick a symptom variant
    def pick_sym(cat):
        tup = rng.choice(CATEGORY_SYMPTOM_VARIANTS[cat])
        norm = tup[1]
        raw = rng.choice(tup[2])
        return norm, raw

    # Helper to pick severity & duration
    def pick_modifiers():
        sev_desc, sev_norm = rng.choice(SEVERITIES)
        dur_text, dur_val, dur_unit = rng.choice(DURATIONS)
        onset = rng.choice(ONSET_TYPES)
        return sev_desc, sev_norm, dur_text, dur_val, dur_unit, onset

    mentions_data = []
    text_result = ""

    opener = rng.choice(OPENERS)

    if pattern_type in (0, 1):
        # Single symptom affirmative
        norm, raw = pick_sym(primary_category)
        sev_desc, sev_norm, dur_text, dur_val, dur_unit, onset = pick_modifiers()

        sev_prefix = f"{sev_desc} " if sev_desc else ""
        dur_suffix = f" {dur_text}" if dur_text else ""

        templates = [
            f"{opener} I have been suffering from {sev_prefix}{raw}{dur_suffix}.",
            f"{opener} I have {sev_prefix}{raw}{dur_suffix}.",
            f"{opener} my main complaint is {raw}, which feels {sev_norm.lower()}{dur_suffix}.",
            f"Doctor, I've had {sev_prefix}{raw}{dur_suffix} and wanted to check if this is normal.",
            f"I have been noticing {raw}{dur_suffix}. It is {sev_norm.lower()}."
        ]
        text_result = rng.choice(templates).strip()
        mentions_data.append(_build_mention(norm, raw, primary_category, False, sev_desc, sev_norm, dur_text, dur_val, dur_unit, onset))

    elif pattern_type == 2:
        # Single symptom negated
        norm, raw = pick_sym(primary_category)
        neg_phrases = [
            f"I do not have any {raw}.",
            f"{opener} I deny having {raw}.",
            f"No {raw} noted at this time.",
            f"I am completely free of {raw}.",
            f"{opener} there are no signs of {raw} currently."
        ]
        text_result = rng.choice(neg_phrases)
        mentions_data.append(_build_mention(norm, raw, primary_category, True, "", "unknown", "", None, "unknown", "unknown"))

    elif pattern_type == 3:
        # Compound: Affirmative + Negated
        norm1, raw1 = pick_sym(primary_category)
        sev_desc1, sev_norm1, dur_text1, dur_val1, dur_unit1, onset1 = pick_modifiers()
        sev_prefix1 = f"{sev_desc1} " if sev_desc1 else ""
        dur_suffix1 = f" {dur_text1}" if dur_text1 else ""

        cat2 = rng.choice([c for c in categories if c != primary_category])
        norm2, raw2 = pick_sym(cat2)

        compound_templates = [
            f"{opener} I have {sev_prefix1}{raw1}{dur_suffix1}, but definitely no {raw2}.",
            f"I am experiencing {sev_prefix1}{raw1}{dur_suffix1}. I deny any {raw2}.",
            f"No {raw2}, however I have been troubled by {raw1}{dur_suffix1}.",
            f"{opener} I noticed {sev_prefix1}{raw1}{dur_suffix1}, though I haven't had any {raw2}."
        ]
        text_result = rng.choice(compound_templates)
        mentions_data.append(_build_mention(norm1, raw1, primary_category, False, sev_desc1, sev_norm1, dur_text1, dur_val1, dur_unit1, onset1))
        mentions_data.append(_build_mention(norm2, raw2, cat2, True, "", "unknown", "", None, "unknown", "unknown"))

    elif pattern_type == 4:
        # Multi-symptom affirmative
        norm1, raw1 = pick_sym(primary_category)
        sev_desc1, sev_norm1, dur_text1, dur_val1, dur_unit1, onset1 = pick_modifiers()
        dur_suffix1 = f" {dur_text1}" if dur_text1 else ""

        cat2 = rng.choice(categories)
        norm2, raw2 = pick_sym(cat2)
        sev_desc2, sev_norm2, _, _, _, onset2 = pick_modifiers()

        templates = [
            f"{opener} I have {raw1} and {raw2}{dur_suffix1}.",
            f"I've been feeling {raw1}{dur_suffix1}, alongside moderate {raw2}.",
            f"{opener} both {raw1} and {raw2} have been bothering me{dur_suffix1}.",
            f"Experiencing {raw1} since yesterday, plus some {raw2} today."
        ]
        text_result = rng.choice(templates)
        mentions_data.append(_build_mention(norm1, raw1, primary_category, False, sev_desc1, sev_norm1, dur_text1, dur_val1, dur_unit1, onset1))
        mentions_data.append(_build_mention(norm2, raw2, cat2, False, sev_desc2, sev_norm2, "", None, "unknown", onset2))

    elif pattern_type == 5:
        # 3 symptoms
        norm1, raw1 = pick_sym(primary_category)
        cat2 = rng.choice([c for c in categories if c != primary_category])
        norm2, raw2 = pick_sym(cat2)
        cat3 = rng.choice([c for c in categories if c not in (primary_category, cat2)])
        norm3, raw3 = pick_sym(cat3)

        dur_text, dur_val, dur_unit = rng.choice(DURATIONS[:5])
        text_result = f"{opener} I am having {raw1} and {raw2} {dur_text}, but no {raw3}."
        mentions_data.append(_build_mention(norm1, raw1, primary_category, False, "moderate", "moderate", dur_text, dur_val, dur_unit, "acute"))
        mentions_data.append(_build_mention(norm2, raw2, cat2, False, "mild", "mild", dur_text, dur_val, dur_unit, "acute"))
        mentions_data.append(_build_mention(norm3, raw3, cat3, True, "", "unknown", "", None, "unknown", "unknown"))

    else:
        # Colloquial / inverted
        norm, raw = pick_sym(primary_category)
        sev_desc, sev_norm, dur_text, dur_val, dur_unit, onset = pick_modifiers()
        dur_suffix = f" {dur_text}" if dur_text else ""
        templates = [
            f"Really struggling with {raw}{dur_suffix}. Any suggestions?",
            f"My {raw} has gotten much worse{dur_suffix}.",
            f"Can't seem to shake this {raw}{dur_suffix}.",
            f"Woke up with {raw} and it feels pretty uncomfortable."
        ]
        text_result = rng.choice(templates)
        mentions_data.append(_build_mention(norm, raw, primary_category, False, sev_desc, sev_norm, dur_text, dur_val, dur_unit, onset))

    # Calculate exact character offsets and verify
    resolved_mentions = []
    for m in mentions_data:
        raw_phrase = m["symptom_text"]
        start_char = text_result.find(raw_phrase)
        if start_char != -1:
            end_char = start_char + len(raw_phrase)
            m_copy = dict(m)
            m_copy["start_char"] = start_char
            m_copy["end_char"] = end_char
            # Verify slice
            assert text_result[start_char:end_char] == raw_phrase
            resolved_mentions.append(m_copy)

    return text_result, resolved_mentions, primary_category

def generate_telehealth_corpus(num_conversations: int = 4000, seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates statement-level and mention-level DataFrames.
    Produces ~8,000 total records (pairs of patient turn + clinician response).
    Guarantees reproducible split_group_id per conversation to avoid split leakage.
    """
    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1, 8, 0, 0)

    statements = []
    mentions = []

    mention_counter = 1

    for conv_idx in range(1, num_conversations + 1):
        conv_id = f"CONV-{10000 + conv_idx}"
        split_group_id = f"GRP-{10000 + conv_idx}"
        patient_id = f"PAT-{5000 + (conv_idx % 800)}"
        conv_time = base_time + timedelta(minutes=conv_idx * 17)

        # 1. Patient Statement
        patient_text, patient_mentions, primary_cat = generate_patient_statement(rng)
        rec_id_pat = f"REC-{conv_id}-P"

        statements.append({
            "record_id": rec_id_pat,
            "conversation_id": conv_id,
            "patient_id": patient_id,
            "speaker": "patient",
            "text": patient_text,
            "timestamp": conv_time.strftime("%Y-%m-%d %H:%M:%S"),
            "split_group_id": split_group_id,
            "expected_primary_category": primary_cat,
            "is_synthetic": True,
            "generator_version": GENERATOR_VERSION
        })

        # Record mentions
        for m in patient_mentions:
            mentions.append({
                "mention_id": f"MEN-{mention_counter:07d}",
                "record_id": rec_id_pat,
                "conversation_id": conv_id,
                "symptom_text": m["symptom_text"],
                "normalized_symptom": m["normalized_symptom"],
                "start_char": m["start_char"],
                "end_char": m["end_char"],
                "primary_category": m["primary_category"],
                "is_negated": m["is_negated"],
                "severity": m["severity"],
                "duration_text": m["duration_text"],
                "duration_value": m["duration_value"],
                "duration_unit": m["duration_unit"],
                "onset_type": m["onset_type"],
                "confidence_source": m["confidence_source"]
            })
            mention_counter += 1

        # 2. Clinician Response
        rec_id_cli = f"REC-{conv_id}-C"
        cli_text = rng.choice(CLINICIAN_REPLIES)
        statements.append({
            "record_id": rec_id_cli,
            "conversation_id": conv_id,
            "patient_id": patient_id,
            "speaker": "clinician",
            "text": cli_text,
            "timestamp": (conv_time + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "split_group_id": split_group_id,
            "expected_primary_category": "N/A",
            "is_synthetic": True,
            "generator_version": GENERATOR_VERSION
        })

    df_statements = pd.DataFrame(statements)
    df_mentions = pd.DataFrame(mentions)

    return df_statements, df_mentions

def generate_synthetic_telehealth_data(num_chats: int = 50, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Backward-compatible generator returning a single DataFrame with chat turns
    for instant interactive demo mode in Streamlit.
    """
    df_stat, _ = generate_telehealth_corpus(num_conversations=num_chats, seed=seed)
    # Map column names for legacy compatibility
    df_legacy = df_stat.rename(columns={
        "text": "message_text",
        "speaker": "sender",
        "conversation_id": "chat_id"
    })
    return df_legacy

