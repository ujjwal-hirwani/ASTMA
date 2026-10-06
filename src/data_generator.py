"""
Synthetic Telehealth Chat Dataset Generator.
Generates realistic, unstructured multi-turn patient-clinician chats for text mining and demonstration.
"""

import pandas as pd
import random
from datetime import datetime, timedelta

def generate_synthetic_telehealth_data(num_chats: int = 50) -> pd.DataFrame:
    """
    Generates realistic telehealth chat transcripts with varied patient symptoms,
    negations, durations, and clinical body systems.
    """
    random.seed(42)

    templates = [
        {"text": "Hi Doctor, I've had a severe headache and dizziness for 3 days. No fever though.", "cat": "Neurological"},
        {"text": "Hello, I am experiencing bad nausea and abdominal pain since yesterday. Haven't had any vomiting.", "cat": "Gastrointestinal"},
        {"text": "I'm having sharp chest tightness and shortness of breath when I walk. Denies chest pain.", "cat": "Cardiovascular"},
        {"text": "I noticed a red skin rash on my arms with intense itching for 2 days. No chills or fever.", "cat": "Dermatological"},
        {"text": "I've been coughing a lot with a sore throat and nasal congestion for 4 days. No shortness of breath.", "cat": "Respiratory"},
        {"text": "My knee pain has been unbearable since Monday, along with joint stiffness.", "cat": "Musculoskeletal"},
        {"text": "I have high fever, chills, and severe body ache since last night.", "cat": "General / Systemic"},
        {"text": "Lately I feel extreme fatigue and dizziness. No chest pain or headache.", "cat": "Neurological"},
        {"text": "Experiencing moderate heartburn and bloating after meals. No abdominal pain.", "cat": "Gastrointestinal"},
        {"text": "Dry cough and mild throat pain. Denies fever or chills.", "cat": "Respiratory"},
        {"text": "Lower back pain and muscle stiffness for 1 week. No numbness in legs.", "cat": "Musculoskeletal"},
        {"text": "Sudden hives and redness on neck. No difficulty breathing.", "cat": "Dermatological"},
        {"text": "Fever, night sweats, and severe muscle ache for 3 days.", "cat": "General / Systemic"},
        {"text": "Mild wheezing and chest congestion when laying down.", "cat": "Respiratory"},
        {"text": "Severe migraine and blurred vision for 5 hours. No vomiting.", "cat": "Neurological"},
        {"text": "Diarrhea and stomach cramps since morning. Denies nausea.", "cat": "Gastrointestinal"},
        {"text": "Palpitations and rapid heartbeat during work today. No chest pain.", "cat": "Cardiovascular"},
        {"text": "Itching rash and skin peeling on legs.", "cat": "Dermatological"},
        {"text": "Persistent dry cough for 2 weeks with mild shortness of breath.", "cat": "Respiratory"},
        {"text": "Sharp abdominal pain in upper right side. No fever.", "cat": "Gastrointestinal"}
    ]

    clinician_responses = [
        "Thank you for sharing. How long have these symptoms been present?",
        "I understand. Are you currently taking any prescription medications?",
        "Got it. Please stay rested and hydrated while we review your case.",
        "Thank you. Have you noticed any other symptoms recently?",
        "Understood. Let me check your medical history to advise further."
    ]

    rows = []
    base_time = datetime.now() - timedelta(days=7)

    for i in range(1, num_chats + 1):
        chat_id = f"CHAT-{1000 + i}"
        patient_id = f"PAT-{5000 + (i % 25)}"
        chat_time = base_time + timedelta(hours=i * 3, minutes=random.randint(5, 50))
        time_str = chat_time.strftime("%Y-%m-%d %H:%M")

        # Select template
        sample = random.choice(templates)
        
        # Turn 1: Patient message
        rows.append({
            "chat_id": chat_id,
            "patient_id": patient_id,
            "timestamp": time_str,
            "sender": "patient",
            "message_text": sample["text"],
            "expected_primary_category": sample["cat"]
        })

        # Turn 2: Clinician message
        rows.append({
            "chat_id": chat_id,
            "patient_id": patient_id,
            "timestamp": (chat_time + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M"),
            "sender": "clinician",
            "message_text": random.choice(clinician_responses),
            "expected_primary_category": "N/A"
        })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_telehealth_data(50)
    df.to_csv("data/sample_telehealth_chats.csv", index=False)
    print(f"Generated {len(df)} synthetic chat turns successfully!")

