# 🩺 Automated Telehealth Symptom Text Mining Dashboard

> **ASTMA (Advanced Social & Text Media Analytics) Project**  
> **Topic**: *An Automated Text Mining Dashboard for Categorizing Patient Symptoms from Telehealth Chat Logs using spaCy and Streamlit*

---

## 📌 Project Overview
Telehealth chat platforms generate massive streams of unstructured patient communication. Manually reviewing chat transcripts to log patient symptoms, body systems, severity, and negations (e.g., distinguishing *"patient has fever"* from *"no fever"*) is labor-intensive and error-prone.

This application provides an **end-to-end automated text mining and analytics dashboard** built with **spaCy** for clinical Named Entity Recognition (NER) and dependency negation parsing, and **Streamlit** for interactive visualization and analytics.

---

## ✨ Key Features (MVP Capabilities)

1. **Multi-Source Data Ingestion**:
   - Drag-and-drop support for custom telehealth chat logs (CSV / JSON format).
   - Built-in **1-Click Synthetic Telehealth Dataset Generator** (50+ realistic multi-turn clinical chat sessions).

2. **spaCy Clinical Text Mining Engine**:
   - **Entity Extraction**: Recognizes Symptoms, Body Parts, Severity, and Duration.
   - **Dependency Negation Parsing**: Detects negative triggers (*"no fever"*, *"denies chest pain"*, *"without nausea"*) using syntactic dependency trees and context windows.
   - **Severity & Modifier Parser**: Extracts modifiers like *"severe"*, *"mild"*, *"3 days"*, *"since Monday"*.

3. **Clinical Taxonomy Categorization**:
   - Classifies extracted symptoms into **7 Clinical Body Systems**:
     - 🫁 **Respiratory** (Cough, Wheezing, Shortness of breath, Sore throat)
     - 🩺 **Gastrointestinal** (Nausea, Vomiting, Abdominal pain, Diarrhea)
     - 🧠 **Neurological** (Headache, Dizziness, Vertigo, Numbness)
     - 🫀 **Cardiovascular** (Chest tightness, Palpitations, Chest pain)
     - 🦴 **Musculoskeletal** (Joint pain, Back pain, Knee pain, Stiffness)
     - 🧴 **Dermatological** (Rash, Itching, Hives, Redness)
     - 🌡️ **General / Systemic** (Fever, Chills, Fatigue, Body ache)

4. **Triage Urgency & Risk Scoring**:
   - Automatically flags high-risk clinical symptoms (*chest pain*, *difficulty breathing*, *high fever*, *seizure*) for priority triage.

5. **Interactive Executive Analytics**:
   - **KPI Scorecards**: Active chats, total extracted symptoms, negation rate, high-urgency alerts.
   - **Plotly Visualizations**: Body system distribution donut chart, top 15 symptoms bar chart, affirmative vs. negated ratio chart, triage urgency stacked bar matrix.

6. **Symptom Co-occurrence Network**:
   - Interactive Pyvis / NetworkX node graph showing symptoms that frequently co-occur in the same chat sessions (e.g., *Fever + Cough*).

7. **Interactive Transcript Inspector**:
   - View formatted patient-clinician conversation timelines with extracted spaCy entity breakdowns per session.

8. **Structured Data Export & Report Generation**:
   - Download mined structured symptom tables in **CSV** and **JSON** formats.
   - Download automated **Executive Summary Clinical Reports** (`.txt`).

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **NLP & Text Mining** | spaCy (`en_core_web_sm` + custom `EntityRuler` & Dependency Parsing) |
| **Data Processing** | Pandas, NumPy |
| **Interactive Visualizations** | Plotly Express, Pyvis, NetworkX |
| **Language & Env** | Python 3.10+ |

---

## 🚀 Quick Start & How to Run

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed on your machine.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 📂 Project Architecture

```
ASTMA/
├── app.py                     # Main Streamlit dashboard application
├── requirements.txt           # Python dependencies
├── README.md                  # Detailed documentation
├── src/
│   ├── __init__.py
│   ├── categorizer.py         # Clinical body system taxonomy mapper & urgency scoring
│   ├── nlp_engine.py          # spaCy extraction engine & negation parser
│   ├── data_generator.py      # Synthetic telehealth dataset generator
│   └── visualizer.py          # Plotly charts & Pyvis co-occurrence network graph
└── data/
    └── sample_telehealth_chats.csv # Generated dataset
```

---

## 📜 License & Citation
Developed for **Advanced Social and Text Media Analytics (ASTMA)**.
