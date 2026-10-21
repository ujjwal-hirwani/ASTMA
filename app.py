"""
ASTMA: Automated Symptom Text Mining and Machine Learning Analytics Dashboard.
Integrated Telehealth Analytics, Patient Query Analysis, and ML Model Comparison.
Models are automatically trained by default once a dataset is uploaded.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Config and settings
from config.settings import (
    TAXONOMY_10, CATEGORY_COLORS, DATA_DIR,
    MODEL_EVAL_DIR, MODEL_METADATA_DIR, MODEL_CATEGORY_DIR
)

# Custom src modules
from src.nlp_engine import analyze_chat_message, analyze_chat_batch
from src.predictor import SymptomPredictor
from src.model_trainer import train_and_evaluate_on_dataframe
from src.visualizer import (
    plot_category_distribution,
    plot_top_symptoms,
    plot_negation_status,
    plot_urgency_breakdown,
    plot_model_comparison_bar,
    plot_confusion_matrix_heatmap,
    plot_confidence_distribution_bar
)
from src.reporting import generate_text_summary, generate_pdf_summary, HAS_REPORTLAB

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ASTMA - Clinical Text Mining & ML Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark CSS Styling
st.markdown("""
<style>
    /* Base typography & backgrounds */
    .stApp {
        background-color: #0b0f19;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #e2e8f0;
    }
    .stAppHeader {
        background-color: transparent;
    }
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    /* Headers & Text */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 600;
    }
    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.7);
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to cache NLP extraction
@st.cache_data(show_spinner="Extracting Symptoms & Running spaCy NLP Engine...")
def run_nlp_pipeline(df: pd.DataFrame, process_patient_only: bool = True):
    if process_patient_only and "sender" in df.columns:
        target_df = df[df["sender"].str.lower() == "patient"].copy()
    elif process_patient_only and "speaker" in df.columns:
        target_df = df[df["speaker"].str.lower() == "patient"].copy()
    else:
        target_df = df.copy()

    items = []
    for idx, row in target_df.iterrows():
        chat_id = str(row.get("chat_id", row.get("conversation_id", f"CHAT-{idx}")))
        timestamp = str(row.get("timestamp", "N/A"))
        speaker = str(row.get("sender", row.get("speaker", "patient")))
        text = str(row.get("message_text", row.get("text", "")))

        if text.strip():
            items.append({
                "chat_id": chat_id,
                "timestamp": timestamp,
                "speaker": speaker,
                "text": text
            })

    records = analyze_chat_batch(items, batch_size=256)
    return pd.DataFrame(records)


# Cache predictor instantiation
@st.cache_resource
def get_predictor():
    return SymptomPredictor()

# Automatic on-the-fly model training on uploaded dataset
@st.cache_resource(show_spinner="Training ML Classifiers & Ensemble on Uploaded Dataset...")
def get_trained_models_for_dataset(dataset_key: str, df_extracted: pd.DataFrame):
    """Trains the 4 classifiers and ensemble directly on the uploaded dataset."""
    try:
        return train_and_evaluate_on_dataframe(df_extracted)
    except Exception as e:
        return None

# Header Banner
st.markdown("""
<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 22px 28px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 22px;">
    <h1 style="margin:0; font-size: 2.1rem; color: #38bdf8;">🩺 ASTMA: Telehealth Text Mining & ML Analytics</h1>
    <p style="margin-top: 6px; margin-bottom:0; font-size: 1.0rem; color: #94a3b8;">
        Clinical NLP Text Mining & Machine Learning System — Automatically trained on uploaded dataset
    </p>
</div>
""", unsafe_allow_html=True)

# Main Navigation
st.sidebar.markdown("### 🧭 Navigation")
app_mode = st.sidebar.radio(
    "Select Platform View",
    [
        "📊 Telehealth Chat Analytics",
        "🔬 Patient Query Analyzer",
        "📈 ML Model Comparison"
    ],
    index=0
)

# Global Data Ingestion (Single upload option, empty by default)
st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload Telehealth Chat Logs (CSV / JSON)", type=["csv", "json"])

df_raw = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_json(uploaded_file)
        st.sidebar.success(f"Loaded {len(df_raw)} records from {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        df_raw = None

df_extracted = pd.DataFrame()
training_result = None
predictor = get_predictor()

# Process uploaded dataset
if df_raw is not None and not df_raw.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ NLP Processing Options")
    patient_only = st.sidebar.checkbox("Process Patient Messages Only", value=True)

    df_extracted = run_nlp_pipeline(df_raw, process_patient_only=patient_only)

    if not df_extracted.empty:
        dataset_name = uploaded_file.name if uploaded_file is not None else "uploaded_dataset"
        dataset_cache_key = f"{dataset_name}_{len(df_raw)}_{len(df_extracted)}"
        training_result = get_trained_models_for_dataset(dataset_cache_key, df_extracted)

        if training_result:
            predictor.update_models(training_result["models"], training_result["ensemble"])

# ==============================================================================
# VIEW 1: TELEHEALTH CHAT ANALYTICS (PAGE A)
# ==============================================================================
if app_mode == "📊 Telehealth Chat Analytics":
    if df_raw is not None and not df_raw.empty and not df_extracted.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Dynamic Filters")

        all_cats = list(df_extracted["category"].unique())
        sel_cats = st.sidebar.multiselect("Clinical Body System", options=all_cats, default=all_cats)

        sel_status = st.sidebar.selectbox("Negation Status", ["All", "Affirmative Only", "Negated Only"], index=0)
        sel_urgency = st.sidebar.selectbox("Triage Urgency Level", ["All", "High", "Medium", "Low"], index=0)

        # Apply Dynamic Filtering
        filtered_df = df_extracted[df_extracted["category"].isin(sel_cats)].copy()
        if sel_status == "Affirmative Only":
            filtered_df = filtered_df[filtered_df["is_negated"] == False]
        elif sel_status == "Negated Only":
            filtered_df = filtered_df[filtered_df["is_negated"] == True]

        if sel_urgency != "All":
            filtered_df = filtered_df[filtered_df["urgency"] == sel_urgency]

        # Top KPI Metrics Cards
        total_chats = df_raw["chat_id"].nunique() if "chat_id" in df_raw.columns else (
            df_raw["conversation_id"].nunique() if "conversation_id" in df_raw.columns else len(df_raw)
        )
        total_extracted = len(filtered_df)
        negated_count = int(filtered_df["is_negated"].sum())
        negated_pct = round((negated_count / total_extracted * 100), 1) if total_extracted > 0 else 0
        high_urgency = len(filtered_df[filtered_df["urgency"] == "High"])

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Consultations Analyzed", f"{total_chats}")
        kpi2.metric("Total Symptoms Extracted", f"{total_extracted}")
        kpi3.metric("Negation Rate", f"{negated_pct}%", f"{negated_count} Negated")
        kpi4.metric("High-Urgency Flagged", f"{high_urgency}", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tab Navigation for Analytics
        tab_analytics, tab_inspector, tab_table, tab_report = st.tabs([
            "📊 Executive Analytics",
            "🔍 Transcript Timeline Inspector",
            "📋 Extracted Dataset & Export",
            "📄 Automated Clinical Report"
        ])

        # Tab 1: Visuals
        with tab_analytics:
            col1, col2 = st.columns(2)
            with col1:
                fig_cat = plot_category_distribution(filtered_df)
                if fig_cat:
                    st.plotly_chart(fig_cat, use_container_width=True)
            with col2:
                fig_top = plot_top_symptoms(filtered_df, top_n=12)
                if fig_top:
                    st.plotly_chart(fig_top, use_container_width=True)

            st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

            col3, col4 = st.columns(2)
            with col3:
                fig_neg = plot_negation_status(filtered_df)
                if fig_neg:
                    st.plotly_chart(fig_neg, use_container_width=True)
            with col4:
                fig_urg = plot_urgency_breakdown(filtered_df)
                if fig_urg:
                    st.plotly_chart(fig_urg, use_container_width=True)

        # Tab 2: Transcript Inspector
        with tab_inspector:
            st.subheader("🔍 Interactive Transcript & Entity Breakdown")
            chat_col = "chat_id" if "chat_id" in df_raw.columns else "conversation_id"
            chat_list = df_raw[chat_col].dropna().unique() if chat_col in df_raw.columns else []

            if len(chat_list) > 0:
                selected_chat = st.selectbox("Select Consultation Session ID:", chat_list)
                chat_messages = df_raw[df_raw[chat_col] == selected_chat]

                st.markdown("### Chat Timeline")
                for _, msg_row in chat_messages.iterrows():
                    speaker = msg_row.get("sender", msg_row.get("speaker", "patient"))
                    text = msg_row.get("message_text", msg_row.get("text", ""))
                    time = msg_row.get("timestamp", "")

                    if str(speaker).lower() == "patient":
                        st.markdown(f"""
                        <div style="background: rgba(30, 58, 138, 0.4); border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
                            <strong style="color: #60a5fa;">👤 Patient ({time}):</strong><br>
                            <span style="font-size: 1.02rem; color: #f8fafc;">{text}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border-left: 4px solid #64748b; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px;">
                            <strong style="color: #cbd5e1;">🩺 Clinician ({time}):</strong><br>
                            <span style="font-size: 1.02rem; color: #e2e8f0;">{text}</span>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("### Extracted Entities for This Chat")
                chat_extracted = df_extracted[df_extracted["chat_id"] == selected_chat]
                if not chat_extracted.empty:
                    st.dataframe(
                        chat_extracted[["symptom", "category", "status", "severity", "duration", "urgency"]],
                        use_container_width=True
                    )
                else:
                    st.info("No specific symptom entities identified in this chat.")

        # Tab 3: Export
        with tab_table:
            st.subheader("📋 Structured Clinical Symptom Dataset")
            st.caption(f"Showing {len(filtered_df)} records after dynamic filtering.")
            st.dataframe(
                filtered_df[["chat_id", "timestamp", "symptom", "category", "status", "severity", "duration", "urgency", "raw_text"]],
                use_container_width=True,
                height=380
            )
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Extracted Dataset (CSV)", csv_bytes, "telehealth_extracted_symptoms.csv", "text/csv")
            with col_e2:
                json_bytes = filtered_df.to_json(orient="records", indent=2)
                st.download_button("📥 Export Extracted Dataset (JSON)", json_bytes, "telehealth_extracted_symptoms.json", "application/json")

        # Tab 4: Report Generation (ASCII + PDF)
        with tab_report:
            st.subheader("📄 Automated Executive Clinical Report")
            rep_text = generate_text_summary(filtered_df, total_chats=total_chats)
            st.code(rep_text, language="text")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.download_button("📥 Download Report (.txt)", rep_text, "telehealth_executive_report.txt", "text/plain")
            with col_r2:
                if HAS_REPORTLAB:
                    pdf_data = generate_pdf_summary(filtered_df, total_chats=total_chats)
                    st.download_button("📄 Download Formatted PDF Report", pdf_data, "telehealth_executive_report.pdf", "application/pdf")
                else:
                    st.info("ReportLab is not installed; PDF export disabled.")
    else:
        st.info("📂 **The dashboard is currently empty.** Please upload a telehealth chat logs dataset (CSV or JSON format) in the sidebar to begin analysis and train the ML models.")

# ==============================================================================
# VIEW 2: PATIENT QUERY ANALYZER (PAGE B)
# ==============================================================================
elif app_mode == "🔬 Patient Query Analyzer":
    st.markdown("### 🔬 Patient Query Analyzer")
    st.markdown("Enter arbitrary, free-form patient statements to extract clinical symptom mentions, determine negation and modifiers, and classify into clinical body systems using the ML models trained on your uploaded dataset.")

    if df_raw is None or df_raw.empty or training_result is None:
        st.info("📂 **No dataset uploaded yet.** Please upload a telehealth chat logs dataset in the sidebar to train the machine learning models before analyzing patient queries.")
    else:
        sample_queries = [
            "I've been getting severe headaches for the past week, but no fever.",
            "Doctor, I have intense abdominal pain and nausea since yesterday. Haven't had any vomiting though.",
            "I noticed sharp chest tightness and shortness of breath when climbing stairs. Denies chest pain.",
            "I have a red skin rash on my arms with intense itching for 3 days.",
            "Experiencing painful urination and frequent urination for 2 days. No blood in urine.",
            "Severe menstrual cramps and deep pelvic pain since Monday.",
            "I have a sore throat, nasal congestion, and mild fever."
        ]

        sel_sample = st.selectbox("Or choose a sample clinical query:", ["-- Type your own query below --"] + sample_queries)
        default_text = sel_sample if sel_sample != "-- Type your own query below --" else "I've been getting severe headaches for the past week, but no fever."

        user_query = st.text_area("Patient Statement / Telehealth Input Text:", value=default_text, height=100)

        if st.button("🚀 Analyze Patient Query", type="primary"):
            if not user_query.strip():
                st.error("Please enter a valid patient statement.")
            else:
                res = predictor.predict_query(user_query)

                if not res["success"]:
                    st.error(res.get("error", "An error occurred while analyzing query."))
                elif not res["has_symptoms"]:
                    st.info("ℹ️ " + res.get("message", "No symptom mentions recognized in statement."))
                else:
                    st.success(f"Extracted **{res['total_mentions']}** clinical symptom mention(s) from input text!")

                    # Iterate through mentions
                    for idx, m in enumerate(res["mentions"], start=1):
                        symptom_title = m["symptom_text"]
                        is_neg = m["is_negated"]
                        status_badge = "❌ Negated (Denied)" if is_neg else "✅ Affirmative"
                        badge_color = "#ef4444" if is_neg else "#10b981"

                        with st.expander(f"Symptom #{idx}: {symptom_title} ({status_badge})", expanded=True):
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            col_m1.markdown(f"**Normalized:** `{m['normalized_symptom']}`")
                            col_m1.markdown(f"**Status:** <span style='color:{badge_color}; font-weight:600;'>{m['status']}</span>", unsafe_allow_html=True)

                            col_m2.markdown(f"**Severity:** `{m['severity']}`")
                            col_m2.markdown(f"**Duration:** `{m['duration']}`")

                            col_m3.markdown(f"**Triage Urgency:** `{m['urgency']}`")
                            col_m3.markdown(f"**Character Span:** `[{m['char_span'][0]}:{m['char_span'][1]}]`")

                            col_m4.markdown(f"**Rule-based Cat:** `{m['nlp_rule_category']}`")
                            if m.get("ml_available"):
                                ens_cat = m['ensemble_prediction']
                                conf = m['ensemble_confidence'] * 100
                                col_m4.markdown(f"**Ensemble Cat:** <span style='color:#38bdf8; font-weight:bold;'>{ens_cat} ({conf:.1f}%)</span>", unsafe_allow_html=True)

                            st.markdown("---")

                            if m.get("ml_available"):
                                col_sub1, col_sub2 = st.columns([1, 1])

                                with col_sub1:
                                    st.markdown("#### 🤖 Individual Model Classifications")
                                    m_rows = []
                                    for m_name, p_info in m["model_predictions"].items():
                                        clean_m_name = m_name.replace("_", " ").title()
                                        m_rows.append({
                                            "Classifier Model": clean_m_name,
                                            "Predicted Body System": p_info["predicted_category"],
                                            "Model Confidence": f"{(p_info['confidence']*100):.1f}%" if p_info['confidence'] is not None else "N/A"
                                        })
                                    st.dataframe(pd.DataFrame(m_rows), use_container_width=True)

                                    if m.get("is_low_confidence"):
                                        st.warning(f"⚠️ {m['uncertainty_notice']}")
                                    else:
                                        st.caption(f"ℹ️ {m['uncertainty_notice']}")

                                with col_sub2:
                                    st.markdown("#### 📊 Ensemble Posterior Probability Distribution")
                                    fig_prob = plot_confidence_distribution_bar(m["probability_distribution"])
                                    st.plotly_chart(fig_prob, use_container_width=True)
                            else:
                                st.info("Classifiers are currently being fitted on the active dataset.")

                    st.info(f"💡 **Responsible Use Notice**: {res.get('disclaimer')}")

# ==============================================================================
# VIEW 3: MODEL COMPARISON (PAGE C)
# ==============================================================================
elif app_mode == "📈 ML Model Comparison":
    st.markdown("### 📈 Supervised ML Model Comparison & Benchmark")
    st.markdown("Performance metrics of the 4 classifiers and the weighted ensemble evaluated on a held-out test partition of your uploaded dataset.")

    if df_raw is None or df_raw.empty or training_result is None:
        st.info("📂 **No dataset uploaded yet.** Please upload a telehealth chat logs dataset in the sidebar to train and compare the machine learning models.")
    else:
        metrics_payload = training_result.get("metrics_payload", {})
        cms_payload = training_result.get("cms_payload", {})

        if not metrics_payload or "metrics" not in metrics_payload:
            st.warning("Models are currently training on the uploaded dataset...")
        else:
            test_samples = metrics_payload.get("num_test_samples", 0)
            best_model = metrics_payload.get("best_model_by_macro_f1", "N/A").replace("_", " ").title()
            dataset_desc = uploaded_file.name if uploaded_file is not None else "Uploaded Dataset"
            st.caption(f"Active Evaluation Dataset: **{dataset_desc}** (Test Partition N = {test_samples}) | Primary Metric: **Macro-F1**")

            st.success(f"🏆 Top Performing Model: **{best_model}**")

            metrics_dict = metrics_payload["metrics"]

            # Summary Table
            summary_rows = []
            for name, m in metrics_dict.items():
                clean_name = name.replace("_", " ").title()
                summary_rows.append({
                    "Model / Ensemble": clean_name,
                    "Accuracy": f"{m.get('accuracy', 0.0):.4f}",
                    "Macro F1": f"{m.get('macro_f1', 0.0):.4f}",
                    "Weighted F1": f"{m.get('weighted_f1', 0.0):.4f}",
                    "Macro Precision": f"{m.get('macro_precision', 0.0):.4f}",
                    "Macro Recall": f"{m.get('macro_recall', 0.0):.4f}",
                    "Fit Time (s)": f"{m.get('training_time_seconds', 0.0):.3f}",
                    "Inf / Sample (ms)": f"{m.get('inference_time_ms_per_sample', 0.0):.3f}"
                })
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Performance Comparison Grouped Bar Chart
            fig_comp = plot_model_comparison_bar(metrics_dict)
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True)

            st.markdown("---")

            # Confusion Matrix Heatmap Inspection
            st.subheader("🎯 Confusion Matrix Inspection")
            avail_models = list(metrics_dict.keys())
            sel_cm_model = st.selectbox(
                "Select Model for Confusion Matrix:",
                avail_models,
                format_func=lambda x: x.replace("_", " ").title()
            )

            class_labels = metrics_payload.get("class_labels", list(TAXONOMY_10.keys()))
            if cms_payload and "matrices" in cms_payload and sel_cm_model in cms_payload["matrices"]:
                cm_data = cms_payload["matrices"][sel_cm_model]
                fig_cm = plot_confusion_matrix_heatmap(cm_data, class_labels, sel_cm_model)
                st.plotly_chart(fig_cm, use_container_width=True)

            # Per-Class Breakdown Table
            st.subheader(f"📋 Per-Class Performance Breakdown: {sel_cm_model.replace('_', ' ').title()}")
            per_class_data = metrics_dict[sel_cm_model].get("per_class", {})
            if per_class_data:
                pc_rows = []
                for cls_name, pcm in per_class_data.items():
                    pc_rows.append({
                        "Clinical Body System": cls_name,
                        "Precision": f"{pcm['precision']:.4f}",
                        "Recall": f"{pcm['recall']:.4f}",
                        "F1-Score": f"{pcm['f1_score']:.4f}",
                        "Support": pcm['support']
                    })
                st.dataframe(pd.DataFrame(pc_rows), use_container_width=True)

            # Ensemble Weights
            st.markdown("---")
            st.subheader("⚖️ Ensemble Weight Allocation")
            ens_weights = metrics_dict.get("weighted_ensemble", {}).get("ensemble_weights", {})
            if ens_weights:
                w_rows = [{"Model": k.replace("_", " ").title(), "Weight": round(v, 4)} for k, v in ens_weights.items()]
                st.dataframe(pd.DataFrame(w_rows), use_container_width=True)
                st.caption("Weights are normalized dynamically from validation set macro-F1: $w_i = \\text{ValF1}_i / \\sum \\text{ValF1}_j$")
