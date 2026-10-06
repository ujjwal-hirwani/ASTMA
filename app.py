"""
Automated Text Mining Dashboard for Categorizing Patient Symptoms from Telehealth Chat Logs using spaCy and Streamlit.
Author: ASTMA Senior Software Developer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit.components.v1 import html

# Import custom src modules
from src.data_generator import generate_synthetic_telehealth_data
from src.nlp_engine import analyze_chat_message
from src.visualizer import (
    plot_category_distribution,
    plot_top_symptoms,
    plot_negation_status,
    plot_urgency_breakdown,
    generate_cooccurrence_network_html
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Telehealth Symptom Text Mining Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS Styling
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0b0f19;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #e2e8f0;
    }
    .stAppHeader {
        background-color: transparent;
    }
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    /* Headers & Text */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600;
    }
    /* Tab Header Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
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
    else:
        target_df = df.copy()

    records = []
    for idx, row in target_df.iterrows():
        chat_id = str(row.get("chat_id", f"CHAT-{idx}"))
        timestamp = str(row.get("timestamp", "N/A"))
        speaker = str(row.get("sender", "patient"))
        text = str(row.get("message_text", ""))

        if text.strip():
            extracted = analyze_chat_message(text, chat_id=chat_id, timestamp=timestamp, speaker=speaker)
            records.extend(extracted)

    return pd.DataFrame(records)

# Header Title Banner
st.markdown("""
<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 24px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 24px;">
    <h1 style="margin:0; font-size: 2.2rem; color: #38bdf8;">🩺 Telehealth Symptom Text Mining Dashboard</h1>
    <p style="margin-top: 8px; font-size: 1.05rem; color: #94a3b8;">
        An Automated Text Mining & NLP Analytics Platform for Categorizing Patient Symptoms from Telehealth Chat Logs using <b>spaCy</b> & <b>Streamlit</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Data Ingestion & Filters
st.sidebar.header("📁 Data Management & Controls")

dataset_source = st.sidebar.radio(
    "Select Dataset Source",
    ["⚡ Load Demo Telehealth Dataset (50 Chats)", "📤 Upload Custom Telehealth Logs (CSV/JSON)"],
    index=0
)

df_raw = None

if "Upload" in dataset_source:
    uploaded_file = st.sidebar.file_uploader("Upload Chat Logs", type=["csv", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_json(uploaded_file)
            st.sidebar.success(f"Successfully loaded {len(df_raw)} records!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    else:
        st.sidebar.info("Awaiting file upload... Using synthetic dataset as fallback.")
        df_raw = generate_synthetic_telehealth_data(50)
else:
    df_raw = generate_synthetic_telehealth_data(50)
    st.sidebar.success(f"Loaded {len(df_raw)} synthetic telehealth messages!")

# Sidebar Settings
st.sidebar.markdown("---")
st.sidebar.header("⚙️ NLP Processing Options")
patient_turns_only = st.sidebar.checkbox("Process Patient Turns Only", value=True)

# Run NLP Engine
if df_raw is not None and not df_raw.empty:
    df_extracted = run_nlp_pipeline(df_raw, process_patient_only=patient_turns_only)

    if not df_extracted.empty:
        # Sidebar Filter Controls
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Dynamic Filters")

        available_categories = list(df_extracted["category"].unique())
        selected_categories = st.sidebar.multiselect(
            "Filter by Body System",
            options=available_categories,
            default=available_categories
        )

        status_options = ["All", "Affirmative Only", "Negated Only"]
        selected_status = st.sidebar.selectbox("Negation Filter", status_options, index=0)

        urgency_options = ["All", "High", "Medium", "Low"]
        selected_urgency = st.sidebar.selectbox("Filter by Urgency Level", urgency_options, index=0)

        # Apply Filters
        filtered_df = df_extracted[df_extracted["category"].isin(selected_categories)].copy()
        if selected_status == "Affirmative Only":
            filtered_df = filtered_df[filtered_df["is_negated"] == False]
        elif selected_status == "Negated Only":
            filtered_df = filtered_df[filtered_df["is_negated"] == True]

        if selected_urgency != "All":
            filtered_df = filtered_df[filtered_df["urgency"] == selected_urgency]

        # Top KPI Scorecards
        total_chats = df_raw["chat_id"].nunique() if "chat_id" in df_raw.columns else len(df_raw)
        total_symptoms = len(filtered_df)
        negated_count = filtered_df["is_negated"].sum()
        negated_pct = round((negated_count / total_symptoms * 100), 1) if total_symptoms > 0 else 0
        high_urgency_count = len(filtered_df[filtered_df["urgency"] == "High"])

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Conversations Analyzed", f"{total_chats}")
        kpi2.metric("Total Symptoms Extracted", f"{total_symptoms}")
        kpi3.metric("Negation Rate", f"{negated_pct}%", f"{negated_count} Negated")
        kpi4.metric("High-Urgency Flagged", f"{high_urgency_count}", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs Architecture
        tab_analytics, tab_network, tab_inspector, tab_table, tab_report = st.tabs([
            "📊 Executive Analytics",
            "🕸️ Symptom Co-occurrence Network",
            "🔍 Transcript Inspector",
            "📋 Extracted Dataset & Export",
            "📄 Automated Clinical Summary"
        ])

        # TAB 1: EXECUTIVE ANALYTICS
        with tab_analytics:
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_cat = plot_category_distribution(filtered_df)
                if fig_cat:
                    st.plotly_chart(fig_cat, use_container_width=True)
            with col_chart2:
                fig_top = plot_top_symptoms(filtered_df, top_n=12)
                if fig_top:
                    st.plotly_chart(fig_top, use_container_width=True)

            st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

            col_chart3, col_chart4 = st.columns(2)
            with col_chart3:
                fig_neg = plot_negation_status(filtered_df)
                if fig_neg:
                    st.plotly_chart(fig_neg, use_container_width=True)
            with col_chart4:
                fig_urg = plot_urgency_breakdown(filtered_df)
                if fig_urg:
                    st.plotly_chart(fig_urg, use_container_width=True)

        # TAB 2: CO-OCCURRENCE NETWORK
        with tab_network:
            st.subheader("🕸️ Symptom Co-occurrence Network Graph")
            st.write("This interactive network graph maps symptom terms that co-occur within the same patient telehealth chat session.")
            html_net = generate_cooccurrence_network_html(filtered_df)
            html(html_net, height=480, scrolling=True)

        # TAB 3: TRANSCRIPT INSPECTOR
        with tab_inspector:
            st.subheader("🔍 Interactive Transcript & spaCy Entity Highlight Inspector")
            chat_list = df_raw["chat_id"].unique() if "chat_id" in df_raw.columns else []
            if len(chat_list) > 0:
                selected_chat = st.selectbox("Select Consultation Session ID:", chat_list)
                chat_messages = df_raw[df_raw["chat_id"] == selected_chat]

                st.markdown("### Chat Session Timeline")
                for _, msg_row in chat_messages.iterrows():
                    sender = msg_row.get("sender", "patient")
                    text = msg_row.get("message_text", "")
                    time = msg_row.get("timestamp", "")

                    if sender.lower() == "patient":
                        st.markdown(f"""
                        <div style="background: rgba(30, 58, 138, 0.4); border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px;">
                            <strong style="color: #60a5fa;">👤 Patient ({time}):</strong><br>
                            <span style="font-size: 1.05rem; color: #f8fafc;">{text}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border-left: 4px solid #64748b; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px;">
                            <strong style="color: #cbd5e1;">🩺 Clinician ({time}):</strong><br>
                            <span style="font-size: 1.05rem; color: #e2e8f0;">{text}</span>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("### spaCy Mined Entities for Selected Chat")
                chat_extracted = df_extracted[df_extracted["chat_id"] == selected_chat]
                if not chat_extracted.empty:
                    st.dataframe(
                        chat_extracted[["symptom", "category", "status", "severity", "duration", "urgency"]],
                        use_container_width=True
                    )
                else:
                    st.info("No specific symptom entities identified in this chat.")

        # TAB 4: EXTRACTED DATASET & EXPORT
        with tab_table:
            st.subheader("📋 Structured Clinical Symptom Dataset")
            st.write(f"Showing **{len(filtered_df)}** extracted symptom records after dynamic filtering.")

            st.dataframe(
                filtered_df[["chat_id", "timestamp", "symptom", "category", "status", "severity", "duration", "urgency", "raw_text"]],
                use_container_width=True,
                height=400
            )

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Extracted Dataset as CSV",
                    data=csv_data,
                    file_name="telehealth_extracted_symptoms.csv",
                    mime="text/csv"
                )
            with col_exp2:
                json_data = filtered_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Export Extracted Dataset as JSON",
                    data=json_data,
                    file_name="telehealth_extracted_symptoms.json",
                    mime="application/json"
                )

        # TAB 5: AUTOMATED CLINICAL SUMMARY
        with tab_report:
            st.subheader("📄 Automated Executive Clinical Report")

            top_cat = filtered_df["category"].mode()[0] if not filtered_df.empty else "N/A"
            top_sym = filtered_df["symptom"].mode()[0] if not filtered_df.empty else "N/A"

            report_text = f"""========================================================================
AUTOMATED TELEHEALTH TEXT MINING EXECUTIVE SUMMARY REPORT
========================================================================
Generated On: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Telehealth Consultations Processed: {total_chats}
Total Symptoms Identified: {total_symptoms}

1. KEY CLINICAL METRICS
------------------------------------------------------------------------
- Dominant Clinical Body System: {top_cat}
- Most Frequent Patient Symptom: {top_sym}
- Affirmative Symptoms Count: {total_symptoms - negated_count} ({(100 - negated_pct):.1f}%)
- Negated Symptoms Count: {negated_count} ({negated_pct:.1f}%)
- High-Urgency Cases Identified: {high_urgency_count}

2. BODY SYSTEM BREAKDOWN
------------------------------------------------------------------------
"""
            cat_counts = filtered_df["category"].value_counts()
            for cat, count in cat_counts.items():
                pct = (count / total_symptoms) * 100 if total_symptoms > 0 else 0
                report_text += f"  * {cat}: {count} occurrences ({pct:.1f}%)\n"

            report_text += """
3. CLINICAL RECOMMENDATIONS & RECOMMENDATIONS
------------------------------------------------------------------------
- High-urgency flagged cases require immediate clinician triage follow-up.
- Continued monitoring of top recurring symptom patterns is advised for virtual care capacity planning.
========================================================================
"""
            st.code(report_text, language="text")

            st.download_button(
                label="📥 Download Executive Report (.txt)",
                data=report_text,
                file_name="telehealth_clinical_summary_report.txt",
                mime="text/plain"
            )

    else:
        st.warning("No symptom entities were extracted from the loaded dataset. Check data column format.")
else:
    st.info("Please load or upload a telehealth chat log dataset.")
