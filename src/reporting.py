"""
Automated Clinical Report Generator for ASTMA.
Produces both structured text summaries and formatted PDF executive reports using ReportLab.
"""

from typing import Dict, Any, List
import io
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

def generate_text_summary(df_filtered: pd.DataFrame, total_chats: int) -> str:
    """Generates structured ASCII executive summary report."""
    total_symptoms = len(df_filtered)
    negated_count = df_filtered["is_negated"].sum() if "is_negated" in df_filtered.columns else 0
    negated_pct = (negated_count / total_symptoms * 100) if total_symptoms > 0 else 0
    high_urgency_count = len(df_filtered[df_filtered["urgency"] == "High"]) if "urgency" in df_filtered.columns else 0

    top_cat = df_filtered["category"].mode()[0] if not df_filtered.empty and "category" in df_filtered.columns else "N/A"
    top_sym = df_filtered["symptom"].mode()[0] if not df_filtered.empty and "symptom" in df_filtered.columns else "N/A"

    report = f"""========================================================================
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
    if "category" in df_filtered.columns:
        cat_counts = df_filtered["category"].value_counts()
        for cat, count in cat_counts.items():
            pct = (count / total_symptoms) * 100 if total_symptoms > 0 else 0
            report += f"  * {cat:22s}: {count:4d} occurrences ({pct:4.1f}%)\n"

    report += """
3. CLINICAL GOVERNANCE & RESPONSIBLE USE NOTICE
------------------------------------------------------------------------
- High-urgency flagged cases require immediate clinician triage follow-up.
- Continued monitoring of top recurring symptom patterns is advised for virtual care capacity planning.
- This automated summary is generated from NLP text-mining patterns for research and operational intelligence.
- It does NOT constitute medical advice, diagnostic proof, or therapeutic guidance.
========================================================================
"""
    return report

def generate_pdf_summary(df_filtered: pd.DataFrame, total_chats: int) -> bytes:
    """Generates a professional PDF report using ReportLab and returns raw bytes."""
    if not HAS_REPORTLAB:
        return b""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=10
    )
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0284c7'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # Title
    story.append(Paragraph("Telehealth Symptom Text Mining Executive Report", title_style))
    story.append(Paragraph(f"<b>Generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | <b>Total Chats:</b> {total_chats}", body_style))
    story.append(Spacer(1, 12))

    # Metrics Summary Table
    total_symptoms = len(df_filtered)
    negated_count = int(df_filtered["is_negated"].sum()) if "is_negated" in df_filtered.columns else 0
    high_urgency = len(df_filtered[df_filtered["urgency"] == "High"]) if "urgency" in df_filtered.columns else 0

    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Chats Analyzed", str(total_chats), "Total Extracted Symptoms", str(total_symptoms)],
        ["Affirmative Symptoms", str(total_symptoms - negated_count), "Negated Symptoms", str(negated_count)],
        ["High-Urgency Flags", str(high_urgency), "Negation Rate", f"{(negated_count/total_symptoms*100 if total_symptoms>0 else 0):.1f}%"]
    ]
    t = Table(metrics_data, colWidths=[130, 130, 130, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0f172a')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1'))
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Breakdown Table
    story.append(Paragraph("Clinical Body System Breakdown", h2_style))
    cat_counts = df_filtered["category"].value_counts().reset_index()
    cat_counts.columns = ["Body System", "Count"]

    table_data = [["Clinical Category", "Count", "Percentage"]]
    for _, row in cat_counts.iterrows():
        pct = (row["Count"] / total_symptoms * 100) if total_symptoms > 0 else 0
        table_data.append([str(row["Body System"]), str(row["Count"]), f"{pct:.1f}%"])

    t_cat = Table(table_data, colWidths=[240, 130, 150])
    t_cat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0'))
    ]))
    story.append(t_cat)
    story.append(Spacer(1, 16))

    # Disclaimer
    disclaimer_text = (
        "<b>Responsible Use Disclaimer:</b> This report is generated automatically by the ASTMA NLP & Machine Learning platform "
        "for academic research and telehealth quality analytics. It does NOT provide medical diagnoses or replace clinical judgement."
    )
    story.append(Paragraph(disclaimer_text, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
