"""
Interactive Visual Analytics Engine using Plotly, Pyvis, and NetworkX.
Provides rich visual charts for body system distributions, model metric comparisons,
confusion matrices, confidence distributions, and symptom co-occurrence networks.
"""

import os
import tempfile
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network

from config.settings import CATEGORY_COLORS

def plot_category_distribution(df: pd.DataFrame):
    """Creates a Donut Chart of extracted symptoms broken down by Clinical Body System."""
    if df.empty or "category" not in df.columns:
        return None

    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["Body System", "Symptom Count"]

    fig = px.pie(
        cat_counts,
        names="Body System",
        values="Symptom Count",
        hole=0.42,
        title="<b>Symptom Distribution Across Clinical Body Systems</b>",
        color="Body System",
        color_discrete_map=CATEGORY_COLORS
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )
    return fig

def plot_top_symptoms(df: pd.DataFrame, top_n: int = 15):
    """Creates a Horizontal Bar Chart of top extracted symptoms."""
    if df.empty or "symptom" not in df.columns:
        return None

    symptom_counts = df["symptom"].value_counts().head(top_n).reset_index()
    symptom_counts.columns = ["Symptom", "Frequency"]
    symptom_counts = symptom_counts.sort_values(by="Frequency", ascending=True)

    fig = px.bar(
        symptom_counts,
        x="Frequency",
        y="Symptom",
        orientation='h',
        title=f"<b>Top {top_n} Most Frequent Patient Symptoms</b>",
        color="Frequency",
        color_continuous_scale="Tealgrn"
    )
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        xaxis_title="Occurrence Count",
        yaxis_title="Symptom Term",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_negation_status(df: pd.DataFrame):
    """Creates an Affirmative vs. Negated Donut Chart."""
    if df.empty or "status" not in df.columns:
        return None

    neg_counts = df["status"].value_counts().reset_index()
    neg_counts.columns = ["Status", "Count"]

    fig = px.pie(
        neg_counts,
        names="Status",
        values="Count",
        hole=0.5,
        title="<b>Affirmative vs. Negated Clinical Mentions</b>",
        color="Status",
        color_discrete_map={"Affirmative": "#10b981", "Negated": "#ef4444"}
    )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_urgency_breakdown(df: pd.DataFrame):
    """Creates a Stacked Bar Chart of Symptom Urgency by Body System."""
    if df.empty or "urgency" not in df.columns:
        return None

    urgency_counts = df.groupby(["category", "urgency"]).size().reset_index(name="Count")

    fig = px.bar(
        urgency_counts,
        x="category",
        y="Count",
        color="urgency",
        title="<b>Triage Urgency Level by Clinical Category</b>",
        color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
        barmode="stack"
    )
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        xaxis_title="Body System",
        yaxis_title="Total Mentions Detected",
        legend_title="Urgency Level"
    )
    return fig

def plot_model_comparison_bar(metrics_dict: Dict[str, Any]):
    """Creates a Grouped Bar Chart comparing Accuracy, Macro F1, Weighted F1, Macro P, and Macro R."""
    rows = []
    metric_keys = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    metric_labels = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]

    for model_name, m in metrics_dict.items():
        clean_name = model_name.replace("_", " ").title()
        for k, lbl in zip(metric_keys, metric_labels):
            rows.append({
                "Model": clean_name,
                "Metric": lbl,
                "Score": round(float(m.get(k, 0.0)), 4)
            })

    df_plot = pd.DataFrame(rows)
    fig = px.bar(
        df_plot,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="<b>Cross-Model Performance Comparison (Held-Out Test Split)</b>",
        color_discrete_sequence=["#38bdf8", "#818cf8", "#34d399", "#f472b6", "#fbbf24"]
    )
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        yaxis=dict(range=[0.75, 1.02]),
        xaxis_title="Classifier / Ensemble",
        yaxis_title="Metric Value",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35)
    )
    return fig

def plot_confusion_matrix_heatmap(cm_matrix: List[List[int]], class_labels: List[str], model_name: str):
    """Renders an interactive Plotly Heatmap for a classification confusion matrix."""
    clean_name = model_name.replace("_", " ").title()
    z_data = np.array(cm_matrix)

    fig = px.imshow(
        z_data,
        x=class_labels,
        y=class_labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Category", y="True Ground-Truth Category", color="Count"),
        title=f"<b>Confusion Matrix: {clean_name}</b>",
        text_auto=True
    )
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=11),
        xaxis=dict(tickangle=-45),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_confidence_distribution_bar(prob_dist: Dict[str, float]):
    """Renders a horizontal bar chart of predicted class probability distribution for a query."""
    sorted_items = sorted(prob_dist.items(), key=lambda x: x[1], reverse=False)
    classes = [k for k, v in sorted_items]
    scores = [v for k, v in sorted_items]

    fig = px.bar(
        x=scores,
        y=classes,
        orientation='h',
        title="<b>Ensemble Category Probability Distribution</b>",
        labels={"x": "Posterior Probability", "y": "Clinical Category"},
        color=scores,
        color_continuous_scale="Purp"
    )
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=12),
        xaxis=dict(range=[0.0, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def generate_cooccurrence_network_html(df: pd.DataFrame) -> str:
    """Generates an interactive Pyvis co-occurrence graph HTML string."""
    if df.empty or "chat_id" not in df.columns:
        return "<p style='color:#94a3b8;'>No data available for network graph.</p>"

    grouped = df.groupby("chat_id")["symptom"].apply(list)
    G = nx.Graph()

    for symptoms in grouped:
        unique_syms = list(set(symptoms))
        for i in range(len(unique_syms)):
            for j in range(i + 1, len(unique_syms)):
                u, v = unique_syms[i], unique_syms[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    if G.number_of_nodes() == 0:
        return "<p style='color:#94a3b8;'>Not enough co-occurring symptoms to construct network graph.</p>"

    net = Network(height="460px", width="100%", bgcolor="#0f172a", font_color="white")
    net.from_nx(G)

    for node in net.nodes:
        node["size"] = 20
        node["color"] = "#38bdf8"
        node["font"] = {"size": 14}

    temp_dir = tempfile.gettempdir()
    html_path = os.path.join(temp_dir, "cooccurrence_network.html")
    net.write_html(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content
