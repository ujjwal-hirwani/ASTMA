"""
Interactive Visual Analytics Engine using Plotly, Pyvis, and NetworkX.
Provides rich charts for body system breakdown, symptom co-occurrence networks, and negation rates.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# Color Palette Definitions
CATEGORY_COLORS = {
    "Respiratory": "#3b82f6",
    "Gastrointestinal": "#10b981",
    "Neurological": "#8b5cf6",
    "Cardiovascular": "#ef4444",
    "Musculoskeletal": "#f59e0b",
    "Dermatological": "#ec4899",
    "General / Systemic": "#6366f1"
}

def plot_category_distribution(df: pd.DataFrame):
    """
    Creates a Donut Chart of extracted symptoms broken down by Clinical Body System.
    """
    if df.empty or "category" not in df.columns:
        return None

    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["Body System", "Symptom Count"]

    fig = px.pie(
        cat_counts,
        names="Body System",
        values="Symptom Count",
        hole=0.4,
        title="<b>Symptom Distribution by Clinical Body System</b>",
        color="Body System",
        color_discrete_map=CATEGORY_COLORS
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    return fig

def plot_top_symptoms(df: pd.DataFrame, top_n: int = 15):
    """
    Creates a Horizontal Bar Chart of top extracted symptoms.
    """
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
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        xaxis_title="Occurrence Count",
        yaxis_title="Symptom Term",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_negation_status(df: pd.DataFrame):
    """
    Creates a Positivity vs. Negation Ratio Donut Chart.
    """
    if df.empty or "status" not in df.columns:
        return None

    neg_counts = df["status"].value_counts().reset_index()
    neg_counts.columns = ["Status", "Count"]

    fig = px.pie(
        neg_counts,
        names="Status",
        values="Count",
        hole=0.5,
        title="<b>Affirmative vs. Negated Symptoms</b>",
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
    """
    Creates a Stacked Bar Chart of Symptom Urgency by Body System.
    """
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
        yaxis_title="Total Symptoms Detected",
        legend_title="Urgency Level"
    )
    return fig

def generate_cooccurrence_network_html(df: pd.DataFrame) -> str:
    """
    Generates an interactive Pyvis co-occurrence graph HTML string for symptoms co-occurring in the same chat.
    """
    if df.empty or "chat_id" not in df.columns:
        return "<p>No data available for network graph.</p>"

    # Group symptoms by chat_id
    grouped = df.groupby("chat_id")["symptom"].apply(list)

    G = nx.Graph()

    # Build co-occurrence edges
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
        return "<p>Not enough co-occurring symptoms to construct a network graph.</p>"

    # Create Pyvis network
    net = Network(height="450px", width="100%", bgcolor="#0f172a", font_color="white")
    net.from_nx(G)

    # Customize node sizes and physics
    for node in net.nodes:
        node["size"] = 18
        node["color"] = "#38bdf8"
        node["font"] = {"size": 14}

    # Save to temp file
    temp_dir = tempfile.gettempdir()
    html_path = os.path.join(temp_dir, "cooccurrence_network.html")
    net.write_html(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content
