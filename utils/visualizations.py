"""
ResearchIQ - Visualization Utilities
========================================
Plotly charts, citation network visualizations, and analytics displays.
"""

import logging
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from config.settings import CHART_THEME, COLOR_PALETTE

logger = logging.getLogger(__name__)


def _safe_count(v):
    """Handle both {year: int} and {year: {'count': int}} formats."""
    return v["count"] if isinstance(v, dict) else int(v)

def _safe_avg_cit(v):
    return v.get("avg_citations", 0) if isinstance(v, dict) else 0


def make_temporal_chart(temporal_data: Dict) -> go.Figure:
    """Papers published per year bar chart."""
    if not temporal_data:
        return go.Figure().update_layout(title="No temporal data available")

    years = sorted(temporal_data.keys())
    counts = [_safe_count(temporal_data[y]) for y in years]
    avg_cits = [_safe_avg_cit(temporal_data[y]) for y in years]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Papers Published Per Year", "Average Citations Per Year"),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(x=years, y=counts, name="Papers",
               marker_color="#4F8EF7", marker_line_color="#2563EB", marker_line_width=1),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=years, y=avg_cits, name="Avg Citations",
                   line=dict(color="#10B981", width=3), mode="lines+markers",
                   marker=dict(size=8)),
        row=2, col=1,
    )

    fig.update_layout(
        template=CHART_THEME, height=480, showlegend=True,
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font=dict(color="#CBD5E1"),
    )
    return fig


def make_keyword_bubble_chart(keywords: List[Dict]) -> go.Figure:
    """Bubble chart for trending keywords."""
    if not keywords:
        return go.Figure()

    df = pd.DataFrame(keywords[:30])
    if df.empty or "frequency" not in df.columns:
        return go.Figure()

    fig = px.scatter(
        df, x="frequency", y="recent_frequency",
        size="frequency", color="trend_score",
        hover_name="keyword",
        color_continuous_scale="Viridis",
        size_max=40,
        title="Keyword Trend Analysis",
    )
    fig.update_layout(
        template=CHART_THEME, height=420,
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font=dict(color="#CBD5E1"),
        xaxis_title="Total Frequency",
        yaxis_title="Recent Frequency (2022+)",
    )
    return fig


def make_gap_opportunity_matrix(opportunity_matrix: List[Dict]) -> go.Figure:
    """2D scatter: Novelty vs Impact."""
    if not opportunity_matrix:
        return go.Figure()

    quadrant_colors = {
        "🚀 Sweet Spot": "#10B981",
        "🔬 Exploratory": "#4F8EF7",
        "⚡ Quick Win": "#F59E0B",
        "📚 Incremental": "#6B7280",
    }

    fig = go.Figure()
    fig.add_shape(type="rect", x0=7, y0=7, x1=10.5, y1=10.5,
                  fillcolor="rgba(16,185,129,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=7, y0=0, x1=10.5, y1=7,
                  fillcolor="rgba(79,142,247,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=7, x1=7, y1=10.5,
                  fillcolor="rgba(245,158,11,0.08)", line_width=0)

    seen_quadrants = set()
    for item in opportunity_matrix:
        q = item.get("quadrant", "")
        color = quadrant_colors.get(q, "#6B7280")
        show_legend = q not in seen_quadrants
        seen_quadrants.add(q)
        label = item["title"][:25] + "..." if len(item["title"]) > 25 else item["title"]
        fig.add_trace(go.Scatter(
            x=[item["novelty"]], y=[item["impact"]],
            mode="markers+text",
            marker=dict(size=16, color=color, opacity=0.9),
            text=[label], textposition="top center",
            name=q, showlegend=show_legend,
            hovertemplate=(
                f"<b>{item['title']}</b><br>"
                f"Novelty: {item['novelty']}/10<br>"
                f"Impact: {item['impact']}/10<br>"
                f"Feasibility: {item.get('feasibility','N/A')}<extra></extra>"
            ),
        ))

    for x, y, label in [(8.5, 9.5, "🚀 Sweet Spot"), (8.5, 3.5, "🔬 Exploratory"),
                         (3.5, 9.5, "⚡ Quick Win"), (3.5, 3.5, "📚 Incremental")]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                           font=dict(size=11, color="#94A3B8"), opacity=0.6)

    fig.add_shape(type="line", x0=7, y0=0, x1=7, y1=10.5,
                  line=dict(color="#475569", dash="dash", width=1))
    fig.add_shape(type="line", x0=0, y0=7, x1=10.5, y1=7,
                  line=dict(color="#475569", dash="dash", width=1))

    fig.update_layout(
        title="Research Gap Opportunity Matrix",
        xaxis_title="Novelty Score", yaxis_title="Impact Score",
        xaxis=dict(range=[0, 10.5]), yaxis=dict(range=[0, 10.5]),
        template=CHART_THEME, height=480,
        paper_bgcolor="#0F172A", plot_bgcolor="#1E293B",
        font=dict(color="#CBD5E1"),
    )
    return fig


def make_novelty_gauge(novelty_score: float) -> go.Figure:
    """Gauge chart for novelty score."""
    color = "#10B981" if novelty_score >= 75 else "#F59E0B" if novelty_score >= 50 else "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=novelty_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Novelty Score", "font": {"color": "#CBD5E1", "size": 16}},
        delta={"reference": 70, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": color},
            "bgcolor": "#1E293B",
            "borderwidth": 2, "bordercolor": "#334155",
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                {"range": [70, 100], "color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {"line": {"color": "#CBD5E1", "width": 4},
                          "thickness": 0.75, "value": 70},
        },
        number={"font": {"color": color, "size": 32}},
    ))
    fig.update_layout(
        template=CHART_THEME, height=280,
        paper_bgcolor="#0F172A", font=dict(color="#CBD5E1"),
        margin=dict(t=70, b=20, l=20, r=20),
    )
    return fig


def make_citation_network(papers: List[Dict], max_nodes: int = 30) -> str:
    """Generate an interactive citation network as HTML."""
    try:
        from pyvis.network import Network

        G = nx.DiGraph()
        for paper in papers[:max_nodes]:
            citations = paper.get("citations", 0)
            year = paper.get("year", "N/A")
            title = paper.get("title", "")[:50]
            node_size = max(15, min(50, citations / 10 + 15))
            color = "#4F8EF7" if citations > 100 else "#10B981" if citations > 20 else "#6B7280"
            G.add_node(title,
                       title=f"{paper.get('title','')}\nYear: {year}\nCitations: {citations}",
                       size=node_size, color=color, label=title[:30])

        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, min(i + 4, len(nodes))):
                p_i_year = (papers[i].get("year") or 0) if i < len(papers) else 0
                p_j_year = (papers[j].get("year") or 0) if j < len(papers) else 0
                if p_i_year and p_j_year and abs(p_i_year - p_j_year) <= 3:
                    G.add_edge(nodes[i], nodes[j])

        net = Network(height="500px", width="100%", bgcolor="#0F172A",
                      font_color="#CBD5E1", directed=True)
        net.from_nx(G)
        net.set_options("""{
            "physics": {"enabled": true, "solver": "forceAtlas2Based"},
            "edges": {"color": {"color": "#334155"}, "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}},
            "nodes": {"borderWidth": 2}
        }""")
        return net.generate_html()

    except Exception as e:
        logger.error(f"Network generation error: {e}")
        return f"<p style='color:#CBD5E1;padding:20px;'>Citation network unavailable: {e}</p>"


def make_topic_sunburst(topics: List[Dict]) -> go.Figure:
    """Sunburst chart for topic hierarchy."""
    if not topics:
        return go.Figure()

    labels = ["Research Topics"]
    parents = [""]
    values = [100]

    for topic in topics:
        t_name = topic.get("name", "Unknown")
        prevalence = topic.get("prevalence", 15)
        labels.append(t_name)
        parents.append("Research Topics")
        values.append(prevalence)
        for kw in topic.get("keywords", [])[:3]:
            labels.append(kw)
            parents.append(t_name)
            values.append(prevalence / 3)

    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=px.colors.qualitative.Set3),
        hovertemplate="<b>%{label}</b><br>Prevalence: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Research Topic Distribution",
        template=CHART_THEME, height=420,
        paper_bgcolor="#0F172A", font=dict(color="#CBD5E1"),
    )
    return fig
