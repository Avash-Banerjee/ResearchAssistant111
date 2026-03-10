"""
ResearchIQ - Main Streamlit Application (v2)
=============================================
Multi-Agent Research Intelligence Framework
Fixes: duplicate plotly keys, auto API key from .env,
       separate agent sections, PDF upload support
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Load .env BEFORE anything else so os.environ is populated
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import streamlit as st
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchIQ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0F1E !important;
    font-family: 'Space Grotesk', sans-serif;
    color: #E2E8F0;
}
[data-testid="stSidebar"] {
    background: #0D1629 !important;
    border-right: 1px solid #1E2D4A;
}
.riq-header {
    background: linear-gradient(135deg, #0D1629 0%, #1a2744 50%, #0D1629 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.riq-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4F8EF7, #8B5CF6, #10B981);
}
.riq-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4F8EF7 0%, #8B5CF6 50%, #10B981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.riq-subtitle { color: #64748B; font-size: 0.95rem; margin-top: 6px; }

.section-box {
    background: #0D1629;
    border: 1px solid #1E2D4A;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}
.section-title {
    color: #E2E8F0;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #1E2D4A;
}
.metric-card {
    background: #111827;
    border: 1px solid #1E2D4A;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #4F8EF7;
}
.metric-label { color: #64748B; font-size: 0.78rem; margin-top: 4px; }

.paper-card {
    background: #111827;
    border: 1px solid #1E2D4A;
    border-radius: 8px;
    padding: 12px;
    margin: 5px 0;
}
.paper-title { font-weight: 600; color: #93C5FD; font-size: 0.88rem; }
.paper-meta { color: #475569; font-size: 0.76rem; margin-top: 4px; }
.badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 4px;
    font-size: 0.69rem;
    font-weight: 600;
    margin-right: 4px;
}
.badge-arxiv { background: #1E2D4A; color: #4F8EF7; }
.badge-ss { background: #1A2D1A; color: #10B981; }
.badge-year { background: #2D2214; color: #FBBF24; }

.gap-card {
    background: #111827;
    border: 1px solid #1E2D4A;
    border-left: 3px solid #F59E0B;
    border-radius: 8px;
    padding: 14px;
    margin: 7px 0;
}
.gap-title { font-weight: 700; color: #FCD34D; font-size: 0.93rem; }
.gap-desc { color: #94A3B8; font-size: 0.82rem; margin-top: 6px; line-height: 1.5; }
.score-pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.71rem;
    font-weight: 600;
    margin: 4px 3px 0 0;
}
.status-badge { padding: 3px 10px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }
.status-idle { background: #1E293B; color: #64748B; }
.status-done { background: #0F2D1F; color: #10B981; }

.stButton button {
    background: linear-gradient(135deg, #1E40AF, #3730A3) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #2563EB, #4F46E5) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #111827 !important;
    border: 1px solid #1E2D4A !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}
hr { border-color: #1E2D4A !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0A0F1E; }
::-webkit-scrollbar-thumb { background: #1E2D4A; border-radius: 3px; }
code { background: #111827 !important; color: #10B981 !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "lit_result": None,
        "trend_result": None,
        "gap_result": None,
        "method_result": None,
        "grant_result": None,
        "novelty_result": None,
        "papers": [],
        "input_source": "topic",
        "pdf_text": "",
        "pdf_title": "",
        "detected_topic": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="riq-header">
    <p class="riq-title">🔬 ResearchIQ</p>
    <p class="riq-subtitle">Multi-Agent Research Intelligence · Literature · Trends · Gaps · Methodology · Grant Writing · Novelty Scoring</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Auto-load API key from .env — no prompt if already set
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        st.success("✅ API Key loaded from .env")
        api_key = env_key
    else:
        api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIzaSy...")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            st.success("✅ API Key set")

    st.divider()
    st.markdown("### 📥 Input Mode")
    input_mode = st.radio("Input type", ["🔎 Search by Topic", "📄 Upload PDF Paper"])
    st.session_state.input_source = "pdf" if "PDF" in input_mode else "topic"

    st.divider()
    st.markdown("### 🔬 Research Settings")
    research_query = st.text_input(
        "Research Topic",
        placeholder="e.g., Federated Learning for Healthcare",
        disabled=(st.session_state.input_source == "pdf"),
        key="sidebar_query",
    )
    sources = st.multiselect("Paper Sources", ["arxiv", "semantic_scholar"],
                             default=["arxiv", "semantic_scholar"])
    max_papers = st.slider("Max Papers per Source", 5, 25, 10)

    st.divider()
    st.markdown("### 📋 Grant Settings")
    agency = st.selectbox("Funding Agency", ["NSF", "NIH", "DARPA", "DOE", "EU Horizon"])
    budget_range = st.selectbox("Budget Range", ["$50K-$100K", "$100K-$500K", "$500K-$1M", "$1M-$5M"])
    duration = st.selectbox("Duration", ["1 year", "2 years", "3 years", "4 years", "5 years"], index=2)

    st.divider()
    st.markdown("### 👤 PI Details")
    pi_name = st.text_input("PI Name", placeholder="Dr. Jane Smith")
    pi_institution = st.text_input("Institution", placeholder="MIT")
    pi_dept = st.text_input("Department", placeholder="Computer Science")

    st.divider()
    st.markdown("### 🤖 Agent Status")
    for icon, name, done in [
        ("🔍", "Literature Mining",  bool(st.session_state.lit_result)),
        ("📈", "Trend Analysis",     bool(st.session_state.trend_result)),
        ("🎯", "Gap Identification", bool(st.session_state.gap_result)),
        ("⚗️", "Methodology",        bool(st.session_state.method_result)),
        ("📝", "Grant Writing",      bool(st.session_state.grant_result)),
        ("🏆", "Novelty Scoring",    bool(st.session_state.novelty_result)),
    ]:
        cls = "status-done" if done else "status-idle"
        label = "✓ Done" if done else "Idle"
        st.markdown(
            f'<div style="background:#0D1629;border:1px solid #1E2D4A;border-radius:8px;'
            f'padding:7px 12px;margin:3px 0;display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="color:#CBD5E1;font-size:0.85rem;">{icon} {name}</span>'
            f'<span class="status-badge {cls}">{label}</span></div>',
            unsafe_allow_html=True,
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────
def check_api():
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        st.error("⚠️ No Gemini API key. Add GEMINI_API_KEY to your .env file or enter it in the sidebar.")
        return False
    return True

def get_active_query():
    if st.session_state.input_source == "pdf":
        return st.session_state.detected_topic or st.session_state.pdf_title or "research"
    return research_query or "research"

def pi_info():
    return {
        "name": pi_name or "Principal Investigator",
        "institution": pi_institution or "University",
        "department": pi_dept or "Computer Science",
    }

def section_header(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def metric_card(col, val, label):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PDF UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.input_source == "pdf":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    section_header("📄 Upload Research Paper (PDF)")

    uploaded = st.file_uploader("Upload a PDF research paper", type=["pdf"], key="pdf_uploader")
    if uploaded and not st.session_state.pdf_text:
        with st.spinner("Extracting text from PDF..."):
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                st.session_state.pdf_text = text[:15000]
                st.session_state.pdf_title = uploaded.name.replace(".pdf", "")
                st.success(f"✅ Extracted {len(text):,} characters from **{uploaded.name}**")
                with st.expander("Preview"):
                    st.text(text[:800] + "...")
            except ImportError:
                st.error("Run: `pip install pypdf`")
            except Exception as e:
                st.error(f"PDF error: {e}")

    if st.session_state.pdf_text and not st.session_state.detected_topic:
        if check_api() and st.button("🔍 Auto-detect Topic from PDF"):
            with st.spinner("Detecting topic..."):
                try:
                    from core.llm_client import get_llm
                    llm = get_llm()
                    topic = llm.generate(
                        f"Extract the main research topic (5-8 words) from this paper excerpt:\n\n"
                        f"{st.session_state.pdf_text[:2000]}\n\nReply with ONLY the topic phrase."
                    ).strip()
                    st.session_state.detected_topic = topic
                    st.info(f"📌 Detected topic: **{topic}**")
                except Exception as e:
                    st.error(f"Detection error: {e}")

    if st.session_state.detected_topic:
        st.info(f"📌 Using topic: **{st.session_state.detected_topic}**")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LITERATURE MINING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-box">', unsafe_allow_html=True)
section_header("📚 1 · Literature Mining")

col_btn, col_status = st.columns([2, 3])
with col_btn:
    run_lit = st.button("🔍 Mine Literature", key="btn_lit", use_container_width=True)
with col_status:
    if st.session_state.lit_result:
        n = len(st.session_state.lit_result.get("papers", []))
        st.success(f"✅ {n} papers indexed. Re-run to refresh.")
    elif st.session_state.input_source == "topic" and not research_query:
        st.info("ℹ️ Enter a topic in the sidebar.")
    elif st.session_state.input_source == "pdf" and not st.session_state.pdf_text:
        st.info("ℹ️ Upload a PDF above.")

if run_lit:
    if not check_api():
        pass
    elif st.session_state.input_source == "pdf" and not st.session_state.pdf_text:
        st.warning("Upload a PDF first.")
    elif st.session_state.input_source == "topic" and not research_query:
        st.warning("Enter a research topic in the sidebar.")
    else:
        with st.spinner("🔍 Fetching and indexing papers..."):
            try:
                from agents.literature_mining_agent import LiteratureMiningAgent
                agent = LiteratureMiningAgent()
                query = get_active_query()
                result = agent.mine_literature(
                    query=query, sources=list(sources), max_papers=max_papers
                )
                st.session_state.lit_result = result
                st.session_state.papers = result.get("papers", [])
                st.success(f"✅ Found {len(result.get('papers', []))} papers!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

if st.session_state.lit_result:
    lit = st.session_state.lit_result
    papers = lit.get("papers", [])
    years = [p.get("year") for p in papers if p.get("year")]

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, len(papers), "Papers Found")
    metric_card(c2, lit.get("new_papers_indexed", 0), "Newly Indexed")
    metric_card(c3, f"{min(years)}-{max(years)}" if years else "N/A", "Year Range")
    metric_card(c4, sum(p.get("citations", 0) for p in papers), "Total Citations")

    st.markdown("&nbsp;")

    from utils.visualizations import make_temporal_chart
    temporal = lit.get("temporal_distribution", {})
    if temporal:
        st.plotly_chart(
            make_temporal_chart(temporal),
            use_container_width=True,
            key="lit_temporal_chart",
        )

    col_a, col_v = st.columns(2)
    with col_a:
        st.markdown("**👥 Key Authors**")
        for a in lit.get("key_authors", [])[:6]:
            st.markdown(
                f'<div class="paper-card" style="padding:8px 12px;">'
                f'<span style="color:#93C5FD;">{a.get("name","N/A")}</span>'
                f'<span style="color:#475569;float:right;font-size:0.76rem;">'
                f'{a.get("papers",0)} papers · {a.get("total_citations",0)} cit.</span></div>',
                unsafe_allow_html=True,
            )
    with col_v:
        st.markdown("**🏛️ Key Venues**")
        for v in lit.get("key_venues", [])[:6]:
            st.markdown(
                f'<div class="paper-card" style="padding:8px 12px;">'
                f'<span style="color:#93C5FD;">{v.get("venue","N/A")}</span>'
                f'<span style="color:#475569;float:right;font-size:0.76rem;">{v.get("count",0)} papers</span>'
                f'</div>', unsafe_allow_html=True,
            )

    st.markdown("**📄 Papers**")
    filt = st.text_input("Filter", "", placeholder="Search by title or author...", key="lit_filter")
    shown = [
        p for p in papers
        if filt.lower() in p.get("title", "").lower()
        or filt.lower() in str(p.get("authors", "")).lower()
    ] if filt else papers

    for i, p in enumerate(shown[:25]):
        src = p.get("source", "arxiv")
        badge_cls = "badge-arxiv" if src == "arxiv" else "badge-ss"
        badge_txt = "ArXiv" if src == "arxiv" else "S2"
        authors_str = ", ".join(p.get("authors", [])[:3])
        st.markdown(
            f'<div class="paper-card">'
            f'<a href="{p.get("url","#")}" target="_blank" style="text-decoration:none;">'
            f'<span class="paper-title">{p.get("title","N/A")}</span></a>'
            f'<div class="paper-meta">'
            f'<span class="badge {badge_cls}">{badge_txt}</span>'
            f'<span class="badge badge-year">{p.get("year","N/A")}</span>'
            f'👤 {authors_str[:70]} · 📊 {p.get("citations",0)} cit.</div>'
            f'<div style="color:#64748B;font-size:0.76rem;margin-top:5px;">'
            f'{p.get("abstract","")[:180]}...</div></div>',
            unsafe_allow_html=True,
        )

    if lit.get("summary"):
        with st.expander("🤖 AI Literature Analysis"):
            st.markdown(lit["summary"])

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-box">', unsafe_allow_html=True)
section_header("📈 2 · Trend Analysis")

col_btn, col_status = st.columns([2, 3])
with col_btn:
    run_trend = st.button("📈 Analyze Trends", key="btn_trend", use_container_width=True)
with col_status:
    if st.session_state.trend_result:
        st.success("✅ Trend analysis complete.")
    elif not st.session_state.papers:
        st.info("ℹ️ Run Literature Mining first.")

if run_trend:
    if not check_api():
        pass
    elif not st.session_state.papers:
        st.warning("Run Literature Mining first.")
    else:
        with st.spinner("📈 Analyzing trends..."):
            try:
                from agents.trend_analysis_agent import TrendAnalysisAgent
                agent = TrendAnalysisAgent()
                result = agent.analyze_trends(
                    papers=st.session_state.papers,
                    query=get_active_query(),
                )
                st.session_state.trend_result = result
                st.success("✅ Done!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

if st.session_state.trend_result:
    trend = st.session_state.trend_result
    mom = trend.get("momentum_scores", {})

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, mom.get("momentum", "N/A"), "Momentum")
    metric_card(c2, mom.get("recent_papers_count", 0), "Recent (2022+)")
    metric_card(c3, mom.get("avg_citations", 0), "Avg Citations")
    metric_card(c4, f"{round(mom.get('recent_ratio',0)*100,1)}%", "Recent Ratio")

    from utils.visualizations import make_keyword_bubble_chart, make_topic_sunburst
    col_k, col_t = st.columns(2)
    with col_k:
        st.plotly_chart(
            make_keyword_bubble_chart(trend.get("trending_keywords", [])),
            use_container_width=True, key="trend_kw_chart",
        )
    with col_t:
        st.plotly_chart(
            make_topic_sunburst(trend.get("topics", [])),
            use_container_width=True, key="trend_sunburst_chart",
        )

    st.markdown("**🏷️ Topics**")
    for i, topic in enumerate(trend.get("topics", [])):
        with st.expander(f"Topic {i+1}: {topic.get('name','N/A')} — {topic.get('prevalence',0):.0f}%"):
            st.markdown(topic.get("description", ""))
            st.markdown("**Keywords:** " + " · ".join([f"`{k}`" for k in topic.get("keywords", [])]))

    if trend.get("trend_report"):
        with st.expander("📋 Full Trend Report"):
            st.markdown(trend["trend_report"])

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GAP IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-box">', unsafe_allow_html=True)
section_header("🎯 3 · Research Gap Identification")

col_btn, col_status = st.columns([2, 3])
with col_btn:
    run_gap = st.button("🎯 Identify Gaps", key="btn_gap", use_container_width=True)
with col_status:
    if st.session_state.gap_result:
        n = len(st.session_state.gap_result.get("gaps", []))
        st.success(f"✅ {n} gaps identified.")
    elif not st.session_state.papers:
        st.info("ℹ️ Run Literature Mining first.")

if run_gap:
    if not check_api():
        pass
    elif not st.session_state.papers:
        st.warning("Run Literature Mining first.")
    else:
        with st.spinner("🎯 Identifying research gaps..."):
            try:
                from agents.gap_identification_agent import GapIdentificationAgent
                agent = GapIdentificationAgent()
                result = agent.identify_gaps(
                    papers=st.session_state.papers,
                    query=get_active_query(),
                )
                st.session_state.gap_result = result
                st.success(f"✅ Found {len(result.get('gaps',[]))} gaps!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

if st.session_state.gap_result:
    gap = st.session_state.gap_result
    c1, c2, c3 = st.columns(3)
    metric_card(c1, len(gap.get("gaps", [])), "Gaps Found")
    metric_card(c2, f"{gap.get('gap_score', 0):.1f}/10", "Gap Score")
    metric_card(c3, len(gap.get("semantic_clusters", [])), "Clusters")

    from utils.visualizations import make_gap_opportunity_matrix
    if gap.get("opportunity_matrix"):
        st.plotly_chart(
            make_gap_opportunity_matrix(gap["opportunity_matrix"]),
            use_container_width=True, key="gap_matrix_chart",
        )

    st.markdown("**🎯 Identified Gaps**")
    for g in gap.get("gaps", []):
        feas_color = {"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"}.get(
            g.get("feasibility", "Medium"), "#F59E0B")
        direction_html = (
            f'<div style="color:#4ADE80;font-size:0.8rem;margin-top:8px;">→ {g.get("suggested_direction","")}</div>'
            if g.get("suggested_direction") else ""
        )
        st.markdown(
            f'<div class="gap-card">'
            f'<div class="gap-title">🎯 {g.get("title","N/A")}</div>'
            f'<div class="gap-desc">{g.get("description","")}</div>'
            f'<div style="margin-top:10px;">'
            f'<span class="score-pill" style="background:#1a2744;color:#4F8EF7;">⭐ Novelty: {g.get("novelty_score",0):.1f}/10</span>'
            f'<span class="score-pill" style="background:#1A2D1A;color:#10B981;">🚀 Impact: {g.get("impact_score",0):.1f}/10</span>'
            f'<span class="score-pill" style="background:#1A1A2D;color:{feas_color};">🔧 {g.get("feasibility","N/A")}</span>'
            f'</div>{direction_html}</div>',
            unsafe_allow_html=True,
        )

    if gap.get("gap_report"):
        with st.expander("📋 Full Gap Report"):
            st.markdown(gap["gap_report"])

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — METHODOLOGY DESIGN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-box">', unsafe_allow_html=True)
section_header("⚗️ 4 · Methodology Design")

# Auto-populate from top gap
default_gap_text = ""
if st.session_state.gap_result:
    gaps_list = st.session_state.gap_result.get("gaps", [])
    if gaps_list:
        default_gap_text = gaps_list[0].get("title", "")

method_gap = st.text_area(
    "Research gap to design methodology for",
    value=default_gap_text,
    placeholder="e.g., Lack of efficient few-shot methods for low-resource NLP",
    key="method_gap_input",
)

col_btn, col_status = st.columns([2, 3])
with col_btn:
    run_method = st.button("⚗️ Design Methodology", key="btn_method", use_container_width=True)
with col_status:
    if st.session_state.method_result:
        st.success("✅ Methodology designed.")

if run_method:
    if not check_api():
        pass
    elif not method_gap.strip():
        st.warning("Enter a research gap above.")
    else:
        with st.spinner("⚗️ Designing methodology..."):
            try:
                from agents.methodology_design_agent import MethodologyDesignAgent
                agent = MethodologyDesignAgent()
                result = agent.design_methodology(
                    research_gap=method_gap,
                    domain=get_active_query(),
                    papers=st.session_state.papers,
                )
                st.session_state.method_result = result
                st.success("✅ Methodology designed!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

if st.session_state.method_result:
    method = st.session_state.method_result
    tab_h, tab_d, tab_b, tab_m, tab_r = st.tabs(
        ["💡 Hypotheses", "📊 Datasets", "🏁 Baselines", "📏 Metrics", "📋 Report"])

    with tab_h:
        for h in method.get("hypotheses", []):
            with st.expander(f"{h.get('id','H?')}: {h.get('statement','')[:90]}"):
                st.markdown(f"**Type:** {h.get('type','')}")
                st.markdown(f"**Rationale:** {h.get('rationale','')}")
                st.markdown(f"**Test Approach:** {h.get('test_approach','')}")
                st.markdown(f"**Expected Outcome:** {h.get('expected_outcome','')}")

    with tab_d:
        for ds in method.get("datasets", []):
            with st.expander(f"📦 {ds.get('name','Dataset')}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Source:** {ds.get('source','')}")
                    st.markdown(f"**Size:** {ds.get('size','')}")
                    st.markdown(f"**Type:** {ds.get('type','')}")
                with c2:
                    st.markdown(f"**Use Case:** {ds.get('use_case','')}")
                    st.markdown(f"**Pros:** {ds.get('pros','')}")
                    st.markdown(f"**Cons:** {ds.get('cons','')}")

    with tab_b:
        for b in method.get("baselines", []):
            with st.expander(f"🏁 {b.get('name','Baseline')} [{b.get('type','')}]"):
                st.markdown(f"**Description:** {b.get('description','')}")
                st.markdown(f"**Strengths:** {b.get('strengths','')}")
                st.markdown(f"**Weaknesses:** {b.get('weaknesses','')}")

    with tab_m:
        for m in method.get("evaluation_metrics", []):
            with st.expander(f"📏 {m.get('name','Metric')} [{m.get('category','')}]"):
                st.markdown(f"**Formula:** `{m.get('formula','')}`")
                st.markdown(f"**Use When:** {m.get('use_when','')}")
                st.markdown(f"**Limitations:** {m.get('limitations','')}")

    with tab_r:
        st.markdown(method.get("methodology_report", ""))

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GRANT WRITING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-box">', unsafe_allow_html=True)
section_header("📝 5 · Grant Proposal Generator")

g_topic = st.text_input("Research Topic", value=get_active_query(), key="grant_topic_input")
g_gap = st.text_area("Research Gap", value=default_gap_text, key="grant_gap_input",
                     placeholder="Describe the gap your research addresses...")
g_method = st.text_area("Methodology Overview (optional)", key="grant_method_input",
                        placeholder="Brief description of your approach...")

col_btn, col_status = st.columns([2, 3])
with col_btn:
    run_grant = st.button(f"📝 Write {agency} Proposal", key="btn_grant", use_container_width=True)
with col_status:
    if st.session_state.grant_result:
        st.success(f"✅ {st.session_state.grant_result.get('agency','')} proposal ready.")

if run_grant:
    if not check_api():
        pass
    elif not g_topic.strip():
        st.warning("Enter a research topic.")
    else:
        with st.spinner(f"📝 Writing {agency} grant proposal..."):
            try:
                from agents.grant_writing_agent import GrantWritingAgent
                agent = GrantWritingAgent()
                result = agent.write_grant(
                    research_topic=g_topic,
                    research_gap=g_gap,
                    methodology=g_method,
                    agency=agency,
                    budget_range=budget_range,
                    duration=duration,
                    pi_info=pi_info(),
                )
                st.session_state.grant_result = result
                st.success("✅ Proposal generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

if st.session_state.grant_result:
    grant = st.session_state.grant_result
    tab_full, tab_exec, tab_budget, tab_impact, tab_secs = st.tabs(
        ["📋 Full Proposal", "📄 Executive Summary", "💰 Budget", "🌍 Broader Impacts", "📑 Sections"])

    with tab_full:
        full = grant.get("full_proposal", "")
        if full:
            st.download_button(
                "⬇️ Download Proposal (.txt)",
                data=full,
                file_name=f"grant_{grant.get('agency','NSF')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                key="dl_grant_btn",
            )
            st.code(full, language=None)
    with tab_exec:
        st.markdown(grant.get("executive_summary", "Not available"))
    with tab_budget:
        st.markdown(grant.get("budget_justification", "Not available"))
    with tab_impact:
        st.markdown(grant.get("broader_impacts", "Not available"))
    with tab_secs:
        for sec_name, content in grant.get("sections", {}).items():
            with st.expander(f"📌 {sec_name}"):
                st.markdown(content)

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NOVELTY SCORING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-box">', unsafe_allow_html=True)
section_header("🏆 6 · Novelty & Originality Scoring")

# Auto-fill
nov_title_default = get_active_query()
nov_abstract_default = (st.session_state.grant_result or {}).get("executive_summary", "")
if st.session_state.input_source == "pdf" and st.session_state.pdf_text:
    nov_abstract_default = st.session_state.pdf_text[:600]

n_title = st.text_input("Research Title", value=nov_title_default, key="nov_title_input")
n_abstract = st.text_area("Abstract / Description", value=nov_abstract_default[:500],
                          key="nov_abstract_input")
n_method = st.text_area("Methodology (optional)", key="nov_method_input")

col_btn, col_status = st.columns([2, 3])
with col_btn:
    run_nov = st.button("🏆 Score Novelty", key="btn_nov", use_container_width=True)
with col_status:
    if st.session_state.novelty_result:
        score = st.session_state.novelty_result.get("novelty_score", 0)
        st.success(f"✅ Novelty Score: {score:.0f}/100")

if run_nov:
    if not check_api():
        pass
    elif not n_title.strip():
        st.warning("Enter a research title.")
    else:
        with st.spinner("🏆 Scoring novelty..."):
            try:
                from agents.novelty_scoring_agent import NoveltyScoringAgent
                agent = NoveltyScoringAgent()
                result = agent.score_novelty(
                    title=n_title, abstract=n_abstract, methodology=n_method
                )
                st.session_state.novelty_result = result
                st.success(f"✅ Score: {result.get('novelty_score',0):.0f}/100")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

if st.session_state.novelty_result:
    nov = st.session_state.novelty_result
    from utils.visualizations import make_novelty_gauge

    col_g, col_i = st.columns([1, 2])
    with col_g:
        st.plotly_chart(
            make_novelty_gauge(nov.get("novelty_score", 0)),
            use_container_width=True, key="nov_gauge_chart",
        )
    with col_i:
        st.markdown(f"### {nov.get('verdict','')}")
        st.markdown(f"**Similarity to prior work:** {nov.get('similarity_score',0)*100:.1f}%")
        breakdown = nov.get("novelty_breakdown", {})
        for dim, data in breakdown.items():
            score_val = data.get("score", 0) if isinstance(data, dict) else data
            exp = data.get("explanation", "") if isinstance(data, dict) else ""
            sc = "#10B981" if score_val >= 7 else "#F59E0B" if score_val >= 5 else "#EF4444"
            st.markdown(
                f'<div class="paper-card" style="padding:8px 12px;">'
                f'<b>{dim}</b>: <span style="color:{sc};">{score_val}/10</span>'
                f'<span style="color:#64748B;font-size:0.76rem;"> — {str(exp)[:100]}</span>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("**🔍 Similar Papers**")
    for i, p in enumerate(nov.get("similar_papers", [])[:8], 1):
        sim = p.get("similarity", 0) * 100
        sc = "#EF4444" if sim > 80 else "#F59E0B" if sim > 60 else "#10B981"
        st.markdown(
            f'<div class="paper-card">'
            f'<b style="color:#93C5FD;">{i}. {p.get("title","N/A")}</b>'
            f'<span style="float:right;color:{sc};font-weight:700;">{sim:.1f}%</span>'
            f'<div class="paper-meta">📅 {p.get("year","N/A")} · 📊 {p.get("citations",0)} cit.</div>'
            f'</div>', unsafe_allow_html=True)

    unique = nov.get("uniqueness_aspects", [])
    concerns = nov.get("overlap_concerns", [])
    recs = nov.get("recommendations", [])
    if unique:
        st.success("**✅ Unique Aspects:**\n" + "\n".join(f"• {u}" for u in unique))
    if concerns:
        st.warning("**⚠️ Overlap Concerns:**\n" + "\n".join(f"• {c}" for c in concerns))
    if recs:
        st.info("**💡 Recommendations:**\n" + "\n".join(f"→ {r}" for r in recs))

    if nov.get("novelty_report"):
        with st.expander("📋 Full Novelty Report"):
            st.markdown(nov["novelty_report"])

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CITATION NETWORK & SEMANTIC SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-box">', unsafe_allow_html=True)
section_header("🕸️ 7 · Citation Network & Semantic Search")

col_net, col_search = st.columns(2)

with col_net:
    st.markdown("**Citation Network**")
    if st.session_state.papers:
        max_nodes = st.slider("Max nodes", 10, 40, 20, key="net_nodes_slider")
        if st.button("🕸️ Build Network", key="btn_net"):
            with st.spinner("Building..."):
                try:
                    from utils.visualizations import make_citation_network
                    html = make_citation_network(st.session_state.papers, max_nodes=max_nodes)
                    st.components.v1.html(html, height=480, scrolling=True)
                except Exception as e:
                    st.error(f"Network error: {e}")
    else:
        st.info("Run Literature Mining first.")

with col_search:
    st.markdown("**Semantic Search**")
    search_q = st.text_input("Search indexed papers", placeholder="e.g., attention mechanisms",
                             key="sem_search_input")
    n_res = st.slider("Results", 3, 15, 6, key="sem_n_slider")
    if st.button("🔍 Search", key="btn_search") and search_q:
        if check_api():
            try:
                from core.vector_store import get_vector_store
                vs = get_vector_store()
                results = vs.semantic_search(search_q, n_results=n_res)
                if results:
                    for r in results:
                        st.markdown(
                            f'<div class="paper-card">'
                            f'<b style="color:#93C5FD;">{r.get("title","N/A")}</b>'
                            f'<span style="float:right;color:#10B981;font-weight:700;">'
                            f'{r.get("similarity",0)*100:.1f}%</span>'
                            f'<div class="paper-meta">📅 {r.get("year","N/A")} · '
                            f'📊 {r.get("citations",0)} cit.</div></div>',
                            unsafe_allow_html=True)
                else:
                    st.warning("No results. Run Literature Mining first.")
            except Exception as e:
                st.error(f"Search error: {e}")

st.markdown('</div>', unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.76rem;padding:16px 0 8px;">
    🔬 <b>ResearchIQ</b> · Multi-Agent Research Intelligence ·
    Powered by <b>Gemini AI</b> · <b>ChromaDB</b> · <b>ArXiv</b> · <b>Semantic Scholar</b>
</div>
""", unsafe_allow_html=True)
