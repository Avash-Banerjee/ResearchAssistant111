"""
ResearchIQ - Core Configuration
================================
Central configuration management for the Research Intelligence System.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
TEMPLATES_DIR = BASE_DIR / "templates"

for d in [DATA_DIR, LOGS_DIR, DATA_DIR / "chroma_db", DATA_DIR / "exports"]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Keys ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# ─── Model Config ─────────────────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "research_papers")

# ─── Application ──────────────────────────────────────────────────────────────
APP_TITLE = "ResearchIQ"
APP_SUBTITLE = "Multi-Agent Research Intelligence Framework"
MAX_PAPERS_PER_QUERY = int(os.getenv("MAX_PAPERS_PER_QUERY", "20"))

# ─── Agent Configs ────────────────────────────────────────────────────────────
AGENT_CONFIGS = {
    "literature_miner": {
        "name": "Literature Mining Agent",
        "role": "Senior Research Librarian & Data Crawler",
        "goal": "Discover and index the most relevant research papers from ArXiv and Semantic Scholar",
        "backstory": (
            "You are an expert research librarian with 20 years of experience in "
            "academic literature. You excel at finding the most relevant papers, "
            "understanding research domains deeply, and building comprehensive literature bases."
        ),
        "icon": "🔍",
        "color": "#4F8EF7",
    },
    "trend_analyzer": {
        "name": "Trend Analysis Agent",
        "role": "Research Trend Analyst & Topic Modeler",
        "goal": "Identify emerging research trends and topic evolution using advanced NLP techniques",
        "backstory": (
            "You are a data scientist specializing in academic trend analysis. "
            "You use BERTopic and semantic analysis to identify how research fields evolve, "
            "spot emerging topics, and quantify research momentum."
        ),
        "icon": "📈",
        "color": "#10B981",
    },
    "gap_identifier": {
        "name": "Gap Identification Agent",
        "role": "Research Gap Analyst & Citation Network Expert",
        "goal": "Detect under-explored research intersections and identify high-impact research opportunities",
        "backstory": (
            "You are a research strategist who maps citation networks to find white spaces "
            "in academic literature. You excel at identifying where research is sparse, "
            "contradictory, or ripe for breakthrough contributions."
        ),
        "icon": "🎯",
        "color": "#F59E0B",
    },
    "methodology_designer": {
        "name": "Methodology Design Agent",
        "role": "Research Methodology Expert",
        "goal": "Suggest optimal experimental designs, datasets, evaluation metrics, and baselines",
        "backstory": (
            "You are an experienced researcher who has designed hundreds of experiments. "
            "You know which datasets are most appropriate, which metrics matter, "
            "and how to structure rigorous experimental comparisons."
        ),
        "icon": "⚗️",
        "color": "#8B5CF6",
    },
    "grant_writer": {
        "name": "Grant Writing Agent",
        "role": "Expert Grant Proposal Writer",
        "goal": "Generate compelling, funding-ready grant proposals aligned with agency guidelines",
        "backstory": (
            "You are a professional grant writer who has secured over $50M in research funding. "
            "You craft clear, compelling narratives that highlight innovation, feasibility, "
            "and societal impact for NSF, NIH, DARPA, and other agencies."
        ),
        "icon": "📝",
        "color": "#EF4444",
    },
    "novelty_scorer": {
        "name": "Novelty Scoring Agent",
        "role": "Research Novelty & Plagiarism Analyst",
        "goal": "Compute semantic novelty scores and detect similarity with prior work",
        "backstory": (
            "You are a research integrity expert who evaluates the originality of research. "
            "Using semantic analysis, you compare proposed work against existing literature "
            "to provide novelty scores and identify potential plagiarism."
        ),
        "icon": "🏆",
        "color": "#06B6D4",
    },
}

# ─── Grant Templates ──────────────────────────────────────────────────────────
GRANT_AGENCIES = {
    "NSF": "National Science Foundation",
    "NIH": "National Institutes of Health",
    "DARPA": "Defense Advanced Research Projects Agency",
    "DOE": "Department of Energy",
    "EU Horizon": "European Research Council (Horizon Europe)",
    "Custom": "Custom Agency Template",
}

PAPER_FORMATS = ["IEEE", "ACM", "NeurIPS", "Nature", "Custom"]

# ─── Visualization ────────────────────────────────────────────────────────────
CHART_THEME = "plotly_dark"
COLOR_PALETTE = ["#4F8EF7", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4"]
