# 🔬 ResearchIQ - Multi-Agent Research Intelligence Framework

> AI-powered research assistant for literature discovery, gap identification, methodology design, and grant proposal generation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)

---

## 🏗️ Architecture

```
ResearchIQ/
├── app.py                          # Main Streamlit UI
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment config template
├── .streamlit/
│   └── config.toml                 # Streamlit theme config
│
├── config/
│   └── settings.py                 # Central configuration
│
├── core/
│   ├── llm_client.py               # Gemini API client
│   ├── vector_store.py             # ChromaDB vector store
│   ├── paper_fetcher.py            # ArXiv + Semantic Scholar
│   └── crew_orchestrator.py        # Multi-agent pipeline
│
├── agents/
│   ├── literature_mining_agent.py  # Agent 1: Literature
│   ├── trend_analysis_agent.py     # Agent 2: Trends
│   ├── gap_identification_agent.py # Agent 3: Gaps
│   ├── methodology_design_agent.py # Agent 4: Methodology
│   ├── grant_writing_agent.py      # Agent 5: Grant Writing
│   └── novelty_scoring_agent.py    # Agent 6: Novelty Scoring
│
└── utils/
    └── visualizations.py           # Plotly charts & graphs
```

---

## 🤖 Agent Roles

| Agent | Role | Key Capabilities |
|-------|------|-----------------|
| 🔍 Literature Miner | Senior Research Librarian | ArXiv + S2 crawling, embeddings, literature summary |
| 📈 Trend Analyst | Research Trend Expert | BERTopic-style analysis, keyword trending, momentum scoring |
| 🎯 Gap Identifier | Research Gap Analyst | Semantic clustering, citation analysis, opportunity matrix |
| ⚗️ Methodology Designer | Research Methods Expert | Hypothesis generation, dataset/baseline recommendations |
| 📝 Grant Writer | Expert Grant Writer | NSF/NIH/DARPA proposals, budget justification |
| 🏆 Novelty Scorer | Research Integrity Expert | Semantic similarity, novelty scoring, plagiarism detection |

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- Gemini API key ([get here](https://makersuite.google.com/app/apikey))

### 2. Installation

```bash
# Clone or extract the project
cd research_intelligence

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🎮 Usage Guide

### Full Pipeline (Recommended)
1. Enter your **Gemini API key** in the sidebar
2. Type a **research topic** (e.g., "Federated Learning for Healthcare")
3. Configure grant settings (agency, budget, duration)
4. Click **🚀 Run Full Pipeline**
5. Navigate tabs to explore all 6 agent outputs

### Individual Agents
Each tab has a "Run ... Only" expander to run individual agents:
- **Literature tab** → Mine papers for any query
- **Gaps tab** → Identify gaps from current papers
- **Grant tab** → Write custom proposals
- **Novelty tab** → Score any research idea

---

## 📊 Features

### Dashboard
- Pipeline execution status
- Key metrics: papers found, gaps identified, novelty score
- Publication timeline chart
- Executive summary

### Literature Mining
- Real-time ArXiv and Semantic Scholar search
- ChromaDB vector indexing with sentence transformers
- Key author and venue analysis
- AI-generated literature landscape summary

### Trend Analysis
- Keyword frequency and trend scoring
- Topic modeling with LLM-based extraction
- Research momentum indicators (High/Medium/Low)
- Topic sunburst visualization

### Gap Identification
- Opportunity matrix (Novelty × Impact quadrants)
- Semantic cluster analysis
- Gap scoring with feasibility assessment
- Explainable gap evidence

### Methodology Design
- Testable hypothesis generation
- Dataset recommendations with pros/cons
- Baseline method identification
- Evaluation metrics recommendations
- Publication-ready methodology section

### Grant Writing
- Full proposals for NSF, NIH, DARPA, DOE, EU Horizon
- Section-by-section generation
- Budget justification narrative
- Broader impacts statement
- Downloadable .txt output

### Novelty Scoring
- 0-100 novelty score with gauge visualization
- Multi-dimensional breakdown (Problem/Method/Application)
- Similar paper detection
- Actionable enhancement recommendations

### Citation Network
- Interactive pyvis network visualization
- Semantic paper search
- Node sizing by citation count

---

## 🔧 Configuration

### Environment Variables (`.env`)
```
GEMINI_API_KEY=your_key_here
SEMANTIC_SCHOLAR_API_KEY=optional_for_higher_limits
MAX_PAPERS_PER_QUERY=20
EMBEDDING_MODEL=all-MiniLM-L6-v2
GEMINI_MODEL=gemini-1.5-flash
```

### Model Selection
Change `GEMINI_MODEL` in `.env` for different capabilities:
- `gemini-1.5-flash` — Fast, cost-effective (default)
- `gemini-1.5-pro` — More capable, slower

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `google-generativeai` | Gemini LLM |
| `chromadb` | Vector database |
| `sentence-transformers` | Text embeddings |
| `arxiv` | ArXiv paper fetching |
| `requests` | Semantic Scholar API |
| `plotly` | Interactive charts |
| `networkx` + `pyvis` | Citation network |
| `scikit-learn` | KMeans clustering |
| `bertopic` | Topic modeling (optional) |
| `tenacity` | API retry logic |

---

## 🚀 Performance Tips

1. **Start with ArXiv only** for faster results (fewer API calls)
2. **Reduce max papers** to 10-15 for quick testing
3. **Gemini Flash** is faster than Pro for most use cases
4. Papers are **cached in ChromaDB** — repeat queries are instant
5. Run individual agents if you don't need the full pipeline

---

## 📄 License

MIT License - Free for academic and research use.

---

## 🙏 Acknowledgments

Built with:
- [Google Gemini](https://deepmind.google/technologies/gemini/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [ArXiv API](https://arxiv.org/help/api/)
- [Semantic Scholar API](https://www.semanticscholar.org/product/api)
