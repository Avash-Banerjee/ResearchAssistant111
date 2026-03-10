"""
ResearchIQ - Literature Mining Agent (v2 - RAG Grounded)
=========================================================
Every LLM generation call is grounded in retrieved paper chunks from ChromaDB.
No more Gemini hallucinating from training data — all claims come from indexed papers.

RAG usage:
  _generate_summary()     → retrieves chunks for "overview themes methodology"
  _generate_section()     → sub-queries for each report section individually
"""

import logging
from typing import List, Dict, Optional, Callable
from core.llm_client import get_llm
from core.vector_store import get_vector_store
from core.paper_fetcher import PaperFetcher
from config.settings import AGENT_CONFIGS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Senior Research Librarian and Literature Mining Expert.
Analyse research papers and provide structured, evidence-grounded insights.
Every claim you make must be traceable to a specific paper in the provided context.
Use precise academic language and cite paper titles explicitly."""


class LiteratureMiningAgent:
    """
    Agent 1: Literature Mining (RAG-grounded)
    - Crawls ArXiv and Semantic Scholar
    - Indexes papers + RAG chunks in ChromaDB
    - Generates literature summary grounded in retrieved chunks
    - Identifies key authors, venues, temporal distribution
    """

    def __init__(self):
        self.llm          = get_llm()
        self.vector_store = get_vector_store()
        self.fetcher      = PaperFetcher()
        self.config       = AGENT_CONFIGS["literature_miner"]

    def mine_literature(
        self,
        query: str,
        sources: List[str] = ("arxiv", "semantic_scholar"),
        max_papers: int = 20,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:

        result = {
            "query":                query,
            "papers":               [],
            "new_papers_indexed":   0,
            "summary":              "",
            "key_authors":          [],
            "key_venues":           [],
            "temporal_distribution":{},
            "rag_chunks_indexed":   0,
            "agent":                self.config["name"],
        }

        # ── Step 1: Fetch ─────────────────────────────────────────────────
        if progress_callback:
            progress_callback("🔍 Fetching papers from ArXiv and Semantic Scholar...")

        papers = self.fetcher.fetch(
            query=query,
            sources=list(sources),
            max_per_source=max_papers,
        )
        result["papers"] = papers

        if not papers:
            result["summary"] = "No papers found. Try broadening your search terms."
            return result

        # ── Step 2: Index (papers + RAG chunks) ───────────────────────────
        if progress_callback:
            progress_callback(
                f"📦 Indexing {len(papers)} papers + chunking for RAG..."
            )
        result["new_papers_indexed"] = self.vector_store.add_papers(papers)
        stats = self.vector_store.get_stats()
        result["rag_chunks_indexed"] = stats.get("total_chunks", 0)
        logger.info(
            f"Indexed {result['new_papers_indexed']} new papers. "
            f"Total chunks: {result['rag_chunks_indexed']}"
        )

        # ── Step 3: Metadata ───────────────────────────────────────────────
        result["key_authors"]          = self._extract_key_authors(papers)
        result["key_venues"]           = self._extract_key_venues(papers)
        result["temporal_distribution"]= self._temporal_distribution(papers)

        # ── Step 4: RAG-grounded summary ───────────────────────────────────
        if progress_callback:
            progress_callback("🤖 Generating RAG-grounded literature analysis...")
        result["summary"] = self._generate_rag_summary(query, papers)

        return result

    # ─── RAG-Grounded Summary ─────────────────────────────────────────────────

    def _generate_rag_summary(self, query: str, papers: List[Dict]) -> str:
        """
        Generate literature summary fully grounded in retrieved paper chunks.

        Runs 5 targeted sub-queries against the chunk collection so each
        section of the summary has dedicated retrieved evidence.
        """

        # Sub-queries → each maps to a report section
        sub_queries = {
            "overview":      f"{query} overview landscape introduction",
            "themes":        f"{query} main themes research directions",
            "methods":       f"{query} methodology techniques approaches algorithms",
            "findings":      f"{query} key results findings contributions",
            "limitations":   f"{query} limitations challenges future work open problems",
        }

        section_contexts: Dict[str, str] = {}
        for section, sq in sub_queries.items():
            ctx = self.vector_store.rag_retrieve(sq, n_chunks=5, max_per_paper=2)
            section_contexts[section] = ctx or "(no relevant chunks retrieved)"

        # Also build a quick paper list for the "recommended reading" section
        top_papers = sorted(papers, key=lambda p: p.get("citations", 0), reverse=True)[:8]
        top_list   = "\n".join(
            f'- "{p.get("title","")}" ({p.get("year","")}) — {p.get("citations",0)} citations'
            for p in top_papers
        )

        prompt = f"""Write a comprehensive Literature Analysis for the research domain: "{query}"

You MUST ground every claim in the RETRIEVED CONTEXT sections below.
Cite papers by title and year. Do not invent papers or facts not found in the context.

═══════════════════════════════════════════════
CONTEXT — OVERVIEW & LANDSCAPE:
{section_contexts['overview']}

CONTEXT — MAIN THEMES:
{section_contexts['themes']}

CONTEXT — METHODS & TECHNIQUES:
{section_contexts['methods']}

CONTEXT — KEY FINDINGS:
{section_contexts['findings']}

CONTEXT — LIMITATIONS & OPEN PROBLEMS:
{section_contexts['limitations']}

TOP CITED PAPERS (by citation count):
{top_list}
═══════════════════════════════════════════════

Write the following sections using ONLY information from the retrieved context above:

## 1. Research Landscape Overview
What is the general state of this field? What problems are being addressed?

## 2. Major Research Themes
Identify and describe 3–5 major themes. For each, cite specific papers.

## 3. Key Contributions & Milestones
Which papers made the most significant contributions and what did they achieve?

## 4. Dominant Methodologies
What techniques and approaches are most widely used? Cite papers that use them.

## 5. Research Consensus & Debates
Where do papers agree? Where do they disagree or point to contradictions?

## 6. Open Challenges
What limitations and unsolved problems do papers themselves identify?

## 7. Recommended Starting Points
Which 3–5 papers should a new researcher read first, and why?

Maintain academic rigour. Be specific. Cite paper titles throughout."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.4)

    # ─── Metadata Extraction ──────────────────────────────────────────────────

    def semantic_search(self, query: str, n_results: int = 10) -> List[Dict]:
        return self.vector_store.semantic_search(query, n_results=n_results)

    def _extract_key_authors(self, papers: List[Dict]) -> List[Dict]:
        author_stats: Dict[str, Dict] = {}
        for paper in papers:
            citations = paper.get("citations", 0)
            for author in paper.get("authors", [])[:5]:
                if not author:
                    continue
                if author not in author_stats:
                    author_stats[author] = {"papers": 0, "total_citations": 0}
                author_stats[author]["papers"]          += 1
                author_stats[author]["total_citations"] += citations
        return sorted(
            [{"name": k, **v} for k, v in author_stats.items()],
            key=lambda x: x["total_citations"],
            reverse=True,
        )[:10]

    def _extract_key_venues(self, papers: List[Dict]) -> List[Dict]:
        venue_counts: Dict[str, int] = {}
        for paper in papers:
            venue = paper.get("venue", "")
            if venue and venue.lower() not in ("", "unknown", "none"):
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
        return sorted(
            [{"venue": k, "count": v} for k, v in venue_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

    def _temporal_distribution(self, papers: List[Dict]) -> Dict:
        dist: Dict[str, int] = {}
        for paper in papers:
            year = paper.get("year")
            if year:
                dist[str(year)] = dist.get(str(year), 0) + 1
        return dict(sorted(dist.items()))
