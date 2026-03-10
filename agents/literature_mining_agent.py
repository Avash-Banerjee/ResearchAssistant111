"""
ResearchIQ - Literature Mining Agent
======================================
Crawls research repositories, builds embeddings, and structures literature bases.
"""

import logging
from typing import List, Dict, Optional, Callable
from core.llm_client import get_llm
from core.vector_store import get_vector_store
from core.paper_fetcher import PaperFetcher
from config.settings import AGENT_CONFIGS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Senior Research Librarian and Literature Mining Expert.
Your task is to analyze research papers and provide structured insights about the literature.
Always respond with precise, academically rigorous analysis.
Format your output using clear headings and bullet points where appropriate."""


class LiteratureMiningAgent:
    """
    Agent 1: Literature Mining
    - Crawls ArXiv and Semantic Scholar
    - Builds vector embeddings with ChromaDB
    - Summarizes literature landscapes
    - Identifies key papers and authors
    """

    def __init__(self):
        self.llm = get_llm()
        self.vector_store = get_vector_store()
        self.fetcher = PaperFetcher()
        self.config = AGENT_CONFIGS["literature_miner"]

    def mine_literature(
        self,
        query: str,
        sources: List[str] = ("arxiv", "semantic_scholar"),
        max_papers: int = 20,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Main entry point: mine literature for a given research query.
        Returns structured literature analysis.
        """
        result = {
            "query": query,
            "papers": [],
            "new_papers_indexed": 0,
            "summary": "",
            "key_authors": [],
            "key_venues": [],
            "temporal_distribution": {},
            "agent": self.config["name"],
        }

        # Step 1: Fetch papers
        if progress_callback:
            progress_callback("🔍 Fetching papers from repositories...")

        papers = self.fetcher.fetch(
            query=query,
            sources=list(sources),
            max_per_source=max_papers // len(sources),
        )
        result["papers"] = papers

        if not papers:
            result["summary"] = "No papers found for the given query. Try broadening your search terms."
            return result

        # Step 2: Index in vector store
        if progress_callback:
            progress_callback(f"📦 Indexing {len(papers)} papers in vector store...")
        result["new_papers_indexed"] = self.vector_store.add_papers(papers)

        # Step 3: Extract metadata insights
        result["key_authors"] = self._extract_key_authors(papers)
        result["key_venues"] = self._extract_key_venues(papers)
        result["temporal_distribution"] = self._temporal_distribution(papers)

        # Step 4: Generate LLM summary
        if progress_callback:
            progress_callback("🤖 Generating literature analysis...")
        result["summary"] = self._generate_summary(query, papers)

        return result

    def semantic_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search indexed literature semantically."""
        return self.vector_store.semantic_search(query, n_results=n_results)

    def _generate_summary(self, query: str, papers: List[Dict]) -> str:
        """Generate a comprehensive literature summary using Gemini."""
        paper_texts = []
        for i, p in enumerate(papers[:15], 1):
            paper_texts.append(
                f"{i}. **{p['title']}** ({p.get('year', 'N/A')})\n"
                f"   Authors: {', '.join(p.get('authors', [])[:3])}\n"
                f"   Citations: {p.get('citations', 0)}\n"
                f"   Abstract: {p.get('abstract', '')[:300]}..."
            )

        prompt = f"""Analyze the following {len(papers)} research papers on the topic: "{query}"

PAPERS:
{chr(10).join(paper_texts)}

Provide a comprehensive literature analysis including:

## 1. Research Landscape Overview
[Describe the overall state of research in this area]

## 2. Major Research Themes
[List and describe the 3-5 main research themes found]

## 3. Key Contributions
[Highlight the most impactful papers and what they contributed]

## 4. Research Evolution
[How has this research area evolved over time?]

## 5. Dominant Methodologies
[What methods/approaches are most commonly used?]

## 6. Research Consensus & Debates
[What do researchers agree on? Where are the debates?]

## 7. Recommended Starting Points
[Which 3-5 papers should a new researcher read first and why?]

Be specific, cite paper titles, and maintain academic rigor."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.5)

    def _extract_key_authors(self, papers: List[Dict]) -> List[Dict]:
        """Extract most prolific/cited authors."""
        author_stats = {}
        for paper in papers:
            citations = paper.get("citations", 0)
            for author in paper.get("authors", [])[:5]:
                if author not in author_stats:
                    author_stats[author] = {"papers": 0, "total_citations": 0}
                author_stats[author]["papers"] += 1
                author_stats[author]["total_citations"] += citations

        return sorted(
            [{"name": k, **v} for k, v in author_stats.items()],
            key=lambda x: x["total_citations"],
            reverse=True,
        )[:10]

    def _extract_key_venues(self, papers: List[Dict]) -> List[Dict]:
        """Extract most common publication venues."""
        venue_counts = {}
        for paper in papers:
            venue = paper.get("venue", "Unknown")
            if venue and venue != "Unknown":
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
        return sorted(
            [{"venue": k, "count": v} for k, v in venue_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

    def _temporal_distribution(self, papers: List[Dict]) -> Dict:
        """Count papers per year."""
        dist = {}
        for paper in papers:
            year = paper.get("year")
            if year:
                dist[str(year)] = dist.get(str(year), 0) + 1
        return dict(sorted(dist.items()))
