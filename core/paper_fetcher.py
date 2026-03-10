"""
ResearchIQ - Paper Fetcher
============================
Fetches research papers from ArXiv and Semantic Scholar APIs.
"""

import time
import logging
import requests
from typing import List, Dict, Optional
import arxiv
from config.settings import SEMANTIC_SCHOLAR_API_KEY, MAX_PAPERS_PER_QUERY

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"


class ArXivFetcher:
    """Fetches papers from ArXiv."""

    def search(self, query: str, max_results: int = MAX_PAPERS_PER_QUERY) -> List[Dict]:
        """Search ArXiv for papers matching the query."""
        papers = []
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for result in client.results(search):
                papers.append({
                    "id": result.entry_id.split("/")[-1],
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [a.name for a in result.authors],
                    "year": result.published.year if result.published else None,
                    "venue": "ArXiv",
                    "url": result.entry_id,
                    "citations": 0,
                    "categories": result.categories,
                    "source": "arxiv",
                })
        except Exception as e:
            logger.error(f"ArXiv fetch error: {e}")
        return papers


class SemanticScholarFetcher:
    """Fetches papers from Semantic Scholar API."""

    def __init__(self):
        self.headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            self.headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def search(self, query: str, max_results: int = MAX_PAPERS_PER_QUERY) -> List[Dict]:
        """Search Semantic Scholar for papers."""
        papers = []
        try:
            url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
            params = {
                "query": query,
                "limit": min(max_results, 100),
                "fields": "title,abstract,authors,year,venue,externalIds,citationCount,openAccessPdf",
            }
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            for p in data.get("data", []):
                if not p.get("title") or not p.get("abstract"):
                    continue
                pid = p.get("paperId", "")
                ext_ids = p.get("externalIds", {}) or {}
                paper_url = (
                    (p.get("openAccessPdf") or {}).get("url")
                    or f"https://www.semanticscholar.org/paper/{pid}"
                )
                papers.append({
                    "id": f"ss_{pid}",
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "authors": [a.get("name", "") for a in (p.get("authors") or [])],
                    "year": p.get("year"),
                    "venue": (p.get("venue") or ""),
                    "url": paper_url,
                    "citations": p.get("citationCount", 0),
                    "categories": [],
                    "source": "semantic_scholar",
                })
                time.sleep(0.1)  # Rate limiting

        except Exception as e:
            logger.error(f"Semantic Scholar fetch error: {e}")
        return papers

    def get_citations(self, paper_id: str) -> List[Dict]:
        """Get papers that cite the given paper."""
        citations = []
        try:
            url = f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/citations"
            params = {"fields": "title,year,authors,citationCount", "limit": 50}
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                p = item.get("citingPaper", {})
                if p.get("title"):
                    citations.append({
                        "id": p.get("paperId", ""),
                        "title": p.get("title", ""),
                        "year": p.get("year"),
                        "citations": p.get("citationCount", 0),
                    })
        except Exception as e:
            logger.error(f"Citation fetch error: {e}")
        return citations


class PaperFetcher:
    """Unified paper fetcher combining multiple sources."""

    def __init__(self):
        self.arxiv = ArXivFetcher()
        self.semantic_scholar = SemanticScholarFetcher()

    def fetch(
        self,
        query: str,
        sources: List[str] = ("arxiv", "semantic_scholar"),
        max_per_source: int = 15,
    ) -> List[Dict]:
        """Fetch papers from specified sources."""
        all_papers = []
        seen_titles = set()

        if "arxiv" in sources:
            logger.info(f"Fetching from ArXiv: {query}")
            papers = self.arxiv.search(query, max_results=max_per_source)
            for p in papers:
                title_key = p["title"].lower().strip()[:100]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_papers.append(p)

        if "semantic_scholar" in sources:
            logger.info(f"Fetching from Semantic Scholar: {query}")
            papers = self.semantic_scholar.search(query, max_results=max_per_source)
            for p in papers:
                title_key = p["title"].lower().strip()[:100]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_papers.append(p)

        logger.info(f"Fetched {len(all_papers)} unique papers for: {query}")
        return all_papers
