"""
ResearchIQ - Gap Identification Agent
========================================
Identifies research gaps using semantic clustering and citation analysis.
"""

import logging
import json
import numpy as np
from typing import List, Dict, Optional, Callable
from core.llm_client import get_llm
from core.vector_store import get_vector_store
from config.settings import AGENT_CONFIGS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Research Gap Analyst with deep expertise in identifying 
under-explored research intersections and high-impact opportunities.
You provide rigorous, evidence-based gap analysis with clear novelty justifications.
Your recommendations are specific, actionable, and grounded in the existing literature."""


class GapIdentificationAgent:
    """
    Agent 3: Research Gap Identification
    - Semantic clustering to find sparse areas
    - Citation network analysis
    - Cross-domain intersection identification
    - Explainable gap scoring
    """

    def __init__(self):
        self.llm = get_llm()
        self.vector_store = get_vector_store()
        self.config = AGENT_CONFIGS["gap_identifier"]

    def identify_gaps(
        self,
        papers: List[Dict],
        query: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """Identify research gaps from the paper collection."""

        result = {
            "query": query,
            "gaps": [],
            "citation_clusters": [],
            "semantic_clusters": [],
            "gap_score": 0.0,
            "opportunity_matrix": [],
            "gap_report": "",
            "agent": self.config["name"],
        }

        if not papers:
            result["gap_report"] = "No papers available for gap analysis."
            return result

        if progress_callback:
            progress_callback("🔗 Building citation network...")
        result["citation_clusters"] = self._analyze_citation_patterns(papers)

        if progress_callback:
            progress_callback("🧩 Performing semantic clustering...")
        result["semantic_clusters"] = self._semantic_clustering(papers, query)

        if progress_callback:
            progress_callback("🎯 Identifying research gaps...")
        gaps_raw = self._identify_gaps_via_llm(papers, query)
        result["gaps"] = gaps_raw

        if progress_callback:
            progress_callback("📋 Computing gap scores and opportunity matrix...")
        result["gap_score"] = self._compute_overall_gap_score(papers, gaps_raw)
        result["opportunity_matrix"] = self._build_opportunity_matrix(gaps_raw)

        if progress_callback:
            progress_callback("📝 Generating gap analysis report...")
        result["gap_report"] = self._generate_gap_report(query, papers, result)

        return result

    # ─── Citation Pattern Analysis ────────────────────────────────────────────

    def _analyze_citation_patterns(self, papers: List[Dict]) -> List[Dict]:
        """Analyze citation patterns to find under-cited areas."""
        clusters = []
        decade_groups = {}
        for paper in papers:
            year = paper.get("year") or 0
            decade = f"{(year // 5) * 5}s" if year else "Unknown"
            if decade not in decade_groups:
                decade_groups[decade] = []
            decade_groups[decade].append(paper)

        for decade, group in decade_groups.items():
            avg_citations = np.mean([p.get("citations", 0) for p in group])
            clusters.append({
                "period": decade,
                "paper_count": len(group),
                "avg_citations": round(float(avg_citations), 1),
                "is_under_cited": avg_citations < 10 and len(group) > 2,
                "papers": [p.get("title", "")[:60] for p in group[:3]],
            })

        return sorted(clusters, key=lambda x: x["avg_citations"])

    # ─── Semantic Clustering ──────────────────────────────────────────────────

    def _semantic_clustering(self, papers: List[Dict], query: str) -> List[Dict]:
        """Cluster papers semantically to find sparse regions."""
        if len(papers) < 3:
            return []
        try:
            texts = [
                f"{p.get('title', '')} {p.get('abstract', '')[:200]}"
                for p in papers
            ]
            embeddings = np.array(self.vector_store.embed(texts))
            n_clusters = min(5, len(papers) // 3)
            return self._simple_clustering(papers, embeddings, n_clusters)
        except Exception as e:
            logger.error(f"Semantic clustering error: {e}")
            return []

    def _simple_clustering(
        self, papers: List[Dict], embeddings: np.ndarray, n_clusters: int
    ) -> List[Dict]:
        """KMeans centroid-based clustering."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize

            normalized = normalize(embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(normalized)

            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = {"papers": [], "size": 0}
                clusters[label]["papers"].append(papers[i])
                clusters[label]["size"] += 1

            result = []
            for cid, cluster in clusters.items():
                cluster_papers = cluster["papers"]
                avg_citations = np.mean([p.get("citations", 0) for p in cluster_papers])
                years = [p.get("year") for p in cluster_papers if p.get("year")]
                result.append({
                    "cluster_id": int(cid),
                    "size": cluster["size"],
                    "density": (
                        "sparse" if cluster["size"] <= 2
                        else "dense" if cluster["size"] >= 6
                        else "medium"
                    ),
                    "avg_citations": round(float(avg_citations), 1),
                    "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
                    "sample_titles": [p.get("title", "")[:70] for p in cluster_papers[:3]],
                    "gap_potential": (
                        "High" if cluster["size"] <= 2 and float(avg_citations) < 20
                        else "Medium"
                    ),
                })
            return sorted(result, key=lambda x: x["size"])

        except ImportError:
            logger.warning("sklearn not available, skipping clustering")
            return []

    # ─── LLM Gap Identification ───────────────────────────────────────────────

    def _identify_gaps_via_llm(self, papers: List[Dict], query: str) -> List[Dict]:
        """Use Gemini to identify specific research gaps. Tries structured format
        first, falls back to JSON if parsing yields nothing."""

        paper_summaries = []
        for i, p in enumerate(papers[:20], 1):
            paper_summaries.append(
                f"{i}. {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) "
                f"[Citations: {p.get('citations', 0)}]\n"
                f"   {p.get('abstract', '')[:200]}..."
            )

        prompt = f"""You are analyzing research papers on "{query}" to identify RESEARCH GAPS.

EXISTING RESEARCH ({len(papers)} papers analyzed):
{chr(10).join(paper_summaries)}

YOU MUST respond using EXACTLY this repeated block format. Output ONLY these structured blocks with no prose, no headers, no extra text before or after.

GAP_ID: 1
GAP_TITLE: [5-10 word concise gap name]
DESCRIPTION: [2-3 sentences describing exactly what is missing in the literature]
EVIDENCE: [cite specific paper titles or patterns that reveal this gap]
NOVELTY_SCORE: 8
IMPACT_SCORE: 9
FEASIBILITY: High
SUGGESTED_DIRECTION: [1-2 sentences on how to address this gap]
RELATED_AREAS: keyword1, keyword2, keyword3
---
GAP_ID: 2
GAP_TITLE: [5-10 word concise gap name]
DESCRIPTION: [2-3 sentences]
EVIDENCE: [evidence]
NOVELTY_SCORE: 7
IMPACT_SCORE: 8
FEASIBILITY: Medium
SUGGESTED_DIRECTION: [direction]
RELATED_AREAS: keyword1, keyword2, keyword3
---
GAP_ID: 3
GAP_TITLE: [title]
DESCRIPTION: [description]
EVIDENCE: [evidence]
NOVELTY_SCORE: 9
IMPACT_SCORE: 7
FEASIBILITY: Medium
SUGGESTED_DIRECTION: [direction]
RELATED_AREAS: keyword1, keyword2, keyword3
---
GAP_ID: 4
GAP_TITLE: [title]
DESCRIPTION: [description]
EVIDENCE: [evidence]
NOVELTY_SCORE: 6
IMPACT_SCORE: 8
FEASIBILITY: High
SUGGESTED_DIRECTION: [direction]
RELATED_AREAS: keyword1, keyword2, keyword3
---
GAP_ID: 5
GAP_TITLE: [title]
DESCRIPTION: [description]
EVIDENCE: [evidence]
NOVELTY_SCORE: 8
IMPACT_SCORE: 6
FEASIBILITY: Low
SUGGESTED_DIRECTION: [direction]
RELATED_AREAS: keyword1, keyword2, keyword3
---

Replace ALL placeholder text in brackets with real content about "{query}".
Use exact key names. Every block must end with ---.
Do NOT write any text outside the blocks."""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.2)
        logger.debug(f"Gap LLM raw response (first 500 chars):\n{response[:500]}")

        gaps = self._parse_gaps(response)

        if not gaps:
            logger.warning("Structured parse yielded 0 gaps — trying JSON fallback")
            gaps = self._json_fallback(response, query, papers)

        logger.info(f"Final gap count: {len(gaps)}")
        return gaps

    # ─── Structured Parser ────────────────────────────────────────────────────

    def _parse_gaps(self, text: str) -> List[Dict]:
        """Parse key-value block format into structured gap dicts."""
        gaps = []

        # Normalize
        text = text.replace('\r\n', '\n').replace('\r', '\n').strip()

        # Split on --- separator; also handle ``` code fences Gemini sometimes adds
        text = text.replace("```", "")

        blocks = [b.strip() for b in text.split("---") if b.strip()]

        for block in blocks:
            gap = {}
            # Some models wrap keys in ** bold markers — strip those
            block = block.replace("**", "")

            for line in block.split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue

                key, _, value = line.partition(":")
                key = key.strip().upper().replace(" ", "_")
                value = value.strip()

                if not value:
                    continue

                if key == "GAP_ID":
                    gap["id"] = value
                elif key == "GAP_TITLE":
                    gap["title"] = value
                elif key == "DESCRIPTION":
                    gap["description"] = value
                elif key == "EVIDENCE":
                    gap["evidence"] = value
                elif key == "NOVELTY_SCORE":
                    gap["novelty_score"] = self._safe_float(value, 7.0)
                elif key == "IMPACT_SCORE":
                    gap["impact_score"] = self._safe_float(value, 7.0)
                elif key == "FEASIBILITY":
                    # Take only the first word in case model adds explanation
                    gap["feasibility"] = value.split()[0].capitalize()
                elif key == "SUGGESTED_DIRECTION":
                    gap["suggested_direction"] = value
                elif key == "RELATED_AREAS":
                    gap["related_areas"] = [a.strip() for a in value.split(",") if a.strip()]

            # Accept gap if it has at minimum a title OR description
            if gap.get("title") or gap.get("description"):
                gap.setdefault("title", "Unnamed Gap")
                gap.setdefault("description", "")
                gap.setdefault("evidence", "")
                gap.setdefault("novelty_score", 7.0)
                gap.setdefault("impact_score", 7.0)
                gap.setdefault("feasibility", "Medium")
                gap.setdefault("suggested_direction", "")
                gap.setdefault("related_areas", [])
                gap["opportunity_score"] = round(
                    gap["novelty_score"] * 0.4 + gap["impact_score"] * 0.6, 2
                )
                gaps.append(gap)

        return gaps

    # ─── JSON Fallback ────────────────────────────────────────────────────────

    def _json_fallback(
        self, previous_response: str, query: str, papers: List[Dict]
    ) -> List[Dict]:
        """
        Two-stage fallback:
        Stage 1 — ask Gemini to reformat its own prose output as JSON.
        Stage 2 — if Stage 1 also fails, generate fresh gaps directly as JSON.
        """
        gaps = self._reformat_as_json(previous_response, query)
        if gaps:
            return gaps

        logger.warning("Stage 1 JSON fallback failed — generating fresh JSON gaps")
        return self._fresh_json_gaps(query, papers)

    def _reformat_as_json(self, text: str, query: str) -> List[Dict]:
        """Ask Gemini to reformat its own prose output as a JSON array."""
        prompt = f"""The following text contains a research gap analysis about "{query}".
Extract exactly 5 research gaps from it and return them as a JSON array.

TEXT:
{text[:4000]}

Return ONLY valid JSON. No markdown, no code fences, no explanation. Format:
[
  {{
    "title": "short gap title",
    "description": "what is missing",
    "evidence": "which papers or patterns show this",
    "novelty_score": 8,
    "impact_score": 8,
    "feasibility": "Medium",
    "suggested_direction": "how to address this gap"
  }},
  ...5 items total...
]"""

        try:
            response = self.llm.generate(prompt, system_prompt=None, temperature=0.1)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Reformat JSON failed: {e}")
            return []

    def _fresh_json_gaps(self, query: str, papers: List[Dict]) -> List[Dict]:
        """Generate gaps directly as JSON from scratch."""
        titles = [p.get("title", "") for p in papers[:15]]
        titles_text = "\n".join(f"- {t}" for t in titles)

        prompt = f"""Identify 5 research gaps for the topic: "{query}"

Related papers:
{titles_text}

Return ONLY a valid JSON array with exactly 5 items. No markdown, no code fences:
[
  {{
    "title": "gap title (5-10 words)",
    "description": "2-3 sentences on what is missing",
    "evidence": "brief evidence from the papers listed",
    "novelty_score": 8,
    "impact_score": 8,
    "feasibility": "High",
    "suggested_direction": "1-2 sentence research direction"
  }}
]"""

        try:
            response = self.llm.generate(prompt, system_prompt=None, temperature=0.2)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Fresh JSON gaps failed: {e}")
            return []

    def _parse_json_response(self, response: str) -> List[Dict]:
        """Clean and parse a JSON array from LLM response."""
        clean = response.strip()

        # Strip markdown code fences
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    clean = part
                    break

        # Find the JSON array boundaries
        start = clean.find("[")
        end = clean.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in response")

        clean = clean[start:end + 1]
        items = json.loads(clean)

        gaps = []
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                continue
            gap = {
                "id": str(i),
                "title": str(item.get("title", f"Gap {i}")),
                "description": str(item.get("description", "")),
                "evidence": str(item.get("evidence", "")),
                "novelty_score": self._safe_float(str(item.get("novelty_score", 7)), 7.0),
                "impact_score": self._safe_float(str(item.get("impact_score", 7)), 7.0),
                "feasibility": str(item.get("feasibility", "Medium")).split()[0].capitalize(),
                "suggested_direction": str(item.get("suggested_direction", "")),
                "related_areas": item.get("related_areas", []),
            }
            gap["opportunity_score"] = round(
                gap["novelty_score"] * 0.4 + gap["impact_score"] * 0.6, 2
            )
            gaps.append(gap)

        return gaps

    # ─── Scoring & Matrix ─────────────────────────────────────────────────────

    def _compute_overall_gap_score(self, papers: List[Dict], gaps: List[Dict]) -> float:
        """Compute overall research gap score for the query domain."""
        if not gaps:
            return 0.0
        avg_novelty = np.mean([g.get("novelty_score", 5) for g in gaps])
        sparsity_bonus = max(0, 2 - len(papers) / 20)
        return round(min(10.0, float(avg_novelty) + sparsity_bonus), 2)

    def _build_opportunity_matrix(self, gaps: List[Dict]) -> List[Dict]:
        """Build 2D opportunity matrix (novelty vs impact)."""
        matrix = []
        for gap in gaps:
            matrix.append({
                "title": gap.get("title", "Gap"),
                "novelty": gap.get("novelty_score", 5),
                "impact": gap.get("impact_score", 5),
                "feasibility": gap.get("feasibility", "Medium"),
                "quadrant": self._get_quadrant(
                    gap.get("novelty_score", 5), gap.get("impact_score", 5)
                ),
            })
        return matrix

    def _get_quadrant(self, novelty: float, impact: float) -> str:
        if novelty >= 7 and impact >= 7:
            return "🚀 Sweet Spot"
        elif novelty >= 7 and impact < 7:
            return "🔬 Exploratory"
        elif novelty < 7 and impact >= 7:
            return "⚡ Quick Win"
        else:
            return "📚 Incremental"

    # ─── Report Generation ────────────────────────────────────────────────────

    def _generate_gap_report(self, query: str, papers: List[Dict], analysis: Dict) -> str:
        """Generate detailed gap analysis report."""
        gap_summaries = []
        for g in analysis["gaps"][:5]:
            gap_summaries.append(
                f"- **{g.get('title', 'N/A')}**: {g.get('description', '')} "
                f"[Novelty: {g.get('novelty_score', 0)}/10, "
                f"Impact: {g.get('impact_score', 0)}/10]"
            )

        prompt = f"""Generate a comprehensive Research Gap Analysis Report for: "{query}"

OVERALL GAP SCORE: {analysis['gap_score']}/10
PAPERS ANALYZED: {len(papers)}
SEMANTIC CLUSTERS: {len(analysis['semantic_clusters'])}

IDENTIFIED GAPS:
{chr(10).join(gap_summaries) if gap_summaries else "No structured gaps — see cluster analysis."}

Write a detailed markdown report with these sections:

## Research Gap Analysis Report

### Executive Summary
[Overall state of research gaps in this domain, 3-4 sentences]

### Critical Gaps (Highest Priority)
[Top 3 gaps with the most significant opportunities, specific and actionable]

### Why These Gaps Exist
[Technical barriers, resource constraints, methodological limitations]

### Cross-Domain Opportunities
[Unexplored intersections with adjacent fields]

### Recommended Research Priorities
[Ranked list of directions with justification]

### Conclusion
[2-3 sentences on the opportunity landscape]

Be specific, evidence-based, and actionable."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.5)

    # ─── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(value: str, default: float = 7.0) -> float:
        """Extract a float from a string, stripping non-numeric chars."""
        try:
            cleaned = ''.join(c for c in str(value) if c.isdigit() or c == '.')
            return float(cleaned) if cleaned else default
        except Exception:
            return default