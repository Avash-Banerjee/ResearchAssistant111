"""
ResearchIQ - Novelty Scoring Agent (v2 - RAG Grounded)
========================================================
Semantic similarity via ChromaDB PLUS chunk-level RAG so the LLM sees
exactly what the similar papers say before judging novelty — not just titles.

RAG usage:
  _analyze_novelty_dimensions() → retrieves actual text from most similar papers
  _generate_novelty_report()    → retrieves chunks for "differentiation" framing
"""

import logging
import json
import numpy as np
from typing import List, Dict, Optional, Callable
from core.llm_client import get_llm
from core.vector_store import get_vector_store
from config.settings import AGENT_CONFIGS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Research Novelty and Integrity Expert.
Evaluate the originality of research proposals against existing literature.
You have been given actual excerpts from related papers — use these to make
calibrated, evidence-based assessments. Be honest and specific."""


class NoveltyScoringAgent:
    """
    Agent 6: Novelty & Plagiarism Scoring (RAG-grounded)
    - Semantic similarity via whole-paper embeddings
    - RAG chunk retrieval so LLM reads actual paper text, not just titles
    - Multi-dimensional novelty breakdown
    - JSON-parsed output for reliability
    """

    def __init__(self):
        self.llm          = get_llm()
        self.vector_store = get_vector_store()
        self.config       = AGENT_CONFIGS["novelty_scorer"]

    def score_novelty(
        self,
        title:       str,
        abstract:    str,
        methodology: str = "",
        progress_callback: Optional[Callable] = None,
    ) -> Dict:

        result = {
            "title":              title,
            "novelty_score":      0.0,
            "similarity_score":   0.0,
            "similar_papers":     [],
            "uniqueness_aspects": [],
            "overlap_concerns":   [],
            "novelty_breakdown":  {},
            "recommendations":    [],
            "novelty_report":     "",
            "verdict":            "",
            "agent":              self.config["name"],
        }

        full_text = f"{title} {abstract} {methodology}".strip()

        # ── Step 1: Whole-paper similarity search ─────────────────────────
        if progress_callback:
            progress_callback("🔍 Searching for similar papers...")

        similar_papers = self.vector_store.semantic_search(full_text, n_results=15)
        result["similar_papers"] = similar_papers

        # ── Step 2: Similarity score ──────────────────────────────────────
        if progress_callback:
            progress_callback("📊 Computing similarity scores...")

        similarities = [p.get("similarity", 0) for p in similar_papers]
        if similarities:
            result["similarity_score"] = round(float(np.mean(similarities[:5])), 4)
        base_novelty = max(0.0, 100.0 - result["similarity_score"] * 100)

        # ── Step 3: RAG chunk retrieval ───────────────────────────────────
        if progress_callback:
            progress_callback("📚 Retrieving relevant paper excerpts via RAG...")

        # Query from the perspective of the proposed work
        rag_context = self.vector_store.rag_retrieve(
            query      = f"{title} {abstract[:300]}",
            n_chunks   = 10,
            max_per_paper = 2,
        )

        # ── Step 4: LLM novelty analysis grounded in RAG chunks ───────────
        if progress_callback:
            progress_callback("🤖 Analysing novelty dimensions with RAG context...")

        llm_analysis = self._analyze_novelty_dimensions(
            title, abstract, methodology,
            similar_papers[:6], rag_context,
        )
        result["novelty_breakdown"]  = llm_analysis.get("breakdown", {})
        result["uniqueness_aspects"] = llm_analysis.get("unique_aspects", [])
        result["overlap_concerns"]   = llm_analysis.get("overlap_concerns", [])
        result["recommendations"]    = llm_analysis.get("recommendations", [])

        llm_score = llm_analysis.get("llm_novelty_score", base_novelty)
        result["novelty_score"] = round(0.4 * base_novelty + 0.6 * llm_score, 2)
        result["verdict"]       = self._get_verdict(result["novelty_score"])

        # ── Step 5: RAG-grounded report ───────────────────────────────────
        if progress_callback:
            progress_callback("📝 Generating novelty report...")
        result["novelty_report"] = self._generate_novelty_report(result, rag_context)

        return result

    # ─── RAG-Grounded Novelty Analysis ───────────────────────────────────────

    def _analyze_novelty_dimensions(
        self,
        title:         str,
        abstract:      str,
        methodology:   str,
        similar_papers: List[Dict],
        rag_context:   str,
    ) -> Dict:
        """
        LLM novelty scoring grounded in actual retrieved paper text.
        Uses JSON output for reliable parsing.
        """
        similar_summary = "\n".join([
            f'[{i+1}] "{p.get("title","")} ({p.get("year","")}) '
            f'— similarity: {p.get("similarity",0):.1%}"'
            for i, p in enumerate(similar_papers)
        ])

        prompt = f"""Assess the novelty of the proposed research using the retrieved literature excerpts.

═══════════════════════════════════════════════
PROPOSED RESEARCH:
Title:       {title}
Abstract:    {abstract[:600]}
Methodology: {methodology[:400] if methodology else "Not specified"}

SEMANTICALLY SIMILAR PAPERS (by embedding distance):
{similar_summary or "None found."}

RETRIEVED LITERATURE EXCERPTS (actual paper text — use this as evidence):
{rag_context or "No chunks available — base assessment on similar paper titles only."}
═══════════════════════════════════════════════

Based on the evidence above, return ONLY valid JSON with no markdown or code fences:
{{
  "llm_novelty_score": 72,
  "breakdown": {{
    "Problem Novelty":      {{"score": 8, "explanation": "..."}},
    "Method Novelty":       {{"score": 7, "explanation": "..."}},
    "Application Novelty":  {{"score": 6, "explanation": "..."}},
    "Combination Novelty":  {{"score": 7, "explanation": "..."}}
  }},
  "unique_aspects": [
    "Specific unique aspect 1 grounded in literature comparison",
    "Specific unique aspect 2",
    "Specific unique aspect 3"
  ],
  "overlap_concerns": [
    "Specific overlap concern 1 citing a retrieved paper",
    "Specific overlap concern 2"
  ],
  "recommendations": [
    "Specific actionable recommendation 1",
    "Specific actionable recommendation 2",
    "Specific actionable recommendation 3"
  ]
}}

Score calibration: 85+ = highly novel, 70–84 = good, 55–69 = moderate, 40–54 = low, <40 = very low.
Base your score on the actual excerpts — if the retrieved text shows the idea already exists, score lower."""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        return self._parse_json_analysis(response)

    @staticmethod
    def _parse_json_analysis(response: str) -> Dict:
        """Parse JSON novelty analysis with fallback defaults."""
        defaults = {
            "llm_novelty_score": 70.0,
            "breakdown":         {},
            "unique_aspects":    [],
            "overlap_concerns":  [],
            "recommendations":   [],
        }
        try:
            clean = response.strip().replace("```json", "").replace("```", "")
            start = clean.find("{")
            end   = clean.rfind("}")
            if start == -1 or end == -1:
                return defaults
            data = json.loads(clean[start:end + 1])

            # Normalise types
            data["llm_novelty_score"] = float(data.get("llm_novelty_score", 70))

            # Ensure breakdown values are dicts with score + explanation
            bd = data.get("breakdown", {})
            for k, v in bd.items():
                if isinstance(v, (int, float)):
                    bd[k] = {"score": float(v), "explanation": ""}
                elif isinstance(v, dict):
                    bd[k] = {
                        "score":       float(v.get("score", 7)),
                        "explanation": str(v.get("explanation", "")),
                    }
            data["breakdown"] = bd

            for key in ("unique_aspects", "overlap_concerns", "recommendations"):
                data[key] = [str(x) for x in data.get(key, []) if x]

            return {**defaults, **data}
        except Exception as e:
            logger.warning(f"Novelty JSON parse failed: {e} — using defaults")
            return defaults

    # ─── RAG-Grounded Report ──────────────────────────────────────────────────

    def _generate_novelty_report(self, result: Dict, rag_context: str) -> str:
        """Full novelty report grounded in RAG-retrieved literature."""

        breakdown_text = "\n".join([
            f"- {dim}: {d.get('score',0)}/10 — {d.get('explanation','')}"
            for dim, d in result["novelty_breakdown"].items()
        ]) or "No breakdown available."

        prompt = f"""Write a comprehensive Novelty Assessment Report.

═══════════════════════════════════════════════
RETRIEVED LITERATURE CONTEXT (use as evidence):
{rag_context or "No chunks available."}
═══════════════════════════════════════════════

RESEARCH BEING ASSESSED:
Title:          {result['title']}
Novelty Score:  {result['novelty_score']}/100
Verdict:        {result['verdict']}
Similarity:     {result['similarity_score']*100:.1f}% to prior work

DIMENSION SCORES:
{breakdown_text}

UNIQUE ASPECTS:    {chr(10).join('• '+u for u in result['uniqueness_aspects'])}
OVERLAP CONCERNS:  {chr(10).join('• '+c for c in result['overlap_concerns'])}
RECOMMENDATIONS:   {chr(10).join('→ '+r for r in result['recommendations'])}

Write a detailed markdown report with these sections:

## Novelty Assessment Report

### Overall Verdict
Explain the score and what it means for publication potential. Cite retrieved papers.

### Novelty Strengths
What makes this work genuinely novel relative to the retrieved literature?

### Areas of Overlap
Where does this work overlap with existing papers? Quote specific retrieved excerpts.

### Differentiation Strategy
How should the researcher position this work to maximise perceived novelty?

### Recommendations
Specific, actionable steps to strengthen novelty, grounded in the literature gaps.

### Publication Venue Suggestions
Which venues suit this novelty level? Give specific conference/journal names.

Ground every claim in the retrieved context. Be honest and constructive."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.4)

    # ─── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_verdict(score: float) -> str:
        if score >= 85:
            return "🏆 Highly Novel — Strong candidate for top-tier venues"
        elif score >= 70:
            return "✅ Good Novelty — Suitable for competitive venues with strong positioning"
        elif score >= 55:
            return "⚠️ Moderate Novelty — Needs clearer differentiation from prior work"
        elif score >= 40:
            return "🔴 Low Novelty — Significant overlap detected, major revision needed"
        else:
            return "❌ Very Low Novelty — Substantial rethinking required"

    def batch_score(self, proposals: List[Dict]) -> List[Dict]:
        scored = []
        for proposal in proposals:
            r = self.score_novelty(
                title=proposal.get("title", ""),
                abstract=proposal.get("abstract", ""),
                methodology=proposal.get("methodology", ""),
            )
            scored.append({
                "title":        proposal.get("title", ""),
                "novelty_score":r["novelty_score"],
                "verdict":      r["verdict"],
                "top_similar":  r["similar_papers"][0].get("title", "N/A")
                                if r["similar_papers"] else "None",
            })
        return sorted(scored, key=lambda x: x["novelty_score"], reverse=True)
