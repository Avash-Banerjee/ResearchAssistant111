"""
ResearchIQ - Novelty Scoring Agent
======================================
Computes semantic novelty scores and detects similarity with prior work.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Callable
from core.llm_client import get_llm
from core.vector_store import get_vector_store
from config.settings import AGENT_CONFIGS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Research Novelty and Integrity Expert.
You evaluate the originality of research proposals and papers against existing literature.
You provide honest, detailed novelty assessments with clear justifications.
Your scores are calibrated, consistent, and actionable."""


class NoveltyScoringAgent:
    """
    Agent 6: Novelty & Plagiarism Scoring
    - Semantic similarity against indexed papers
    - Novelty score computation (0-100)
    - Detailed similarity breakdown
    - Improvement suggestions
    """

    def __init__(self):
        self.llm = get_llm()
        self.vector_store = get_vector_store()
        self.config = AGENT_CONFIGS["novelty_scorer"]

    def score_novelty(
        self,
        title: str,
        abstract: str,
        methodology: str = "",
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """Score the novelty of a research idea/proposal."""

        result = {
            "title": title,
            "novelty_score": 0.0,
            "similarity_score": 0.0,
            "similar_papers": [],
            "uniqueness_aspects": [],
            "overlap_concerns": [],
            "novelty_breakdown": {},
            "recommendations": [],
            "novelty_report": "",
            "verdict": "",
            "agent": self.config["name"],
        }

        full_text = f"{title} {abstract} {methodology}"

        if progress_callback:
            progress_callback("🔍 Searching for similar papers...")

        # Find semantically similar papers
        similar_papers = self.vector_store.semantic_search(full_text, n_results=15)
        result["similar_papers"] = similar_papers

        if progress_callback:
            progress_callback("📊 Computing similarity scores...")

        # Compute similarity scores
        similarities = [p.get("similarity", 0) for p in similar_papers]
        if similarities:
            max_similarity = max(similarities)
            avg_similarity = np.mean(similarities[:5])  # Top 5
            result["similarity_score"] = round(float(avg_similarity), 4)
        else:
            max_similarity = 0
            result["similarity_score"] = 0.0

        # Compute novelty score (inverse of similarity + LLM adjustment)
        base_novelty = max(0, 100 - (result["similarity_score"] * 100))

        if progress_callback:
            progress_callback("🤖 Analyzing novelty dimensions...")

        llm_analysis = self._analyze_novelty_dimensions(
            title, abstract, methodology, similar_papers[:8]
        )
        result["novelty_breakdown"] = llm_analysis.get("breakdown", {})
        result["uniqueness_aspects"] = llm_analysis.get("unique_aspects", [])
        result["overlap_concerns"] = llm_analysis.get("overlap_concerns", [])
        result["recommendations"] = llm_analysis.get("recommendations", [])

        # Weighted final score
        llm_score = llm_analysis.get("llm_novelty_score", base_novelty)
        result["novelty_score"] = round(
            0.4 * base_novelty + 0.6 * llm_score, 2
        )

        result["verdict"] = self._get_verdict(result["novelty_score"])

        if progress_callback:
            progress_callback("📝 Generating novelty report...")
        result["novelty_report"] = self._generate_novelty_report(result)

        return result

    def _analyze_novelty_dimensions(
        self,
        title: str,
        abstract: str,
        methodology: str,
        similar_papers: List[Dict],
    ) -> Dict:
        """Use LLM to analyze novelty across multiple dimensions."""
        similar_texts = "\n".join([
            f"[{i+1}] {p.get('title', '')} ({p.get('year', '')}) - Similarity: {p.get('similarity', 0):.2%}\n"
            f"    {p.get('abstract', '')[:150]}..."
            for i, p in enumerate(similar_papers[:6])
        ])

        prompt = f"""Analyze the novelty of this research:

PROPOSED RESEARCH:
Title: {title}
Abstract: {abstract}
Methodology: {methodology[:400] if methodology else "Not specified"}

MOST SIMILAR EXISTING PAPERS:
{similar_texts if similar_texts else "No similar papers found in database."}

Provide novelty analysis in this EXACT format:

LLM_NOVELTY_SCORE: [0-100 integer]

BREAKDOWN:
- Problem Novelty: [1-10] - [explanation]
- Method Novelty: [1-10] - [explanation]
- Application Novelty: [1-10] - [explanation]
- Combination Novelty: [1-10] - [explanation]

UNIQUE_ASPECTS:
- [Aspect 1: What makes this genuinely novel]
- [Aspect 2]
- [Aspect 3]

OVERLAP_CONCERNS:
- [Concern 1: Area of overlap with existing work]
- [Concern 2]

RECOMMENDATIONS:
- [Recommendation 1: How to strengthen novelty]
- [Recommendation 2]
- [Recommendation 3]

Be honest and calibrated. A score >80 means highly novel. 60-80 = good novelty. 40-60 = moderate. <40 = significant overlap."""

        response = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.4)
        return self._parse_novelty_analysis(response)

    def _parse_novelty_analysis(self, text: str) -> Dict:
        """Parse LLM novelty analysis."""
        result = {
            "llm_novelty_score": 70.0,
            "breakdown": {},
            "unique_aspects": [],
            "overlap_concerns": [],
            "recommendations": [],
        }

        lines = text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("LLM_NOVELTY_SCORE:"):
                try:
                    score_text = line.replace("LLM_NOVELTY_SCORE:", "").strip()
                    result["llm_novelty_score"] = float(score_text.split()[0])
                except:
                    pass

            elif line.startswith("BREAKDOWN:"):
                current_section = "breakdown"
            elif line.startswith("UNIQUE_ASPECTS:"):
                current_section = "unique"
            elif line.startswith("OVERLAP_CONCERNS:"):
                current_section = "overlap"
            elif line.startswith("RECOMMENDATIONS:"):
                current_section = "recs"

            elif line.startswith("- ") and current_section:
                content = line[2:].strip()
                if current_section == "breakdown" and ":" in content:
                    parts = content.split(":", 1)
                    dim_name = parts[0].strip()
                    rest = parts[1].strip() if len(parts) > 1 else ""
                    # Try to extract score
                    score_parts = rest.split("-", 1)
                    try:
                        score = float(score_parts[0].strip().split()[0])
                    except:
                        score = 7.0
                    explanation = score_parts[1].strip() if len(score_parts) > 1 else ""
                    result["breakdown"][dim_name] = {"score": score, "explanation": explanation}
                elif current_section == "unique" and content:
                    result["unique_aspects"].append(content)
                elif current_section == "overlap" and content:
                    result["overlap_concerns"].append(content)
                elif current_section == "recs" and content:
                    result["recommendations"].append(content)

        return result

    def _get_verdict(self, score: float) -> str:
        """Get human-readable verdict for novelty score."""
        if score >= 85:
            return "🏆 Highly Novel - Strong candidate for top-tier venues"
        elif score >= 70:
            return "✅ Good Novelty - Suitable for competitive venues with strong positioning"
        elif score >= 55:
            return "⚠️ Moderate Novelty - Needs clearer differentiation from prior work"
        elif score >= 40:
            return "🔴 Low Novelty - Significant overlap detected, major revision needed"
        else:
            return "❌ Very Low Novelty - Substantial rethinking required"

    def _generate_novelty_report(self, result: Dict) -> str:
        """Generate comprehensive novelty report."""
        prompt = f"""Write a comprehensive novelty assessment report for:

TITLE: {result['title']}
NOVELTY SCORE: {result['novelty_score']}/100
VERDICT: {result['verdict']}
SIMILARITY TO EXISTING WORK: {result['similarity_score']*100:.1f}%

SIMILAR PAPERS FOUND: {len(result['similar_papers'])}
Top similar paper: {result['similar_papers'][0].get('title', 'N/A') if result['similar_papers'] else 'None'}

UNIQUE ASPECTS: {', '.join(result['uniqueness_aspects'][:3])}
OVERLAP CONCERNS: {', '.join(result['overlap_concerns'][:2])}

Write a detailed novelty assessment report:

## Novelty Assessment Report

### Overall Verdict
[Detailed explanation of the score and verdict]

### Novelty Strengths
[What makes this work novel and valuable?]

### Areas of Overlap
[Where does this work overlap with prior art? Is this acceptable?]

### Differentiation Strategy
[How should the researcher position this work?]

### Recommendations for Strengthening Novelty
[Specific, actionable recommendations]

### Publication Venue Recommendations
[Which venues would be appropriate given this novelty level?]

Be constructive, honest, and specific."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.5)

    def batch_score(self, proposals: List[Dict]) -> List[Dict]:
        """Score multiple proposals and return ranked results."""
        scored = []
        for proposal in proposals:
            score_result = self.score_novelty(
                title=proposal.get("title", ""),
                abstract=proposal.get("abstract", ""),
                methodology=proposal.get("methodology", ""),
            )
            scored.append({
                "title": proposal.get("title", ""),
                "novelty_score": score_result["novelty_score"],
                "verdict": score_result["verdict"],
                "top_similar": score_result["similar_papers"][0].get("title", "N/A")
                if score_result["similar_papers"] else "None",
            })
        return sorted(scored, key=lambda x: x["novelty_score"], reverse=True)
