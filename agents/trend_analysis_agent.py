"""
ResearchIQ - Trend Analysis Agent
=====================================
Uses topic modeling and semantic analysis to detect research trends.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Callable
from collections import Counter
from core.llm_client import get_llm
from core.vector_store import get_vector_store
from config.settings import AGENT_CONFIGS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Research Trend Analyst specializing in academic literature.
You identify emerging topics, quantify research momentum, and forecast future directions.
Use data-driven insights and provide actionable analysis for research planning."""


class TrendAnalysisAgent:
    """
    Agent 2: Trend Analysis
    - Identifies emerging research topics using BERTopic/LDA
    - Tracks research momentum (citation velocity, paper volume)
    - Detects hot vs. declining topics
    - Forecasts future research directions
    """

    def __init__(self):
        self.llm = get_llm()
        self.vector_store = get_vector_store()
        self.config = AGENT_CONFIGS["trend_analyzer"]

    def analyze_trends(
        self,
        papers: List[Dict],
        query: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """Analyze research trends from a paper collection."""

        result = {
            "query": query,
            "topics": [],
            "trending_keywords": [],
            "momentum_scores": {},
            "temporal_trends": {},
            "emerging_areas": [],
            "declining_areas": [],
            "trend_report": "",
            "agent": self.config["name"],
        }

        if not papers:
            result["trend_report"] = "No papers available for trend analysis."
            return result

        if progress_callback:
            progress_callback("📊 Extracting keywords and topics...")

        # Extract keywords from abstracts
        result["trending_keywords"] = self._extract_trending_keywords(papers)

        # Compute temporal trends
        result["temporal_trends"] = self._compute_temporal_trends(papers)

        # Compute momentum scores
        result["momentum_scores"] = self._compute_momentum(papers)

        # Topic modeling via LLM (fallback when BERTopic not available)
        if progress_callback:
            progress_callback("🤖 Running topic modeling analysis...")
        result["topics"] = self._extract_topics_via_llm(papers, query)

        # Identify emerging vs declining
        result["emerging_areas"], result["declining_areas"] = self._classify_trends(
            result["temporal_trends"]
        )

        # Generate comprehensive trend report
        if progress_callback:
            progress_callback("📈 Generating trend analysis report...")
        result["trend_report"] = self._generate_trend_report(query, papers, result)

        return result

    def _extract_trending_keywords(self, papers: List[Dict]) -> List[Dict]:
        """Extract and rank keywords by frequency and recency."""
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "been", "be",
            "this", "that", "we", "our", "their", "its", "it", "which", "who",
            "have", "has", "had", "not", "also", "can", "using", "based", "two",
            "three", "four", "five", "paper", "show", "proposed", "method", "approach",
            "result", "results", "dataset", "data", "training", "model", "models",
            "performance", "evaluation", "demonstrate", "presents", "propose",
        }

        recent_year = 2022
        keyword_freq = Counter()
        recent_keyword_freq = Counter()

        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            year = paper.get("year") or 0
            words = text.lower().split()
            # Extract 2-grams and single meaningful words
            for i, word in enumerate(words):
                word = word.strip(".,;:()[]\"'")
                if len(word) > 4 and word not in stop_words and word.isalpha():
                    keyword_freq[word] += 1
                    if year >= recent_year:
                        recent_keyword_freq[word] += 1
                # Bigrams
                if i < len(words) - 1:
                    w2 = words[i + 1].strip(".,;:()[]\"'")
                    if (len(word) > 3 and len(w2) > 3
                            and word not in stop_words and w2 not in stop_words
                            and word.isalpha() and w2.isalpha()):
                        bigram = f"{word} {w2}"
                        keyword_freq[bigram] += 1
                        if year >= recent_year:
                            recent_keyword_freq[bigram] += 1

        # Compute trend score = recent_freq / total_freq
        keywords = []
        for kw, total in keyword_freq.most_common(80):
            recent = recent_keyword_freq.get(kw, 0)
            trend_score = round(recent / max(total, 1), 3)
            keywords.append({
                "keyword": kw,
                "frequency": total,
                "recent_frequency": recent,
                "trend_score": trend_score,
                "is_trending": trend_score > 0.6 and total >= 3,
            })

        return sorted(keywords, key=lambda x: (x["is_trending"], x["frequency"]), reverse=True)[:40]

    def _compute_temporal_trends(self, papers: List[Dict]) -> Dict:
        """Track paper volume and citation velocity by year."""
        yearly = {}
        for paper in papers:
            year = paper.get("year")
            if year and 2010 <= int(year) <= 2026:
                y = str(year)
                if y not in yearly:
                    yearly[y] = {"count": 0, "total_citations": 0, "papers": []}
                yearly[y]["count"] += 1
                yearly[y]["total_citations"] += paper.get("citations", 0)
                yearly[y]["papers"].append(paper.get("title", "")[:60])

        # Compute avg citations per year
        for y in yearly:
            count = yearly[y]["count"]
            yearly[y]["avg_citations"] = round(
                yearly[y]["total_citations"] / max(count, 1), 1
            )

        return dict(sorted(yearly.items()))

    def _compute_momentum(self, papers: List[Dict]) -> Dict:
        """Compute research momentum indicators."""
        if not papers:
            return {}

        years = [p.get("year") for p in papers if p.get("year")]
        citations = [p.get("citations", 0) for p in papers]

        recent_papers = [p for p in papers if (p.get("year") or 0) >= 2022]
        older_papers = [p for p in papers if (p.get("year") or 0) < 2022]

        return {
            "total_papers_analyzed": len(papers),
            "recent_papers_count": len(recent_papers),
            "older_papers_count": len(older_papers),
            "recent_ratio": round(len(recent_papers) / max(len(papers), 1), 3),
            "avg_citations": round(np.mean(citations) if citations else 0, 1),
            "max_citations": max(citations) if citations else 0,
            "median_citations": round(float(np.median(citations)) if citations else 0, 1),
            "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
            "momentum": "High" if len(recent_papers) / max(len(papers), 1) > 0.4 else "Medium" if len(recent_papers) / max(len(papers), 1) > 0.2 else "Low",
        }

    def _classify_trends(self, temporal: Dict) -> tuple:
        """Classify topics as emerging or declining based on recent vs older volume."""
        sorted_years = sorted(temporal.keys())
        if len(sorted_years) < 3:
            return [], []

        recent_years = sorted_years[-3:]
        older_years = sorted_years[:-3]

        recent_avg = np.mean([temporal[y]["count"] for y in recent_years]) if recent_years else 0
        older_avg = np.mean([temporal[y]["count"] for y in older_years]) if older_years else 0

        growth_rate = (recent_avg - older_avg) / max(older_avg, 1)

        emerging = []
        declining = []

        if growth_rate > 0.3:
            emerging.append({
                "area": "Core research area",
                "growth_rate": f"+{round(growth_rate * 100, 1)}%",
                "recent_volume": int(recent_avg),
            })
        elif growth_rate < -0.2:
            declining.append({
                "area": "Core research area",
                "decline_rate": f"{round(growth_rate * 100, 1)}%",
                "recent_volume": int(recent_avg),
            })

        return emerging, declining

    def _extract_topics_via_llm(self, papers: List[Dict], query: str) -> List[Dict]:
        """Use Gemini to extract research topics."""
        paper_titles = [p.get("title", "") for p in papers[:25]]
        titles_text = "\n".join([f"- {t}" for t in paper_titles])

        prompt = f"""Given these research paper titles in the domain of "{query}":

{titles_text}

Extract the 5-7 main research topics/themes. For each topic, provide:
1. A clear topic name (2-5 words)
2. A brief description (1-2 sentences)
3. Representative keywords (5 keywords)
4. Estimated prevalence (what % of papers cover this)

Respond in this EXACT format for each topic:
TOPIC: [name]
DESCRIPTION: [description]
KEYWORDS: [kw1, kw2, kw3, kw4, kw5]
PREVALENCE: [percentage]
---"""

        response = self.llm.generate(prompt, temperature=0.4)
        return self._parse_topics(response)

    def _parse_topics(self, text: str) -> List[Dict]:
        """Parse LLM topic output into structured data."""
        topics = []
        blocks = text.split("---")
        for block in blocks:
            topic = {}
            for line in block.strip().split("\n"):
                if line.startswith("TOPIC:"):
                    topic["name"] = line.replace("TOPIC:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    topic["description"] = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("KEYWORDS:"):
                    kws = line.replace("KEYWORDS:", "").strip()
                    topic["keywords"] = [k.strip() for k in kws.split(",")]
                elif line.startswith("PREVALENCE:"):
                    prev = line.replace("PREVALENCE:", "").strip().replace("%", "")
                    try:
                        topic["prevalence"] = float(prev)
                    except:
                        topic["prevalence"] = 15.0
            if topic.get("name"):
                topics.append(topic)
        return topics

    def _generate_trend_report(self, query: str, papers: List[Dict], analysis: Dict) -> str:
        """Generate comprehensive trend analysis report."""
        trending_kws = [k["keyword"] for k in analysis["trending_keywords"][:15] if k["is_trending"]]
        momentum = analysis["momentum_scores"]

        prompt = f"""Analyze research trends for the topic: "{query}"

MOMENTUM INDICATORS:
- Total papers analyzed: {momentum.get('total_papers_analyzed', 0)}
- Recent papers (2022+): {momentum.get('recent_papers_count', 0)} ({round(momentum.get('recent_ratio', 0)*100, 1)}% of total)
- Average citations: {momentum.get('avg_citations', 0)}
- Research momentum: {momentum.get('momentum', 'Unknown')}

TRENDING KEYWORDS: {', '.join(trending_kws) if trending_kws else 'N/A'}

IDENTIFIED TOPICS: {', '.join([t['name'] for t in analysis['topics']])}

TEMPORAL DATA: {str(analysis['temporal_trends'])}

Generate a comprehensive trend analysis report with:

## Executive Summary
[2-3 sentence overview of research momentum]

## Hot Topics & Emerging Trends
[What's gaining momentum right now?]

## Mature/Stable Areas
[Well-established research directions]

## Declining or Saturated Areas
[Areas where activity is slowing]

## Technology & Methodology Trends
[What techniques/tools are researchers using more?]

## Predicted Future Directions (2025-2027)
[Based on trends, where is this field heading?]

## Strategic Opportunities
[Where should new researchers focus for maximum impact?]

Be specific and actionable. Use data from the analysis above."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.6)
