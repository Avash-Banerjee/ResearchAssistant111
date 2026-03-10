"""
ResearchIQ - Trend Analysis Agent (v2)
========================================
Real topic modeling with BERTopic + LDA + dynamic embeddings.
RAG-grounded report generation — every claim is backed by retrieved paper chunks.

Pipeline:
  1. Keyword extraction         — unigrams + bigrams, trend scoring
  2. BERTopic                   — UMAP + HDBSCAN + c-TF-IDF topic extraction
     └─ fallback: LDA           — gensim LDA if BERTopic unavailable
     └─ fallback: LLM topics    — Gemini extraction if both fail
  3. Temporal analysis          — paper volume + citation velocity per year
  4. Citation momentum          — recent vs older paper ratio
  5. Keyword evolution          — track keyword prevalence decade-over-decade
  6. RAG report generation      — retrieve relevant chunks, inject into report prompt
"""

import logging
import re
import json
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from collections import Counter, defaultdict
from core.llm_client import get_llm
from core.vector_store import get_vector_store
from config.settings import AGENT_CONFIGS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Research Trend Analyst specialising in academic literature.
You identify emerging topics, quantify research momentum, and forecast future directions.
Your analysis is data-driven, evidence-backed, and actionable for research planning."""

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "been", "be", "this", "that",
    "we", "our", "their", "its", "it", "which", "who", "have", "has", "had",
    "not", "also", "can", "using", "based", "two", "three", "paper", "show",
    "proposed", "method", "approach", "result", "results", "dataset", "data",
    "training", "model", "models", "performance", "evaluation", "demonstrate",
    "presents", "propose", "experiments", "study", "work", "new", "different",
    "existing", "current", "recent", "large", "small", "high", "low", "used",
    "each", "both", "such", "than", "more", "most", "while", "when", "then",
    "these", "those", "however", "therefore", "thus", "hence", "since", "through",
    "across", "between", "within", "without", "further", "show", "shows",
}


class TrendAnalysisAgent:
    """
    Agent 2: Trend Analysis
    - BERTopic (UMAP + HDBSCAN + c-TF-IDF) for genuine topic discovery
    - LDA fallback (gensim) for probabilistic topic modeling
    - LLM fallback if both unavailable
    - Keyword evolution tracking across time windows
    - RAG-grounded narrative report (LLM sees real paper chunks as context)
    """

    def __init__(self):
        self.llm          = get_llm()
        self.vector_store = get_vector_store()
        self.config       = AGENT_CONFIGS["trend_analyzer"]

    # ─── Main Entry Point ─────────────────────────────────────────────────────

    def analyze_trends(
        self,
        papers: List[Dict],
        query: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:

        result = {
            "query":            query,
            "topics":           [],
            "topic_method":     "none",
            "trending_keywords": [],
            "keyword_evolution": {},
            "momentum_scores":  {},
            "temporal_trends":  {},
            "emerging_areas":   [],
            "declining_areas":  [],
            "trend_report":     "",
            "agent":            self.config["name"],
        }

        if not papers:
            result["trend_report"] = "No papers available for trend analysis."
            return result

        # ── Step 1: keywords ──────────────────────────────────────────────
        if progress_callback:
            progress_callback("📊 Extracting and ranking keywords...")
        result["trending_keywords"] = self._extract_trending_keywords(papers)
        result["keyword_evolution"] = self._track_keyword_evolution(papers)

        # ── Step 2: temporal + momentum ───────────────────────────────────
        if progress_callback:
            progress_callback("📅 Computing temporal trends and momentum...")
        result["temporal_trends"]  = self._compute_temporal_trends(papers)
        result["momentum_scores"]  = self._compute_momentum(papers)
        result["emerging_areas"], result["declining_areas"] = \
            self._classify_trends(result["temporal_trends"])

        # ── Step 3: topic modeling ────────────────────────────────────────
        if progress_callback:
            progress_callback("🧠 Running topic modeling (BERTopic → LDA → LLM)...")
        topics, method = self._run_topic_modeling(papers, query)
        result["topics"]       = topics
        result["topic_method"] = method
        logger.info(f"Topic modeling method used: {method}, topics found: {len(topics)}")

        # ── Step 4: RAG-grounded report ───────────────────────────────────
        if progress_callback:
            progress_callback("📝 Generating RAG-grounded trend report...")
        result["trend_report"] = self._generate_rag_report(query, papers, result)

        return result

    # ─── Keyword Extraction ───────────────────────────────────────────────────

    def _extract_trending_keywords(self, papers: List[Dict]) -> List[Dict]:
        recent_cutoff = 2022
        freq: Counter         = Counter()
        recent_freq: Counter  = Counter()

        for paper in papers:
            text = f"{paper.get('title','')} {paper.get('abstract','')}"
            year = paper.get("year") or 0
            words = [w.strip(".,;:()[]\"'") for w in text.lower().split()]

            for i, word in enumerate(words):
                if len(word) > 4 and word not in STOP_WORDS and word.isalpha():
                    freq[word] += 1
                    if year >= recent_cutoff:
                        recent_freq[word] += 1
                # Bigrams
                if i < len(words) - 1:
                    w2 = words[i + 1]
                    if (len(word) > 3 and len(w2) > 3
                            and word not in STOP_WORDS and w2 not in STOP_WORDS
                            and word.isalpha() and w2.isalpha()):
                        bigram = f"{word} {w2}"
                        freq[bigram] += 1
                        if year >= recent_cutoff:
                            recent_freq[bigram] += 1

        keywords = []
        for kw, total in freq.most_common(100):
            recent     = recent_freq.get(kw, 0)
            trend_score = round(recent / max(total, 1), 3)
            keywords.append({
                "keyword":          kw,
                "frequency":        total,
                "recent_frequency": recent,
                "trend_score":      trend_score,
                "is_trending":      trend_score > 0.55 and total >= 3,
            })

        return sorted(keywords, key=lambda x: (x["is_trending"], x["frequency"]), reverse=True)[:50]

    def _track_keyword_evolution(self, papers: List[Dict]) -> Dict:
        """
        Tracks top keywords across 5-year windows so we can see
        which terms rose or fell over time.
        """
        windows = {
            "pre_2015":  [p for p in papers if (p.get("year") or 0) < 2015],
            "2015_2019": [p for p in papers if 2015 <= (p.get("year") or 0) <= 2019],
            "2020_2022": [p for p in papers if 2020 <= (p.get("year") or 0) <= 2022],
            "2023_plus": [p for p in papers if (p.get("year") or 0) >= 2023],
        }
        evolution = {}
        for window, wpaper in windows.items():
            if not wpaper:
                continue
            c: Counter = Counter()
            for p in wpaper:
                text  = f"{p.get('title','')} {p.get('abstract','')}".lower()
                words = [w.strip(".,;:()[]\"'") for w in text.split()]
                for w in words:
                    if len(w) > 4 and w not in STOP_WORDS and w.isalpha():
                        c[w] += 1
            evolution[window] = {
                kw: cnt for kw, cnt in c.most_common(20)
            }
        return evolution

    # ─── Temporal Analysis ────────────────────────────────────────────────────

    def _compute_temporal_trends(self, papers: List[Dict]) -> Dict:
        yearly: Dict[str, Dict] = {}
        for paper in papers:
            year = paper.get("year")
            if not year:
                continue
            try:
                y = int(year)
            except (ValueError, TypeError):
                continue
            if not (2005 <= y <= 2026):
                continue
            ys = str(y)
            if ys not in yearly:
                yearly[ys] = {"count": 0, "total_citations": 0, "papers": []}
            yearly[ys]["count"]           += 1
            yearly[ys]["total_citations"] += paper.get("citations", 0)
            yearly[ys]["papers"].append(paper.get("title", "")[:60])

        for ys, d in yearly.items():
            d["avg_citations"] = round(d["total_citations"] / max(d["count"], 1), 1)

        return dict(sorted(yearly.items()))

    def _compute_momentum(self, papers: List[Dict]) -> Dict:
        if not papers:
            return {}
        years     = [p.get("year") for p in papers if p.get("year")]
        citations = [p.get("citations", 0) for p in papers]
        recent    = [p for p in papers if (p.get("year") or 0) >= 2022]
        ratio     = len(recent) / max(len(papers), 1)
        return {
            "total_papers_analyzed": len(papers),
            "recent_papers_count":   len(recent),
            "older_papers_count":    len(papers) - len(recent),
            "recent_ratio":          round(ratio, 3),
            "avg_citations":         round(float(np.mean(citations)) if citations else 0, 1),
            "max_citations":         max(citations) if citations else 0,
            "median_citations":      round(float(np.median(citations)) if citations else 0, 1),
            "year_range":            f"{min(years)}-{max(years)}" if years else "N/A",
            "momentum":              "High" if ratio > 0.4 else "Medium" if ratio > 0.2 else "Low",
        }

    def _classify_trends(self, temporal: Dict) -> Tuple[List[Dict], List[Dict]]:
        sorted_years = sorted(temporal.keys())
        if len(sorted_years) < 3:
            return [], []
        recent_years = sorted_years[-3:]
        older_years  = sorted_years[:-3]
        recent_avg   = float(np.mean([temporal[y]["count"] for y in recent_years]))
        older_avg    = float(np.mean([temporal[y]["count"] for y in older_years])) if older_years else 1.0
        growth       = (recent_avg - older_avg) / max(older_avg, 1)

        emerging, declining = [], []
        if growth > 0.25:
            emerging.append({
                "area":          "Core research area",
                "growth_rate":   f"+{round(growth * 100, 1)}%",
                "recent_volume": int(recent_avg),
            })
        elif growth < -0.2:
            declining.append({
                "area":          "Core research area",
                "decline_rate":  f"{round(growth * 100, 1)}%",
                "recent_volume": int(recent_avg),
            })
        return emerging, declining

    # ─── Topic Modeling ───────────────────────────────────────────────────────

    def _run_topic_modeling(
        self, papers: List[Dict], query: str
    ) -> Tuple[List[Dict], str]:
        """
        Try BERTopic first, then LDA, then LLM.
        Returns (topics_list, method_name).
        """
        docs = [
            f"{p.get('title','')} {p.get('abstract','')}"
            for p in papers
        ]
        docs = [d.strip() for d in docs if len(d.strip()) > 30]

        if len(docs) < 5:
            return self._topics_via_llm(papers, query), "llm"

        # ── Try BERTopic ──────────────────────────────────────────────────
        topics, method = self._bertopic_modeling(docs, papers)
        if topics:
            return topics, method

        # ── Try LDA ──────────────────────────────────────────────────────
        topics, method = self._lda_modeling(docs, papers)
        if topics:
            return topics, method

        # ── LLM fallback ─────────────────────────────────────────────────
        return self._topics_via_llm(papers, query), "llm"

    # ── BERTopic ──────────────────────────────────────────────────────────────

    def _bertopic_modeling(
        self, docs: List[str], papers: List[Dict]
    ) -> Tuple[List[Dict], str]:
        try:
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer

            n = len(docs)

            umap_model = UMAP(
                n_neighbors  = min(10, n - 1),
                n_components = min(5, n - 2),
                min_dist     = 0.0,
                metric       = "cosine",
                random_state = 42,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size = max(2, n // 10),
                min_samples      = 1,
                metric           = "euclidean",
                cluster_selection_method = "eom",
                prediction_data  = True,
            )
            vectorizer = CountVectorizer(
                stop_words  = "english",
                ngram_range = (1, 2),
                min_df      = 1,
            )

            topic_model = BERTopic(
                umap_model      = umap_model,
                hdbscan_model   = hdbscan_model,
                vectorizer_model= vectorizer,
                top_n_words     = 10,
                verbose         = False,
            )

            topics_per_doc, _  = topic_model.fit_transform(docs)
            topic_info         = topic_model.get_topic_info()
            # Drop outlier topic (-1)
            topic_info = topic_info[topic_info["Topic"] != -1]

            if topic_info.empty:
                logger.warning("BERTopic found 0 topics (all outliers)")
                return [], "bertopic_failed"

            result = []
            for _, row in topic_info.iterrows():
                tid      = row["Topic"]
                raw_words= topic_model.get_topic(tid)            # [(word, score), ...]
                keywords = [w for w, _ in raw_words[:8]] if raw_words else []
                count    = int(row.get("Count", 0))
                prevalence = round(count / max(len(docs), 1) * 100, 1)

                # Build a representative name from top 3 keywords
                name = " / ".join(keywords[:3]) if keywords else f"Topic {tid}"

                # Find papers in this cluster
                cluster_papers = [
                    papers[i] for i, t in enumerate(topics_per_doc)
                    if t == tid and i < len(papers)
                ]
                years = [p.get("year") for p in cluster_papers if p.get("year")]
                avg_cit = float(np.mean([p.get("citations", 0) for p in cluster_papers])) \
                          if cluster_papers else 0.0

                result.append({
                    "id":          int(tid),
                    "name":        name,
                    "keywords":    keywords,
                    "description": f"Topic covering {', '.join(keywords[:5])}.",
                    "prevalence":  prevalence,
                    "paper_count": count,
                    "year_range":  f"{min(years)}-{max(years)}" if years else "N/A",
                    "avg_citations": round(avg_cit, 1),
                    "is_emerging": bool(
                        years and max(years) >= 2022 and
                        sum(1 for y in years if y >= 2022) / len(years) > 0.5
                    ),
                })

            # Enrich topic names with LLM (they're just keyword lists right now)
            result = self._enrich_topic_names(result)

            logger.info(f"BERTopic found {len(result)} topics")
            return result, "bertopic"

        except ImportError as e:
            logger.warning(f"BERTopic not installed ({e}), trying LDA")
            return [], "bertopic_missing"
        except Exception as e:
            logger.warning(f"BERTopic failed: {e}, trying LDA")
            return [], "bertopic_error"

    def _enrich_topic_names(self, topics: List[Dict]) -> List[Dict]:
        """Ask LLM to give each BERTopic cluster a human-readable name and description."""
        if not topics:
            return topics

        topics_text = "\n".join([
            f"Topic {t['id']}: keywords = {', '.join(t['keywords'][:6])}"
            for t in topics
        ])

        prompt = f"""Each line below is a research topic represented by its top keywords.
Give each topic a concise human-readable name (3-5 words) and a 1-sentence description.

{topics_text}

Return ONLY valid JSON, no markdown:
[
  {{"id": 0, "name": "Topic name here", "description": "One sentence description."}},
  ...
]"""

        try:
            response = self.llm.generate(prompt, system_prompt=None, temperature=0.2)
            clean = response.strip()
            if "```" in clean:
                parts = clean.split("```")
                for p in parts:
                    p = p.strip().lstrip("json").strip()
                    if p.startswith("["):
                        clean = p
                        break
            start, end = clean.find("["), clean.rfind("]")
            if start != -1 and end != -1:
                enriched = json.loads(clean[start:end + 1])
                enriched_map = {int(e["id"]): e for e in enriched if isinstance(e, dict)}
                for t in topics:
                    if t["id"] in enriched_map:
                        e = enriched_map[t["id"]]
                        t["name"]        = e.get("name", t["name"])
                        t["description"] = e.get("description", t["description"])
        except Exception as ex:
            logger.warning(f"Topic name enrichment failed: {ex}")

        return topics

    # ── LDA ───────────────────────────────────────────────────────────────────

    def _lda_modeling(
        self, docs: List[str], papers: List[Dict]
    ) -> Tuple[List[Dict], str]:
        try:
            import gensim
            from gensim import corpora
            from gensim.models.ldamodel import LdaModel
            from gensim.parsing.preprocessing import (
                preprocess_string, strip_punctuation,
                strip_numeric, remove_stopwords, strip_short,
            )

            FILTERS = [strip_punctuation, strip_numeric, remove_stopwords, strip_short]
            tokenized = [preprocess_string(doc, FILTERS) for doc in docs]
            tokenized = [t for t in tokenized if t]

            if len(tokenized) < 5:
                return [], "lda_insufficient_data"

            dictionary = corpora.Dictionary(tokenized)
            dictionary.filter_extremes(no_below=1, no_above=0.95)
            corpus     = [dictionary.doc2bow(t) for t in tokenized]

            n_topics   = min(7, max(3, len(docs) // 5))
            lda        = LdaModel(
                corpus          = corpus,
                id2word         = dictionary,
                num_topics      = n_topics,
                random_state    = 42,
                passes          = 15,
                alpha           = "auto",
                per_word_topics = True,
            )

            # Get document-topic distribution
            doc_topics = [
                max(lda.get_document_topics(bow, minimum_probability=0.0),
                    key=lambda x: x[1])[0]
                for bow in corpus
            ]

            result = []
            for tid in range(n_topics):
                top_words  = lda.show_topic(tid, topn=10)
                keywords   = [w for w, _ in top_words]
                name       = " / ".join(keywords[:3])

                cluster_papers = [
                    papers[i] for i, dt in enumerate(doc_topics)
                    if dt == tid and i < len(papers)
                ]
                count      = len(cluster_papers)
                years      = [p.get("year") for p in cluster_papers if p.get("year")]
                avg_cit    = float(np.mean([p.get("citations", 0) for p in cluster_papers])) \
                             if cluster_papers else 0.0
                prevalence = round(count / max(len(docs), 1) * 100, 1)

                result.append({
                    "id":           tid,
                    "name":         name,
                    "keywords":     keywords,
                    "description":  f"LDA topic covering {', '.join(keywords[:5])}.",
                    "prevalence":   prevalence,
                    "paper_count":  count,
                    "year_range":   f"{min(years)}-{max(years)}" if years else "N/A",
                    "avg_citations": round(avg_cit, 1),
                    "is_emerging":  bool(
                        years and max(years) >= 2022 and
                        sum(1 for y in years if y >= 2022) / len(years) > 0.5
                    ),
                    "lda_coherence_words": keywords,
                })

            # Enrich with LLM names
            result = self._enrich_topic_names(result)

            logger.info(f"LDA found {len(result)} topics")
            return result, "lda"

        except ImportError as e:
            logger.warning(f"Gensim not installed ({e}), using LLM topics")
            return [], "lda_missing"
        except Exception as e:
            logger.warning(f"LDA failed: {e}, using LLM topics")
            return [], "lda_error"

    # ── LLM Topic Fallback ────────────────────────────────────────────────────

    def _topics_via_llm(self, papers: List[Dict], query: str) -> List[Dict]:
        titles = "\n".join([f"- {p.get('title','')}" for p in papers[:30]])

        prompt = f"""Given these research paper titles in the domain of "{query}":

{titles}

Identify 5-7 main research topics. Return ONLY valid JSON, no markdown:
[
  {{
    "id": 0,
    "name": "Topic name (3-5 words)",
    "keywords": ["kw1","kw2","kw3","kw4","kw5"],
    "description": "1-2 sentence description",
    "prevalence": 20.0
  }}
]"""

        try:
            response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
            clean    = response.strip().replace("```json","").replace("```","")
            start, end = clean.find("["), clean.rfind("]")
            if start != -1 and end != -1:
                items = json.loads(clean[start:end + 1])
                topics = []
                for i, item in enumerate(items):
                    if isinstance(item, dict):
                        topics.append({
                            "id":           i,
                            "name":         item.get("name", f"Topic {i}"),
                            "keywords":     item.get("keywords", []),
                            "description":  item.get("description", ""),
                            "prevalence":   float(item.get("prevalence", 15.0)),
                            "paper_count":  0,
                            "year_range":   "N/A",
                            "avg_citations":0.0,
                            "is_emerging":  False,
                        })
                return topics
        except Exception as e:
            logger.error(f"LLM topic extraction failed: {e}")

        return []

    # ─── RAG-Grounded Report ──────────────────────────────────────────────────

    def _generate_rag_report(
        self, query: str, papers: List[Dict], analysis: Dict
    ) -> str:
        """
        True RAG report:
        1. Retrieve the most relevant paper chunks for several sub-queries
        2. Inject retrieved text as grounding context into the report prompt
        3. LLM writes the report using only information from the retrieved chunks
        """

        # Build sub-queries covering different angles of the trend report
        sub_queries = [
            f"{query} emerging trends recent advances",
            f"{query} methodology techniques approaches",
            f"{query} challenges limitations future directions",
            f"{query} applications real world deployment",
        ]

        all_context_parts = []
        seen_titles = set()

        for sq in sub_queries:
            chunks = self.vector_store.rag_retrieve_structured(sq, n_chunks=3)
            for chunk in chunks:
                title = chunk.get("paper_title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_context_parts.append(
                        f'["{title}" ({chunk.get("year","")}) '
                        f'relevance={chunk.get("relevance",0):.2f}]\n'
                        f'{chunk.get("text","")[:400]}'
                    )

        context = "\n\n---\n\n".join(all_context_parts[:12])  # cap at 12 chunks
        if not context:
            # No chunks yet (e.g. first run before indexing) — use paper titles
            context = "Paper titles:\n" + "\n".join(
                f'- {p.get("title","")} ({p.get("year","")})' for p in papers[:20]
            )

        trending_kws = [
            k["keyword"] for k in analysis["trending_keywords"][:15]
            if k["is_trending"]
        ]
        momentum = analysis["momentum_scores"]
        method   = analysis.get("topic_method", "llm")
        topics   = analysis.get("topics", [])
        topic_summary = "\n".join([
            f'  - {t["name"]} ({t.get("prevalence",0):.1f}%): '
            f'{", ".join(t.get("keywords",[])[:5])}'
            for t in topics[:7]
        ])

        prompt = f"""You are writing a Research Trend Analysis Report for: "{query}"

═══════════════════════════════════════════════
RETRIEVED LITERATURE CONTEXT (use this as evidence):
{context}
═══════════════════════════════════════════════

QUANTITATIVE SUMMARY:
- Papers analyzed       : {momentum.get("total_papers_analyzed", 0)}
- Recent papers (2022+) : {momentum.get("recent_papers_count", 0)} \
({round(momentum.get("recent_ratio", 0) * 100, 1)}%)
- Avg citations         : {momentum.get("avg_citations", 0)}
- Research momentum     : {momentum.get("momentum", "Unknown")}
- Year range            : {momentum.get("year_range", "N/A")}
- Topic modeling method : {method.upper()}

IDENTIFIED TOPICS ({len(topics)} found via {method}):
{topic_summary if topic_summary else "  None identified"}

TRENDING KEYWORDS: {", ".join(trending_kws) if trending_kws else "N/A"}

EMERGING AREAS: {", ".join([e["area"] for e in analysis.get("emerging_areas",[])])}
DECLINING AREAS: {", ".join([d["area"] for d in analysis.get("declining_areas",[])])}

═══════════════════════════════════════════════

Write a comprehensive, evidence-grounded Trend Analysis Report.
Cite specific papers from the RETRIEVED CONTEXT to support your claims.
Use markdown formatting.

## Executive Summary
2-3 sentences on the overall research trajectory and momentum.

## Hot Topics & Emerging Trends
What is gaining momentum? Reference specific papers and years from the context.

## Topic Landscape ({method.upper()} Results)
Describe the {len(topics)} topics discovered, their prevalence, and relationships.

## Keyword Evolution
Which terms are rising vs falling? Use the trending keyword data.

## Methodology Trends
What techniques and tools are researchers increasingly using?

## Predicted Future Directions (2025–2028)
Based on current trajectory, where is this field heading?

## Strategic Opportunities
Where should a new researcher focus for maximum impact?

Ground every claim in evidence from the retrieved literature context above."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.5)
