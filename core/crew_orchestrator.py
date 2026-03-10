"""
ResearchIQ - CrewAI Orchestrator
===================================
Coordinates all agents using CrewAI framework for end-to-end workflows.
"""

import logging
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class ResearchCrew:
    """
    Orchestrates the multi-agent research pipeline using CrewAI.
    Runs agents in sequence: Mine → Trends → Gaps → Methodology → Grant → Score
    """

    def __init__(self):
        self._agents_loaded = False
        self._load_agents()

    def _load_agents(self):
        """Lazy load all agents."""
        try:
            from agents.literature_mining_agent import LiteratureMiningAgent
            from agents.trend_analysis_agent import TrendAnalysisAgent
            from agents.gap_identification_agent import GapIdentificationAgent
            from agents.methodology_design_agent import MethodologyDesignAgent
            from agents.grant_writing_agent import GrantWritingAgent
            from agents.novelty_scoring_agent import NoveltyScoringAgent

            self.literature_agent = LiteratureMiningAgent()
            self.trend_agent = TrendAnalysisAgent()
            self.gap_agent = GapIdentificationAgent()
            self.methodology_agent = MethodologyDesignAgent()
            self.grant_agent = GrantWritingAgent()
            self.novelty_agent = NoveltyScoringAgent()
            self._agents_loaded = True
            logger.info("All agents loaded successfully")
        except Exception as e:
            logger.error(f"Agent loading error: {e}")
            raise

    def run_full_pipeline(
        self,
        query: str,
        agency: str = "NSF",
        budget: str = "$100K-$500K",
        duration: str = "3 years",
        pi_info: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None,
        sources: List[str] = ("arxiv", "semantic_scholar"),
    ) -> Dict:
        """
        Run the complete research intelligence pipeline.
        Returns consolidated results from all agents.
        """
        pipeline_result = {
            "query": query,
            "agency": agency,
            "pipeline_stages": {},
            "status": "running",
        }

        def update_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        try:
            # ─── Stage 1: Literature Mining ───────────────────────────────────
            update_progress("🔍 Stage 1/6: Literature Mining Agent starting...")
            lit_result = self.literature_agent.mine_literature(
                query=query,
                sources=list(sources),
                max_papers=30,
                progress_callback=update_progress,
            )
            pipeline_result["pipeline_stages"]["literature"] = lit_result
            papers = lit_result.get("papers", [])
            update_progress(f"✅ Literature: Found {len(papers)} papers")

            # ─── Stage 2: Trend Analysis ──────────────────────────────────────
            update_progress("📈 Stage 2/6: Trend Analysis Agent starting...")
            trend_result = self.trend_agent.analyze_trends(
                papers=papers,
                query=query,
                progress_callback=update_progress,
            )
            pipeline_result["pipeline_stages"]["trends"] = trend_result
            update_progress("✅ Trends: Analysis complete")

            # ─── Stage 3: Gap Identification ──────────────────────────────────
            update_progress("🎯 Stage 3/6: Gap Identification Agent starting...")
            gap_result = self.gap_agent.identify_gaps(
                papers=papers,
                query=query,
                progress_callback=update_progress,
            )
            pipeline_result["pipeline_stages"]["gaps"] = gap_result
            gaps = gap_result.get("gaps", [])
            top_gap = gaps[0].get("title", query) if gaps else query
            update_progress(f"✅ Gaps: Found {len(gaps)} research gaps")

            # ─── Stage 4: Methodology Design ──────────────────────────────────
            update_progress("⚗️ Stage 4/6: Methodology Design Agent starting...")
            method_result = self.methodology_agent.design_methodology(
                research_gap=top_gap,
                domain=query,
                papers=papers,
                progress_callback=update_progress,
            )
            pipeline_result["pipeline_stages"]["methodology"] = method_result
            update_progress("✅ Methodology: Design complete")

            # ─── Stage 5: Grant Writing ────────────────────────────────────────
            update_progress("📝 Stage 5/6: Grant Writing Agent starting...")
            methodology_text = method_result.get("methodology_report", "")[:1000]
            grant_result = self.grant_agent.write_grant(
                research_topic=query,
                research_gap=top_gap,
                methodology=methodology_text,
                agency=agency,
                budget_range=budget,
                duration=duration,
                pi_info=pi_info,
                progress_callback=update_progress,
            )
            pipeline_result["pipeline_stages"]["grant"] = grant_result
            update_progress("✅ Grant: Proposal complete")

            # ─── Stage 6: Novelty Scoring ──────────────────────────────────────
            update_progress("🏆 Stage 6/6: Novelty Scoring Agent starting...")
            proposal_abstract = grant_result.get("executive_summary", "")
            novelty_result = self.novelty_agent.score_novelty(
                title=f"Research on {query}",
                abstract=proposal_abstract,
                methodology=methodology_text,
                progress_callback=update_progress,
            )
            pipeline_result["pipeline_stages"]["novelty"] = novelty_result
            update_progress("✅ Novelty: Scoring complete")

            pipeline_result["status"] = "completed"
            pipeline_result["summary"] = self._generate_pipeline_summary(pipeline_result)
            update_progress("🎉 Full pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            pipeline_result["status"] = "error"
            pipeline_result["error"] = str(e)

        return pipeline_result

    def _generate_pipeline_summary(self, result: Dict) -> Dict:
        """Generate executive summary of the full pipeline."""
        stages = result.get("pipeline_stages", {})
        lit = stages.get("literature", {})
        gaps = stages.get("gaps", {})
        novelty = stages.get("novelty", {})

        return {
            "papers_found": len(lit.get("papers", [])),
            "papers_indexed": lit.get("new_papers_indexed", 0),
            "gaps_identified": len(gaps.get("gaps", [])),
            "gap_score": gaps.get("gap_score", 0),
            "novelty_score": novelty.get("novelty_score", 0),
            "verdict": novelty.get("verdict", "N/A"),
            "top_gap": gaps.get("gaps", [{}])[0].get("title", "N/A") if gaps.get("gaps") else "N/A",
        }
