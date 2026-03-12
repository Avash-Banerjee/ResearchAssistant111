"""
ResearchIQ - Grant Writing Agent
====================================
Generates structured, funding-ready grant proposals aligned with agency guidelines.
Single LLM call to stay within free-tier rate limits.
"""

import logging
import re
from typing import List, Dict, Optional, Callable
from core.llm_client import get_llm
from config.settings import AGENT_CONFIGS, GRANT_AGENCIES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert grant proposal writer with 20+ years of experience
securing research funding from NSF, NIH, DARPA, DOE, and European agencies.
You craft compelling, technically rigorous proposals that clearly communicate research value,
innovation, feasibility, and broader impact. Your writing is persuasive, precise, and follows
official agency guidelines exactly."""

# Agency-specific section lists
AGENCY_SECTIONS = {
    "NSF": [
        "Project Summary",
        "Introduction & Motivation",
        "Related Work & Research Gap",
        "Research Objectives",
        "Technical Approach & Innovation",
        "Broader Impacts",
        "Evaluation Plan",
        "Timeline & Milestones",
    ],
    "NIH": [
        "Specific Aims",
        "Research Strategy: Significance",
        "Research Strategy: Innovation",
        "Research Strategy: Approach",
        "Human Subjects",
        "Timeline",
    ],
    "DARPA": [
        "Technical Abstract",
        "Technical Approach",
        "Innovation & Novelty",
        "Feasibility & Risk Mitigation",
        "Transition Plan",
        "Team Qualifications",
        "Schedule & Milestones",
    ],
    "DOE": [
        "Executive Summary",
        "Scientific Merit & Impact",
        "Technical Approach",
        "Work Plan",
        "Energy Relevance",
        "Team & Resources",
    ],
    "EU Horizon": [
        "Excellence: Objectives & Innovation",
        "Excellence: Methodology",
        "Impact: Expected Outcomes",
        "Impact: Dissemination Plan",
        "Implementation: Work Packages",
        "Implementation: Team & Resources",
    ],
}

AGENCY_EMPHASIS = {
    "NSF": "broader impacts and intellectual merit",
    "NIH": "significance, innovation, and approach",
    "DARPA": "technical innovation and transition potential",
    "DOE": "scientific excellence and energy relevance",
    "EU Horizon": "scientific excellence and societal impact",
}


class GrantWritingAgent:
    """
    Agent 5: Grant Writing
    - Generates complete proposal in a SINGLE LLM call (free-tier friendly)
    - Adapts to agency-specific guidelines (NSF, NIH, DARPA, etc.)
    - Includes executive summary, budget justification, broader impacts
    """

    def __init__(self):
        self.llm = get_llm()
        self.config = AGENT_CONFIGS["grant_writer"]

    def write_grant(
        self,
        research_topic: str,
        research_gap: str,
        methodology: str,
        agency: str = "NSF",
        budget_range: str = "$100K-$500K",
        duration: str = "3 years",
        pi_info: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """Generate a complete grant proposal in a single LLM call."""

        pi = pi_info or {"name": "Principal Investigator", "institution": "University", "department": "Computer Science"}
        sections = AGENCY_SECTIONS.get(agency, AGENCY_SECTIONS["NSF"])
        emphasis = AGENCY_EMPHASIS.get(agency, AGENCY_EMPHASIS["NSF"])

        if progress_callback:
            progress_callback(f"✍️ Generating {agency} grant proposal (single pass)...")

        # Build section list for the prompt
        section_list = "\n".join(f"  {i}. {s}" for i, s in enumerate(sections, 1))

        prompt = f"""Write a complete {agency} grant proposal for the following research.

RESEARCH TOPIC: {research_topic}
RESEARCH GAP: {research_gap}
METHODOLOGY OVERVIEW: {methodology[:500] if methodology else "To be defined"}
FUNDING: {budget_range} over {duration}
PI: {pi.get('name')} at {pi.get('institution')}, {pi.get('department')}

Write ALL of the following sections (use these exact headings with ## prefix):

{section_list}

Then also include these additional sections:
  ## Executive Summary (200 words — problem, approach, innovation, impact)
  ## Budget Justification (personnel, equipment, travel, materials, overhead — with rough costs)
  ## Broader Impacts (education, community, industry, diversity, dissemination)

Requirements:
- Follow {agency} guidelines, emphasize {emphasis}
- Use formal academic language, be specific and concrete
- Each section: 150-300 words
- DO NOT use placeholder text — write actual compelling content
- Start each section with its ## heading on its own line"""

        raw = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.6, max_tokens=4096)

        if progress_callback:
            progress_callback("📄 Parsing proposal sections...")

        # Parse sections from the single response
        parsed_sections = self._parse_sections(raw, sections)

        # Build header
        header = f"""{'='*70}
{agency.upper()} GRANT PROPOSAL
{'='*70}
PROJECT TITLE: {research_topic.upper()[:80]}
{'─'*70}
Principal Investigator: {pi.get('name', 'N/A')}
Institution: {pi.get('institution', 'N/A')}
Department: {pi.get('department', 'N/A')}
Requested Budget: {budget_range}
Project Duration: {duration}
{'='*70}

"""
        body = ""
        for i, (section, content) in enumerate(parsed_sections.items(), 1):
            body += f"\n{'─'*60}\n{i}. {section.upper()}\n{'─'*60}\n\n{content}\n\n"

        full_proposal = header + body

        return {
            "agency": agency,
            "topic": research_topic,
            "sections": parsed_sections,
            "full_proposal": full_proposal,
            "executive_summary": parsed_sections.get("Executive Summary", ""),
            "budget_justification": parsed_sections.get("Budget Justification", ""),
            "broader_impacts": parsed_sections.get("Broader Impacts", ""),
            "agent": self.config["name"],
        }

    def _parse_sections(self, raw: str, expected_sections: List[str]) -> Dict[str, str]:
        """Parse ## headed sections from raw LLM output."""
        all_expected = list(expected_sections) + ["Executive Summary", "Budget Justification", "Broader Impacts"]

        # Split on ## headings
        parts = re.split(r'\n##\s+', raw)
        parsed = {}

        for part in parts:
            if not part.strip():
                continue
            lines = part.strip().split('\n', 1)
            heading = lines[0].strip().rstrip('#').strip()
            content = lines[1].strip() if len(lines) > 1 else ""

            # Match to expected section (fuzzy: check if expected name is contained)
            matched = False
            for expected in all_expected:
                if expected.lower() in heading.lower() or heading.lower() in expected.lower():
                    parsed[expected] = content
                    matched = True
                    break
            if not matched and content:
                parsed[heading] = content

        return parsed

    def write_ieee_acm_paper(
        self,
        title: str,
        abstract: str,
        gap: str,
        methodology: str,
        format_style: str = "IEEE",
    ) -> str:
        """Generate a structured IEEE/ACM-style research paper draft."""
        prompt = f"""Generate a structured {format_style}-style research paper draft.

PAPER TITLE: {title}
ABSTRACT: {abstract}
RESEARCH GAP ADDRESSED: {gap}
METHODOLOGY: {methodology[:500]}

Write the full paper with these sections:
## Abstract (150-250 words)
## 1. Introduction (Context, Problem, Gap, Contribution)
## 2. Related Work (3-4 paragraphs)
## 3. Problem Formulation (formal definition)
## 4. Proposed Method (main contribution)
## 5. Experiments (setup, datasets, baselines, metrics)
## 6. Results and Discussion
## 7. Conclusion and Future Work
## References (5-8 key references in {format_style} format)

Write actual content for each section. Be technically rigorous."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.6, max_tokens=4096)
