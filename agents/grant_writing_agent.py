"""
ResearchIQ - Grant Writing Agent
====================================
Generates structured, funding-ready grant proposals aligned with agency guidelines.
"""

import logging
from typing import List, Dict, Optional, Callable
from core.llm_client import get_llm
from config.settings import AGENT_CONFIGS, GRANT_AGENCIES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert grant proposal writer with 20+ years of experience 
securing research funding from NSF, NIH, DARPA, DOE, and European agencies.
You craft compelling, technically rigorous proposals that clearly communicate research value,
innovation, feasibility, and broader impact. Your writing is persuasive, precise, and follows
official agency guidelines exactly."""


class GrantWritingAgent:
    """
    Agent 5: Grant Writing
    - Generates structured proposals (Problem → Method → Impact → Budget)
    - Adapts to agency-specific guidelines (NSF, NIH, DARPA, etc.)
    - IEEE/ACM-style research paper drafts
    - Executive summaries and lay abstracts
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
        """Generate a complete grant proposal."""

        result = {
            "agency": agency,
            "topic": research_topic,
            "sections": {},
            "full_proposal": "",
            "executive_summary": "",
            "budget_justification": "",
            "broader_impacts": "",
            "agent": self.config["name"],
        }

        pi = pi_info or {"name": "Principal Investigator", "institution": "University", "department": "Computer Science"}

        if progress_callback:
            progress_callback(f"✍️ Writing {agency} grant proposal...")

        # Generate section by section
        sections = self._get_agency_sections(agency)
        all_sections = {}

        for section_name in sections:
            if progress_callback:
                progress_callback(f"📝 Writing: {section_name}...")
            all_sections[section_name] = self._write_section(
                section_name, research_topic, research_gap, methodology, agency, budget_range, duration, pi
            )

        result["sections"] = all_sections
        result["full_proposal"] = self._assemble_full_proposal(
            all_sections, agency, research_topic, pi, budget_range, duration
        )
        result["executive_summary"] = self._write_executive_summary(
            research_topic, research_gap, agency
        )
        result["budget_justification"] = self._write_budget_justification(
            budget_range, duration, research_topic
        )
        result["broader_impacts"] = self._write_broader_impacts(research_topic, agency)

        return result

    def _get_agency_sections(self, agency: str) -> List[str]:
        """Get required sections for each agency."""
        sections = {
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
        return sections.get(agency, sections["NSF"])

    def _write_section(
        self,
        section: str,
        topic: str,
        gap: str,
        methodology: str,
        agency: str,
        budget: str,
        duration: str,
        pi: Dict,
    ) -> str:
        """Write a specific grant proposal section."""
        prompt = f"""Write the "{section}" section for a {agency} grant proposal.

RESEARCH TOPIC: {topic}
RESEARCH GAP: {gap}
METHODOLOGY OVERVIEW: {methodology[:500] if methodology else "To be defined"}
FUNDING: {budget} over {duration}
PI: {pi.get('name')} at {pi.get('institution')}, {pi.get('department')}

Write this section following strict {agency} guidelines:
- Use formal academic language
- Be specific and concrete (avoid vague statements)
- Cite the research need clearly
- For {agency}, emphasize: {"broader impacts and intellectual merit" if agency == "NSF" else "significance, innovation, and approach" if agency == "NIH" else "technical innovation and transition potential" if agency == "DARPA" else "scientific excellence and societal impact"}
- Length: 300-500 words appropriate for this section
- DO NOT use placeholder text - write actual compelling content

Section: {section}"""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.6, max_tokens=800)

    def _assemble_full_proposal(
        self,
        sections: Dict[str, str],
        agency: str,
        topic: str,
        pi: Dict,
        budget: str,
        duration: str,
    ) -> str:
        """Assemble all sections into a complete proposal document."""
        header = f"""{'='*70}
{agency.upper()} GRANT PROPOSAL
{'='*70}
PROJECT TITLE: {topic.upper()[:80]}
{'─'*70}
Principal Investigator: {pi.get('name', 'N/A')}
Institution: {pi.get('institution', 'N/A')}
Department: {pi.get('department', 'N/A')}
Requested Budget: {budget}
Project Duration: {duration}
{'='*70}

"""
        body = ""
        for i, (section, content) in enumerate(sections.items(), 1):
            body += f"\n{'─'*60}\n{i}. {section.upper()}\n{'─'*60}\n\n{content}\n\n"

        return header + body

    def _write_executive_summary(self, topic: str, gap: str, agency: str) -> str:
        """Write a concise executive summary / abstract."""
        prompt = f"""Write a compelling 200-word executive summary / project abstract for a {agency} grant proposal.

TOPIC: {topic}
GAP BEING ADDRESSED: {gap}

The summary must:
1. State the problem clearly (1-2 sentences)
2. Describe the proposed solution/approach (2-3 sentences)
3. Highlight innovation/novelty (1-2 sentences)
4. State expected outcomes and impact (1-2 sentences)
5. Mention team capability (1 sentence)

Use active voice. Be persuasive. Start with a hook sentence.
Do NOT use jargon without explanation. Keep to exactly ~200 words."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.6)

    def _write_budget_justification(self, budget: str, duration: str, topic: str) -> str:
        """Generate budget justification narrative."""
        prompt = f"""Write a detailed budget justification for a research grant:

TOTAL BUDGET: {budget}
DURATION: {duration}
RESEARCH TOPIC: {topic}

Include justifications for:
1. **Personnel** (PI effort, graduate students, postdocs, undergraduate RAs)
2. **Equipment & Computing** (servers, GPUs, specialized equipment)
3. **Travel** (conferences: NeurIPS, ICML, AAAI, IEEE/ACM venues)
4. **Materials & Supplies** (software licenses, cloud computing credits)
5. **Indirect Costs / Overhead** (standard institutional rate)
6. **Contingency** (if applicable)

For each category:
- Justify WHY it's needed for this specific research
- Provide rough cost breakdown
- Explain how it enables the research objectives

Use specific, realistic numbers. Total should roughly match the budget range."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.5)

    def _write_broader_impacts(self, topic: str, agency: str) -> str:
        """Write broader impacts / societal impact section."""
        prompt = f"""Write a compelling "Broader Impacts" section for a {agency} proposal on: {topic}

Cover:
1. **Educational Impact**: Student training, curriculum development, diversity
2. **Scientific Community**: Open-source tools, datasets, reproducibility
3. **Industry Translation**: Potential commercial applications
4. **Societal Benefits**: How this research benefits society broadly
5. **Diversity & Inclusion**: Plans for engaging underrepresented groups
6. **Dissemination**: Publications, workshops, public outreach

Be specific. Mention concrete activities. Avoid generic statements.
Length: 300-400 words."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.6)

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

Write the full paper structure with actual content:

## Abstract
[150-250 word abstract]

## 1. Introduction
[Context → Problem → Gap → Contribution → Paper Organization]

## 2. Related Work
[3-4 paragraphs covering key related work and positioning]

## 3. Problem Formulation
[Formal problem definition with notation]

## 4. Proposed Method
[Technical approach - the main contribution]

## 5. Experiments
[Experimental setup, datasets, baselines, metrics]

## 6. Results and Discussion
[Expected results format and discussion points]

## 7. Conclusion and Future Work
[Summary and next steps]

## References
[5-8 key references in {format_style} format]

Write actual content for each section. Use {format_style} formatting conventions.
Be technically rigorous and specific."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.6, max_tokens=4096)
