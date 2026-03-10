"""
ResearchIQ - Methodology Design Agent
=========================================
Suggests experimental designs, datasets, evaluation metrics, and baselines.
"""

import logging
import json
from typing import List, Dict, Optional, Callable
from core.llm_client import get_llm
from config.settings import AGENT_CONFIGS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior Research Methodology Expert with expertise across
computer science, AI/ML, and related disciplines. You design rigorous experimental frameworks,
recommend appropriate datasets and baselines, and ensure methodological soundness.
Your recommendations are practical, reproducible, and aligned with top-venue standards."""


class MethodologyDesignAgent:
    """
    Agent 4: Methodology Design
    - Suggests experimental frameworks
    - Recommends datasets and benchmarks
    - Identifies appropriate baselines
    - Designs evaluation protocols
    - Generates hypotheses
    """

    def __init__(self):
        self.llm = get_llm()
        self.config = AGENT_CONFIGS["methodology_designer"]

    # ─── Main Entry Point ─────────────────────────────────────────────────────

    def design_methodology(
        self,
        research_gap: str,
        domain: str,
        papers: List[Dict],
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """Design a complete research methodology for a given gap."""

        result = {
            "research_gap": research_gap,
            "domain": domain,
            "hypotheses": [],
            "experimental_design": {},
            "datasets": [],
            "baselines": [],
            "evaluation_metrics": [],
            "methodology_report": "",
            "agent": self.config["name"],
        }

        if progress_callback:
            progress_callback("💡 Generating research hypotheses...")
        result["hypotheses"] = self._generate_hypotheses(research_gap, domain, papers)
        logger.info(f"Hypotheses generated: {len(result['hypotheses'])}")

        if progress_callback:
            progress_callback("⚗️ Designing experimental framework...")
        result["experimental_design"] = self._design_experiment(research_gap, domain)

        if progress_callback:
            progress_callback("📊 Recommending datasets and benchmarks...")
        result["datasets"] = self._recommend_datasets(domain, papers)
        logger.info(f"Datasets recommended: {len(result['datasets'])}")

        if progress_callback:
            progress_callback("🏁 Identifying baselines...")
        result["baselines"] = self._identify_baselines(domain)
        logger.info(f"Baselines identified: {len(result['baselines'])}")

        if progress_callback:
            progress_callback("📏 Recommending evaluation metrics...")
        result["evaluation_metrics"] = self._recommend_metrics(domain, research_gap)
        logger.info(f"Metrics recommended: {len(result['evaluation_metrics'])}")

        if progress_callback:
            progress_callback("📋 Generating methodology report...")
        result["methodology_report"] = self._generate_methodology_report(result)

        return result

    # ─── Hypotheses ───────────────────────────────────────────────────────────

    def _generate_hypotheses(
        self, research_gap: str, domain: str, papers: List[Dict]
    ) -> List[Dict]:
        """Generate testable hypotheses. Uses JSON output for reliability."""

        context_papers = "\n".join([
            f"- {p.get('title', '')} ({p.get('year', '')})"
            for p in papers[:10]
        ])

        prompt = f"""Generate exactly 4 specific, testable research hypotheses for this research gap.

RESEARCH GAP: {research_gap}
DOMAIN: {domain}
RELATED WORK:
{context_papers}

Return ONLY a valid JSON array, no markdown, no code fences, no explanation:
[
  {{
    "id": "H1",
    "statement": "clear testable hypothesis statement",
    "type": "Alternative",
    "rationale": "why this hypothesis matters",
    "test_approach": "how to test this hypothesis experimentally",
    "expected_outcome": "what result confirms or rejects it",
    "novelty": "what makes this hypothesis novel"
  }},
  {{
    "id": "H2",
    "statement": "...",
    "type": "Directional",
    "rationale": "...",
    "test_approach": "...",
    "expected_outcome": "...",
    "novelty": "..."
  }},
  {{
    "id": "H3",
    "statement": "...",
    "type": "Null",
    "rationale": "...",
    "test_approach": "...",
    "expected_outcome": "...",
    "novelty": "..."
  }},
  {{
    "id": "H4",
    "statement": "...",
    "type": "Alternative",
    "rationale": "...",
    "test_approach": "...",
    "expected_outcome": "...",
    "novelty": "..."
  }}
]"""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        hypotheses = self._parse_json_list(response)

        if not hypotheses:
            logger.warning("Hypothesis JSON parse failed, trying structured fallback")
            hypotheses = self._generate_hypotheses_structured(research_gap, domain, context_papers)

        return hypotheses

    def _generate_hypotheses_structured(
        self, research_gap: str, domain: str, context_papers: str
    ) -> List[Dict]:
        """Fallback: structured key-value format for hypotheses."""
        prompt = f"""Generate 3 research hypotheses for:
RESEARCH GAP: {research_gap}
DOMAIN: {domain}

Use EXACTLY this format:
H_ID: H1
STATEMENT: [hypothesis]
TYPE: Alternative
RATIONALE: [rationale]
TEST_APPROACH: [approach]
EXPECTED_OUTCOME: [outcome]
NOVELTY: [novelty]
---
H_ID: H2
STATEMENT: [hypothesis]
TYPE: Directional
RATIONALE: [rationale]
TEST_APPROACH: [approach]
EXPECTED_OUTCOME: [outcome]
NOVELTY: [novelty]
---
H_ID: H3
STATEMENT: [hypothesis]
TYPE: Null
RATIONALE: [rationale]
TEST_APPROACH: [approach]
EXPECTED_OUTCOME: [outcome]
NOVELTY: [novelty]
---"""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        hypotheses = []
        blocks = [b.strip() for b in response.replace("```", "").split("---") if b.strip()]
        for block in blocks:
            h = {}
            for line in block.split("\n"):
                key, _, value = line.strip().partition(":")
                key = key.strip().upper().replace(" ", "_").replace("*", "")
                value = value.strip()
                if not value:
                    continue
                if key == "H_ID":
                    h["id"] = value
                elif key == "STATEMENT":
                    h["statement"] = value
                elif key == "TYPE":
                    h["type"] = value
                elif key == "RATIONALE":
                    h["rationale"] = value
                elif key == "TEST_APPROACH":
                    h["test_approach"] = value
                elif key == "EXPECTED_OUTCOME":
                    h["expected_outcome"] = value
                elif key == "NOVELTY":
                    h["novelty"] = value
            if h.get("statement"):
                h.setdefault("id", f"H{len(hypotheses)+1}")
                h.setdefault("type", "Alternative")
                hypotheses.append(h)
        return hypotheses

    # ─── Experimental Design ──────────────────────────────────────────────────

    def _design_experiment(self, research_gap: str, domain: str) -> Dict:
        """Design experimental framework — returns prose markdown."""
        prompt = f"""Design a rigorous experimental framework for this research:

RESEARCH GAP: {research_gap}
DOMAIN: {domain}

Write a detailed experimental design covering these sections:

## Research Objectives
List 3-4 specific, measurable objectives.

## Experimental Setup
Hardware, software stack, environment configuration.

## Data Pipeline
Data collection → preprocessing → augmentation → train/val/test splitting.

## Proposed Approach
Core technical approach with architectural details and justification.

## Training Protocol
Training setup, optimizer, learning rate schedule, batch size, epochs.

## Evaluation Protocol
Cross-validation strategy, ablation studies, statistical testing.

## Ablation Studies
List the specific components to ablate and why each matters.

## Reproducibility Checklist
Steps to ensure full reproducibility: seeds, code release, data availability.

## Timeline
Phase-by-phase timeline estimate (months).

Be specific with numbers, tools, and techniques."""

        response = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.4)
        return {"framework": response}

    # ─── Datasets ─────────────────────────────────────────────────────────────

    def _recommend_datasets(self, domain: str, papers: List[Dict]) -> List[Dict]:
        """Recommend datasets using JSON output."""

        # Collect dataset hints from abstracts
        hints = []
        for p in papers[:15]:
            abstract = p.get("abstract", "").lower()
            for kw in ["dataset", "benchmark", "corpus", "collection"]:
                if kw in abstract:
                    hints.append(p.get("title", "")[:80])
                    break

        prompt = f"""Recommend 6 appropriate datasets or benchmarks for research in: "{domain}"

Hints from related papers: {str(hints[:5]) if hints else "none"}

Return ONLY a valid JSON array, no markdown, no code fences:
[
  {{
    "name": "dataset name",
    "source": "URL or access method",
    "size": "approximate size (e.g. 50K samples)",
    "type": "text / image / graph / multimodal / tabular",
    "use_case": "specific use for this research",
    "pros": "key advantages",
    "cons": "key limitations",
    "citation": "key paper name if known"
  }}
]
Include 6 items total."""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        datasets = self._parse_json_list(response)

        if not datasets:
            logger.warning("Dataset JSON parse failed, trying structured fallback")
            datasets = self._recommend_datasets_structured(domain)

        return datasets

    def _recommend_datasets_structured(self, domain: str) -> List[Dict]:
        """Fallback structured parser for datasets."""
        prompt = f"""Recommend 5 datasets for research in: "{domain}"

Use EXACTLY this format for each:
DATASET_NAME: [name]
SOURCE: [url or location]
SIZE: [size]
TYPE: [type]
USE_CASE: [use case]
PROS: [pros]
CONS: [cons]
CITATION: [paper]
---"""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        datasets = []
        blocks = [b.strip() for b in response.replace("```", "").split("---") if b.strip()]
        for block in blocks:
            d = {}
            for line in block.split("\n"):
                key, _, value = line.strip().partition(":")
                key = key.strip().upper().replace(" ", "_").replace("*", "")
                value = value.strip()
                if not value:
                    continue
                mapping = {
                    "DATASET_NAME": "name", "SOURCE": "source", "SIZE": "size",
                    "TYPE": "type", "USE_CASE": "use_case", "PROS": "pros",
                    "CONS": "cons", "CITATION": "citation",
                }
                if key in mapping:
                    d[mapping[key]] = value
            if d.get("name"):
                datasets.append(d)
        return datasets

    # ─── Baselines ────────────────────────────────────────────────────────────

    def _identify_baselines(self, domain: str) -> List[Dict]:
        """Identify baseline methods using JSON output."""

        prompt = f"""Identify 6 appropriate baseline methods for research in: "{domain}"

Return ONLY a valid JSON array, no markdown, no code fences:
[
  {{
    "name": "method name",
    "type": "classical / ML / deep_learning / LLM / domain_specific",
    "description": "brief description of what this method does",
    "strengths": "why include this as a baseline",
    "weaknesses": "known limitations",
    "implementation": "GitHub URL or library name"
  }}
]
Include 6 items total."""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        baselines = self._parse_json_list(response)

        if not baselines:
            logger.warning("Baseline JSON parse failed, trying structured fallback")
            baselines = self._identify_baselines_structured(domain)

        return baselines

    def _identify_baselines_structured(self, domain: str) -> List[Dict]:
        """Fallback structured parser for baselines."""
        prompt = f"""List 5 baseline methods for research in: "{domain}"

Use EXACTLY this format:
BASELINE_NAME: [name]
TYPE: [type]
DESCRIPTION: [description]
STRENGTHS: [strengths]
WEAKNESSES: [weaknesses]
IMPLEMENTATION: [code/library]
---"""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        baselines = []
        blocks = [b.strip() for b in response.replace("```", "").split("---") if b.strip()]
        for block in blocks:
            b = {}
            for line in block.split("\n"):
                key, _, value = line.strip().partition(":")
                key = key.strip().upper().replace(" ", "_").replace("*", "")
                value = value.strip()
                if not value:
                    continue
                mapping = {
                    "BASELINE_NAME": "name", "TYPE": "type",
                    "DESCRIPTION": "description", "STRENGTHS": "strengths",
                    "WEAKNESSES": "weaknesses", "IMPLEMENTATION": "implementation",
                }
                if key in mapping:
                    b[mapping[key]] = value
            if b.get("name"):
                baselines.append(b)
        return baselines

    # ─── Metrics ──────────────────────────────────────────────────────────────

    def _recommend_metrics(self, domain: str, research_gap: str) -> List[Dict]:
        """Recommend evaluation metrics using JSON output."""

        prompt = f"""Recommend 6 evaluation metrics for research in "{domain}" addressing:
"{research_gap[:150]}"

Return ONLY a valid JSON array, no markdown, no code fences:
[
  {{
    "name": "metric name",
    "category": "accuracy / efficiency / robustness / fairness / interpretability",
    "formula": "brief formula or definition",
    "use_when": "when this metric is most appropriate",
    "limitations": "known caveats or limitations"
  }}
]
Include 6 items total."""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        metrics = self._parse_json_list(response)

        if not metrics:
            logger.warning("Metrics JSON parse failed, trying structured fallback")
            metrics = self._recommend_metrics_structured(domain, research_gap)

        return metrics

    def _recommend_metrics_structured(self, domain: str, research_gap: str) -> List[Dict]:
        """Fallback structured parser for metrics."""
        prompt = f"""List 5 evaluation metrics for research in "{domain}".

Use EXACTLY this format:
METRIC_NAME: [name]
CATEGORY: [category]
FORMULA: [formula]
USE_WHEN: [when to use]
LIMITATIONS: [limitations]
---"""

        response = self.llm.generate(prompt, system_prompt=None, temperature=0.3)
        metrics = []
        blocks = [b.strip() for b in response.replace("```", "").split("---") if b.strip()]
        for block in blocks:
            m = {}
            for line in block.split("\n"):
                key, _, value = line.strip().partition(":")
                key = key.strip().upper().replace(" ", "_").replace("*", "")
                value = value.strip()
                if not value:
                    continue
                mapping = {
                    "METRIC_NAME": "name", "CATEGORY": "category",
                    "FORMULA": "formula", "USE_WHEN": "use_when",
                    "LIMITATIONS": "limitations",
                }
                if key in mapping:
                    m[mapping[key]] = value
            if m.get("name"):
                metrics.append(m)
        return metrics

    # ─── Methodology Report ───────────────────────────────────────────────────

    def _generate_methodology_report(self, result: Dict) -> str:
        """Generate final publication-ready methodology section."""

        hyp_lines = "\n".join([
            f"- {h.get('id','H?')}: {h.get('statement','')}"
            for h in result["hypotheses"]
        ]) or "No hypotheses generated."

        dataset_names = ", ".join([
            d.get("name", "") for d in result["datasets"][:5] if d.get("name")
        ]) or "To be determined."

        baseline_names = ", ".join([
            b.get("name", "") for b in result["baselines"][:5] if b.get("name")
        ]) or "To be determined."

        metric_names = ", ".join([
            m.get("name", "") for m in result["evaluation_metrics"][:5] if m.get("name")
        ]) or "To be determined."

        prompt = f"""Write a complete, formal, publication-ready Methodology section (IEEE/ACM style).

RESEARCH GAP: {result["research_gap"]}
DOMAIN: {result["domain"]}

HYPOTHESES:
{hyp_lines}

DATASETS: {dataset_names}
BASELINES: {baseline_names}
EVALUATION METRICS: {metric_names}

Write the full methodology section with these subsections:

## 3. Methodology

### 3.1 Problem Formulation
Formal problem definition with mathematical notation where appropriate.

### 3.2 Proposed Approach
Describe the core technical approach, architecture, and key innovations.

### 3.3 Data Collection & Preprocessing
Detail the datasets used, preprocessing pipeline, and data splits.

### 3.4 Experimental Setup
Hardware specifications, software frameworks, hyperparameter configuration.

### 3.5 Evaluation Protocol
Describe evaluation methodology: cross-validation, ablations, significance tests.

### 3.6 Baseline Comparisons
Justify the choice of each baseline and how comparisons are structured.

### 3.7 Statistical Significance Testing
Describe statistical tests used to validate results (t-test, Wilcoxon, etc.).

Use formal academic language. Be specific — include example values for hyperparameters,
dataset splits, hardware specs. Avoid vague statements."""

        return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.4)

    # ─── Shared JSON Parser ───────────────────────────────────────────────────

    @staticmethod
    def _parse_json_list(response: str) -> List[Dict]:
        """
        Robustly extract a JSON array from LLM response.
        Handles: markdown fences, leading prose, trailing text.
        """
        if not response:
            return []

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

        # Find JSON array boundaries
        start = clean.find("[")
        end = clean.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.debug(f"No JSON array found. Response preview: {clean[:200]}")
            return []

        try:
            items = json.loads(clean[start:end + 1])
            if isinstance(items, list):
                # Filter to only dict items
                return [item for item in items if isinstance(item, dict)]
            return []
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}. Preview: {clean[start:start+200]}")
            return []