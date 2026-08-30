"""
LLM Synthesis Engine.

Synthesizes outputs from multiple scientific tools and GeoChat visual
reasoning into a single coherent, authoritative, grounded answer.
Strictly guards against numerical hallucination by injecting ground-truth metrics.
"""

from __future__ import annotations

import logging
from typing import Any

from core.models import AnalysisResult, RasterImage
from llm.base import LLMMessage
from llm.client import get_llm_client, LLMClient
from llm.planner import ExecutionPlan

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """You are SatQuery AI, an authoritative multimodal remote-sensing assistant.
Your job is to synthesize raw tool outputs and visual reasoning into ONE coherent, clear, professional answer for the user.

CRITICAL GROUNDING RULES:
1. SCIENTIFIC TOOLS ARE AUTHORITATIVE:
   - ALL numbers, percentages, pixel counts, thresholds, and index formulas MUST match the Tool Outputs exactly.
   - NEVER invent or estimate new remote-sensing numbers, indices, or coverage metrics.
2. HONEST REPORTING:
   - If a tool used an RGB visual proxy (heuristic) instead of true multispectral data (NDVI/NDWI/NDBI), clearly explain that true spectral calculation requires NIR/SWIR bands.
3. COHESIVE SYNTHESIS:
   - Provide one unified response. Do NOT repeat redundant boilerplate from individual tools.
   - If GeoChat provided visual context, weave it together with the quantitative measurements naturally.
4. FORMAT:
   - Use clean Markdown with bullet points or bold headers where helpful.
"""


def synthesize_results(
    query: str,
    plan: ExecutionPlan,
    tool_results: list[AnalysisResult],
    image1: RasterImage,
    session_context: Any = None,
    llm_client: LLMClient | None = None,
) -> str:
    """
    Synthesize multiple tool outputs and visual reasoning into a single grounded answer.
    """
    if not tool_results:
        return "⚠️ No analysis results were produced for your query."

    # If only one tool was run and synthesis is not strictly required or LLM is offline,
    # return the tool's answer directly
    client = llm_client or get_llm_client()
    if not client.is_available:
        return _fallback_rule_based_synthesis(tool_results)

    # Format tool outputs and ground-truth metrics for the synthesizer
    context_blocks = []
    for i, res in enumerate(tool_results, 1):
        block = [
            f"--- Tool #{i}: {res.tool_used} ---",
            f"Answer: {res.answer}",
        ]
        if res.metadata:
            metrics = {k: v for k, v in res.metadata.items() if k not in ("error", "filename")}
            if metrics:
                block.append(f"Authoritative Metrics: {metrics}")
        context_blocks.append("\n".join(block))

    tools_text = "\n\n".join(context_blocks)

    user_prompt = f"""USER QUERY: "{query}"

PLANNED GOAL: {plan.reasoning}

IMAGE INFORMATION:
- Primary Image: {image1.width}x{image1.height} px, {image1.num_bands} bands ({', '.join(image1.bands)}), Sensor: {image1.sensor_type.value}

GROUND-TRUTH TOOL OUTPUTS:
{tools_text}

Please generate the final synthesized answer following the grounding rules.
"""

    messages = [
        LLMMessage(role="system", content=_SYNTHESIS_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]

    try:
        response = client.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )

        if response and response.content.strip():
            logger.info("LLM synthesis completed successfully.")
            return response.content.strip()

    except Exception as e:
        logger.warning(f"LLM synthesis failed: {e}. Falling back to rule-based synthesis.")

    return _fallback_rule_based_synthesis(tool_results)


def _fallback_rule_based_synthesis(tool_results: list[AnalysisResult]) -> str:
    """Combine answers deterministically when LLM is unavailable."""
    if len(tool_results) == 1:
        return tool_results[0].answer

    sections = []
    for res in tool_results:
        sections.append(f"### {res.tool_used}\n{res.answer}")

    return "\n\n---\n\n".join(sections)
