"""
LLM Synthesis Engine with Structured Grounded Output.

Synthesizes outputs from multiple scientific tools and GeoChat visual
reasoning into a single coherent, authoritative, grounded answer.
Strictly guards against numerical hallucination by injecting ground-truth metrics.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from pydantic import BaseModel, Field

from core.models import AnalysisResult, RasterImage
from llm.base import LLMMessage
from llm.client import get_llm_client, LLMClient
from llm.planner import ExecutionPlan

logger = logging.getLogger(__name__)


class StructuredSynthesis(BaseModel):
    """Structured synthesis model ensuring strict grounding and scannable output."""
    status: str = Field(default="success", description="'success' | 'out_of_scope' | 'insufficient_evidence'")
    intent: str = Field(default="", description="Identified query intent")
    summary: str = Field(default="", description="Direct concise answer to the user's question")
    observations: list[str] = Field(default_factory=list, description="Key grounded factual observations")
    evidence: list[str] = Field(default_factory=list, description="Explicit evidence sources (e.g. Uploaded image, NDWI calculation)")
    confidence: str = Field(default="", description="Confidence level with brief factual justification")
    sources: list[str] = Field(default_factory=list, description="Authoritative tools or algorithms used")
    answer: str = Field(default="", description="Full formatted markdown text")

    def __str__(self) -> str:
        return self.answer or self.summary

    def __contains__(self, item: str) -> bool:
        combined = f"{self.summary} {self.answer} {' '.join(self.observations)} {' '.join(self.sources)} {' '.join(self.evidence)}"
        return item in combined

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.answer == other or self.summary == other or str(self) == other
        return super().__eq__(other)

    def format_markdown(self) -> str:
        """Format the structured synthesis into clean, scannable Markdown."""
        parts = []

        if self.summary:
            parts.append(f"### Summary\n{self.summary}")

        if self.observations:
            obs_lines = "\n".join(f"• {obs}" for obs in self.observations if obs.strip())
            if obs_lines:
                parts.append(f"### Key Observations\n{obs_lines}")

        if self.evidence:
            ev_lines = "\n".join(f"✓ {ev}" for ev in self.evidence if ev.strip())
            if ev_lines:
                parts.append(f"### Evidence\n{ev_lines}")

        if self.confidence:
            parts.append(f"### Confidence\n{self.confidence}")

        if not parts:
            return self.answer or "No analysis results available."

        return "\n\n".join(parts)


_SYNTHESIS_SYSTEM_PROMPT = """You are SatQuery AI, an authoritative multimodal remote-sensing assistant.
Your job is to synthesize raw tool outputs, visual reasoning, and image metadata into ONE grounded, concise, professional response.

CRITICAL GROUNDING RULES:
1. SCIENTIFIC TOOLS ARE AUTHORITATIVE:
   - ALL numbers, percentages, pixel counts, thresholds, and index formulas MUST match the Tool Outputs exactly.
   - NEVER invent or estimate new remote-sensing numbers, indices, or coverage metrics.
2. HONEST REPORTING & TRANSPARENCY:
   - If a tool used an RGB visual proxy (heuristic) instead of true multispectral data (NDVI/NDWI/NDBI), clearly explain that true spectral calculation requires NIR/SWIR bands.
   - If evidence is insufficient (e.g. missing bands or missing second image), explicitly state what is missing.
3. CONCISE & USER-FRIENDLY:
   - Keep the summary direct and understandable. Avoid unnecessary technical jargon.
   - Do NOT create sections with empty or fabricated content.
4. OUT-OF-SCOPE QUERIES:
   - If the query is unrelated to remote sensing or satellite analysis, return status "out_of_scope" with a friendly message asking for a satellite-related question.

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "success" | "out_of_scope" | "insufficient_evidence",
  "intent": "intent_name",
  "summary": "Direct, clear answer to the user's question.",
  "observations": [
    "Key factual observation 1",
    "Key factual observation 2"
  ],
  "evidence": [
    "Uploaded satellite image (dimensions, bands)",
    "Authoritative spectral calculation or model name"
  ],
  "confidence": "High / Moderate / Low with honest factual justification (e.g. 'High — derived from true multispectral NDWI calculation' or 'Moderate — RGB visual proxy without NIR band')",
  "sources": [
    "Tool or algorithm name"
  ]
}}
"""


def synthesize_results(
    query: str,
    plan: ExecutionPlan,
    tool_results: list[AnalysisResult],
    image1: RasterImage,
    session_context: Any = None,
    llm_client: LLMClient | None = None,
) -> StructuredSynthesis:
    """
    Synthesize multiple tool outputs and visual reasoning into a structured grounded answer.
    """
    if not tool_results:
        synth = StructuredSynthesis(
            status="insufficient_evidence",
            intent=plan.intent,
            summary="No analysis results were produced for your query.",
            observations=[],
            evidence=[],
            confidence="None",
            sources=[],
            answer="⚠️ No analysis results were produced for your query.",
        )
        return synth

    # Check if query is out of scope
    if plan.status == "out_of_scope" or plan.intent == "out_of_scope":
        msg = (
            "🛰️ **SatQuery AI is designed for satellite and Earth observation analysis.**\n\n"
            "Your question appears unrelated to satellite imagery or remote sensing.\n\n"
            "Please ask a question related to:\n"
            "• Satellite imagery interpretation & land cover\n"
            "• Water body detection & NDWI indices\n"
            "• Vegetation & crop health analysis (NDVI)\n"
            "• Urban & built-up area detection (NDBI)\n"
            "• Bi-temporal change detection (ChangeFormer)\n"
            "• Optical + SAR multi-modal classification (BIFOLD RDNet)"
        )
        return StructuredSynthesis(
            status="out_of_scope",
            intent="out_of_scope",
            summary="Query is outside the scope of satellite and Earth observation analysis.",
            observations=[],
            evidence=[],
            confidence="Not Applicable",
            sources=[],
            answer=msg,
        )

    client = llm_client or get_llm_client()

    # If synthesis is not required by the plan or LLM is offline,
    # generate structured synthesis from tool result(s) directly
    if not client.is_available or not plan.synthesis_required:
        return _fallback_rule_based_synthesis(query, plan, tool_results, image1)

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

Please generate the final synthesized JSON output following the grounding rules.
"""

    messages = [
        LLMMessage(role="system", content=_SYNTHESIS_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]

    try:
        response = client.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=900,
            response_format="json",
        )

        if response and response.content.strip():
            raw_content = response.content.strip()
            # Try parsing as JSON first
            try:
                data = json.loads(raw_content)
                if isinstance(data, dict) and ("summary" in data or "answer" in data or "observations" in data):
                    synth = StructuredSynthesis.model_validate(data)
                    if not synth.answer:
                        synth.answer = synth.format_markdown()
                    logger.info("LLM structured synthesis completed successfully from JSON.")
                    return synth
            except Exception as json_err:
                logger.warning(f"LLM JSON decode failed ({json_err}). Falling back to rule-based synthesis.")
                return _fallback_rule_based_synthesis(query, plan, tool_results, image1)

            # If plain text that does not look like JSON syntax, wrap cleanly
            if "{" not in raw_content and "}" not in raw_content:
                synth = StructuredSynthesis(
                    status="success",
                    intent=plan.intent,
                    summary=raw_content.split("\n\n")[0] if "\n\n" in raw_content else raw_content,
                    observations=[raw_content] if len(raw_content) < 300 else [p.strip() for p in raw_content.split("\n") if p.strip()],
                    evidence=[f"Uploaded satellite image ({image1.width}x{image1.height} px, {image1.num_bands} bands)"],
                    confidence="Moderate — LLM grounded synthesis",
                    sources=list(dict.fromkeys([r.tool_used for r in tool_results if r.tool_used])),
                    answer=raw_content,
                )
                return synth

            return _fallback_rule_based_synthesis(query, plan, tool_results, image1)


    except Exception as e:
        logger.warning(f"LLM synthesis failed or malformed: {e}. Falling back to rule-based synthesis.")

    return _fallback_rule_based_synthesis(query, plan, tool_results, image1)


def _fallback_rule_based_synthesis(
    query: str,
    plan: ExecutionPlan,
    tool_results: list[AnalysisResult],
    image1: RasterImage,
) -> StructuredSynthesis:
    """Combine answers deterministically into a rich, structured output when LLM is unavailable."""
    if not tool_results:
        return StructuredSynthesis(
            status="insufficient_evidence",
            intent=plan.intent,
            summary="No analysis results available.",
            answer="⚠️ No analysis results available.",
        )

    observations: list[str] = []
    evidence: list[str] = []
    sources: list[str] = []
    confidence = "Moderate"
    status = "success"

    # Default evidence from uploaded image
    evidence.append(
        f"Uploaded satellite image ({image1.width}x{image1.height} px, {image1.num_bands} bands: {', '.join(image1.bands)})"
    )

    # Collect findings across all tool results
    for res in tool_results:
        if res.tool_used:
            sources.append(res.tool_used)

        meta = res.metadata or {}

        # Water detection tool
        if "water" in res.tool_used.lower() or meta.get("method") in ("true_ndwi", "rgb_water_proxy", "missing_bands"):
            if meta.get("method") == "true_ndwi":
                cov = meta.get("coverage_percent", 0.0)
                px = meta.get("water_pixels", 0)
                total = meta.get("total_pixels", 0)
                observations.append(f"Water coverage identified across {cov:.1f}% of the scene ({px:,} of {total:,} pixels).")
                observations.append("Spectral analysis performed using McFeeters NDWI (Green and NIR bands).")
                evidence.append(f"Authoritative spectral calculation: NDWI (threshold > {meta.get('threshold', 0.0):.2f})")
                confidence = "High — derived from true multispectral Green and NIR bands"
            elif meta.get("method") == "rgb_water_proxy":
                cov = meta.get("coverage_percent", 0.0)
                observations.append(f"Estimated water-like surface area is approximately {cov:.1f}% based on RGB color proxy.")
                observations.append("Near-infrared (NIR) data is absent; calculation is a visual proxy rather than true NDWI.")
                evidence.append("RGB color heuristic proxy (heuristic approximation)")
                confidence = "Moderate — estimated via RGB proxy without near-infrared (NIR) telemetry"
            elif meta.get("method") == "missing_bands":
                status = "insufficient_evidence"
                observations.append(f"Required spectral bands missing: {', '.join(meta.get('missing_bands', []))}.")
                confidence = "Low — insufficient spectral bands for NDWI calculation"
            else:
                if res.answer:
                    observations.append(f"{res.tool_used}: {res.answer}" if res.tool_used else res.answer)

        # Vegetation detection tool
        elif "vegetation" in res.tool_used.lower() or meta.get("method") in ("true_ndvi", "rgb_greenness", "missing_bands"):
            if meta.get("method") == "true_ndvi":
                cov = meta.get("coverage_percent", 0.0)
                px = meta.get("veg_pixels", 0)
                total = meta.get("total_pixels", 0)
                observations.append(f"Vegetation canopy detected across {cov:.1f}% of the scene ({px:,} of {total:,} pixels).")
                observations.append("Spectral index calculated using Rouse NDVI (NIR and Red bands).")
                evidence.append(f"Authoritative spectral calculation: NDVI (threshold > {meta.get('threshold', 0.2):.2f})")
                confidence = "High — calculated from true multispectral Red and NIR bands"
            elif meta.get("method") == "rgb_greenness":
                cov = meta.get("coverage_percent", 0.0)
                observations.append(f"Green vegetation-like canopy estimated at {cov:.1f}% using RGB visual greenness proxy.")
                observations.append("Near-infrared (NIR) band missing; result is an RGB color proxy rather than true NDVI.")
                evidence.append("RGB visual greenness proxy")
                confidence = "Moderate — visual greenness proxy without NIR spectral data"
            elif meta.get("method") == "missing_bands":
                status = "insufficient_evidence"
                observations.append("Missing NIR band required for true NDVI calculation.")
                confidence = "Low — insufficient spectral bands"
            else:
                if res.answer:
                    observations.append(f"{res.tool_used}: {res.answer}" if res.tool_used else res.answer)

        # Built-up detection tool
        elif "built" in res.tool_used.lower() or meta.get("method") in ("true_ndbi", "rgb_urban_proxy"):
            if meta.get("method") == "true_ndbi":
                cov = meta.get("coverage_percent", 0.0)
                observations.append(f"Built-up urban structures detected across {cov:.1f}% of the area.")
                evidence.append("Authoritative spectral index: NDBI (SWIR and NIR bands)")
                confidence = "High — calculated from true SWIR and NIR bands"
            elif meta.get("method") == "rgb_urban_proxy":
                cov = meta.get("coverage_percent", 0.0)
                observations.append(f"Urban and impervious structures estimated at {cov:.1f}% via RGB proxy.")
                evidence.append("RGB urban proxy")
                confidence = "Moderate — RGB proxy without SWIR/NIR bands"
            else:
                if res.answer:
                    observations.append(f"{res.tool_used}: {res.answer}" if res.tool_used else res.answer)

        # Change detection / ChangeFormer
        elif "change" in res.tool_used.lower():
            chg = meta.get("changed_percent", meta.get("change_percentage", 0.0))
            if meta.get("reason") == "missing_second_image":
                status = "insufficient_evidence"
                observations.append("Change detection requires two temporal satellite images (before and after).")
                confidence = "None — second image missing"
            else:
                observations.append(f"Bi-temporal change detected across {chg:.2f}% of the monitored region.")
                evidence.append("ChangeFormer deep Transformer bi-temporal change detection")
                if "alignment_method" in meta:
                    evidence.append(f"Pair alignment method: {meta['alignment_method']}")
                confidence = "High — ChangeFormer Transformer architecture"

        # Multi-modal Optical + SAR (BIFOLD)
        elif "bifold" in res.tool_used.lower() or "optical_sar" in res.tool_used.lower():
            observations.append("Multi-modal Optical and Synthetic Aperture Radar (SAR) fusion analyzed.")
            evidence.append("BIFOLD RDNet Base 12-channel Sentinel-1 + Sentinel-2 classification")
            confidence = "High — 12-band multi-modal Optical+SAR model"

        # GeoChat / VLM visual reasoning
        elif "geochat" in res.tool_used.lower() or "vlm" in res.tool_used.lower():
            observations.append(res.answer.strip())
            evidence.append("GeoChat-7B remote GPU visual reasoning")
            if confidence == "Moderate":
                confidence = "Moderate — visual inspection from GeoChat-7B"

        # Other / Generic
        else:
            if res.answer and res.answer not in observations:
                observations.append(res.answer.strip())

    # Build primary summary
    if len(tool_results) == 1:
        summary = tool_results[0].answer
        answer_text = tool_results[0].answer
    elif observations:
        summary = observations[0]
        answer_text = "\n\n".join(observations)
    else:
        summary = "Analysis completed for the provided satellite imagery."
        answer_text = summary

    # Remove duplicates preserving order
    seen_obs = set()
    dedup_obs = []
    for obs in observations:
        if obs not in seen_obs and obs.strip():
            seen_obs.add(obs)
            dedup_obs.append(obs)

    seen_ev = set()
    dedup_ev = []
    for ev in evidence:
        if ev not in seen_ev and ev.strip():
            seen_ev.add(ev)
            dedup_ev.append(ev)

    synth = StructuredSynthesis(
        status=status,
        intent=plan.intent,
        summary=summary,
        observations=dedup_obs,
        evidence=dedup_ev,
        confidence=confidence,
        sources=list(dict.fromkeys(sources)),
        answer=answer_text,
    )
    if not synth.answer:
        synth.answer = synth.format_markdown()
    return synth
