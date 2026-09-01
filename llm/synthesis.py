"""
LLM Synthesis Engine with Structured Grounded Output.

Synthesizes outputs from multiple scientific tools and GeoChat-7B visual
reasoning into a single coherent, authoritative, grounded answer.
Strictly guards against numerical hallucination by injecting ground-truth metrics.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from pydantic import BaseModel, Field

from core.models import AnalysisResult, RasterImage
from llm.base import LLMMessage
from llm.client import get_llm_client, LLMClient
from llm.planner import ExecutionPlan

logger = logging.getLogger(__name__)


class StructuredSynthesis(BaseModel):
    """Structured synthesis model ensuring strict grounding, broad explanations, and scannable output."""
    status: str = Field(default="success", description="'success' | 'out_of_scope' | 'insufficient_evidence'")
    intent: str = Field(default="", description="Identified query intent")
    summary: str = Field(default="", description="Scene Overview and direct answer to the user's question")
    observations: list[str] = Field(default_factory=list, description="Key factual observations directly observed")
    key_visual_features: list[str] = Field(default_factory=list, description="Identified visual elements (e.g. vegetation, water, buildings, roads, agricultural plots, terrain)")
    interpretation: str = Field(default="", description="Domain interpretation and what the observations suggest")
    evidence: list[str] = Field(default_factory=list, description="Explicit evidence sources (e.g. GeoChat visual reasoning, uploaded image dimensions, spectral calculations)")
    confidence: str = Field(default="", description="Confidence level with honest factual justification")
    limitations: list[str] = Field(default_factory=list, description="Scientific limitations, optical proxy disclosures, or atmospheric constraints")
    sources: list[str] = Field(default_factory=list, description="Authoritative tools or algorithms used")
    answer: str = Field(default="", description="Full formatted markdown text")

    def __str__(self) -> str:
        return self.answer or self.summary

    def __contains__(self, item: str) -> bool:
        combined = f"{self.summary} {self.answer} {' '.join(self.observations)} {' '.join(self.key_visual_features)} {self.interpretation} {' '.join(self.sources)} {' '.join(self.evidence)}"
        return item in combined

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.answer == other or self.summary == other or str(self) == other
        return super().__eq__(other)

    def format_markdown(self) -> str:
        """Format the structured synthesis into clean, professional, scannable Markdown."""
        parts = []

        if self.summary:
            parts.append(f"### 🌐 Scene Overview\n{self.summary}")

        if self.observations:
            obs_lines = "\n".join(f"• {obs}" for obs in self.observations if obs.strip())
            if obs_lines:
                parts.append(f"### 🔍 Key Observations\n{obs_lines}")

        if self.key_visual_features:
            feat_lines = "\n".join(f"• {feat}" for feat in self.key_visual_features if feat.strip())
            if feat_lines:
                parts.append(f"### 🗺️ Visual Features & Land Cover\n{feat_lines}")

        if self.interpretation:
            parts.append(f"### 💡 Interpretation & Analysis\n{self.interpretation}")

        if self.evidence:
            ev_lines = "\n".join(f"✓ {ev}" for ev in self.evidence if ev.strip())
            if ev_lines:
                parts.append(f"### 📋 Evidence Sources\n{ev_lines}")

        if self.confidence:
            parts.append(f"### 🎯 Confidence Rating\n{self.confidence}")

        if self.limitations:
            lim_lines = "\n".join(f"⚠️ {lim}" for lim in self.limitations if lim.strip())
            if lim_lines:
                parts.append(f"### ⚠️ Limitations & Disclosures\n{lim_lines}")

        if not parts:
            return self.answer or "No analysis results available."

        return "\n\n".join(parts)


_SYNTHESIS_SYSTEM_PROMPT = """You are SatQuery AI, an expert multimodal remote-sensing and Earth observation assistant.
Your job is to synthesize raw tool outputs, GeoChat visual reasoning, and image metadata into ONE comprehensive, structured, human-readable satellite explanation.

CRITICAL SYNTHESIS & GROUNDING RULES:
1. GEOCHAT-7B IS OBSERVATIONAL VISUAL EVIDENCE:
   - Treat GeoChat output as observational evidence, NOT as unquestionable truth.
   - Do NOT simply repeat or copy GeoChat's response word-for-word.
   - Translate GeoChat's observations into a broad, articulate, natural-language explanation of what is in the scene.
   - For image-description and analytical queries (e.g. "Describe this scene", "What is in this image?", "Is this urban?"), explain the visual features clearly across multiple distinct sections rather than just a single sentence.
   - Separate:
     * Scene Overview (summary)
     * Key Observations (observations)
     * Identified Visual Features (key_visual_features: vegetation, water bodies, buildings, roads, agricultural patterns, bare soil, terrain, clouds, etc.)
     * Interpretation / What It Suggests (interpretation: land-use context, environmental state, urban development, agricultural activity)
     * Evidence (evidence: tools used, image properties, visual reasoning)
     * Confidence (confidence: clear factual justification)
     * Limitations (limitations: optical constraints, proxy disclosures, resolution factors)
2. MULTIMODAL GROUNDING & NO HALLUCINATIONS:
   - ALL quantitative metrics (percentages, pixel counts, thresholds, NDWI/NDVI/NDBI values) MUST match the Authoritative Scientific Tools exactly.
   - NEVER invent or estimate new numbers, GPS coordinates, satellite names (e.g. Sentinel-2, Landsat), or acquisition dates unless provided in image metadata.
   - When both spectral tools and GeoChat exist:
     * Spectral tools = quantitative scientific metrics & ground truth.
     * GeoChat = descriptive visual context and spatial distribution.
     Combine them harmoniously.
3. HONEST REPORTING & UNCERTAINTY:
   - If GeoChat's output is brief, ambiguous, low-confidence, or indicates blurry/cloudy imagery, explicitly report that visual features are uncertain.
   - If a tool used an RGB color proxy instead of multispectral data, state that true spectral calculation requires NIR/SWIR bands.
   - If evidence is insufficient (e.g. missing second image for change detection), state what is missing.
4. OUT-OF-SCOPE QUERIES:
   - If the user query is unrelated to Earth observation, satellite imagery, or geospatial analysis, return status "out_of_scope" with intent "out_of_scope" and empty observation lists.

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "status": "success" | "out_of_scope" | "insufficient_evidence",
  "intent": "intent_name",
  "summary": "Clear, comprehensive overview of the scene and direct answer to the user query.",
  "observations": [
    "Factual observation 1",
    "Factual observation 2"
  ],
  "key_visual_features": [
    "Vegetation: description of canopy or crop presence",
    "Water Bodies: description of rivers, lakes, or coastal features",
    "Structures & Roads: description of urban/built-up patterns"
  ],
  "interpretation": "Detailed remote-sensing interpretation of what the observed features indicate.",
  "evidence": [
    "Uploaded satellite image (dimensions, bands, sensor)",
    "GeoChat-7B remote GPU visual reasoning",
    "Authoritative spectral calculation or model name"
  ],
  "confidence": "High / Moderate / Low with factual justification",
  "limitations": [
    "Scientific limitation or disclosure 1",
    "Scientific limitation or disclosure 2"
  ],
  "sources": [
    "Tool or algorithm name"
  ]
}
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

Please generate the final comprehensive synthesized JSON output following the grounding rules.
"""

    messages = [
        LLMMessage(role="system", content=_SYNTHESIS_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]

    try:
        response = client.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=1200,
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


def _extract_visual_features_from_text(text: str) -> list[str]:
    """Identify and categorize visible remote-sensing features mentioned in text."""
    features: list[str] = []
    lower = text.lower()

    if any(k in lower for k in ("river", "lake", "water", "ocean", "sea", "canal", "stream", "reservoir", "coastline", "bay", "pond")):
        # Extract specific water description
        if "river" in lower:
            features.append("Water Bodies: River channel / fluvial corridor")
        elif "lake" in lower or "reservoir" in lower or "pond" in lower:
            features.append("Water Bodies: Inland water reservoir / lake")
        elif "ocean" in lower or "sea" in lower or "coast" in lower or "bay" in lower:
            features.append("Water Bodies: Coastal / marine shoreline")
        else:
            features.append("Water Bodies: Surface water features")

    if any(k in lower for k in ("forest", "tree", "vegetation", "green", "agriculture", "crop", "field", "farm", "grass", "canopy", "plant")):
        if "agriculture" in lower or "crop" in lower or "field" in lower or "farm" in lower:
            features.append("Vegetation & Agriculture: Cultivated farmland / crop plots")
        elif "forest" in lower or "dense" in lower or "tree" in lower:
            features.append("Vegetation: Dense forest / tree canopy")
        else:
            features.append("Vegetation: Natural green canopy / sparse ground cover")

    if any(k in lower for k in ("building", "urban", "city", "town", "structure", "road", "highway", "residential", "industrial", "house", "settlement")):
        if "road" in lower or "highway" in lower:
            features.append("Infrastructure: Transportation network / roadways")
        if "building" in lower or "urban" in lower or "city" in lower or "structure" in lower:
            features.append("Built-up Environment: Urban structures / developed settlements")

    if any(k in lower for k in ("mountain", "hill", "desert", "sand", "bare", "soil", "rock", "valley", "terrain")):
        if "desert" in lower or "sand" in lower:
            features.append("Terrain: Arid desert / sandy terrain")
        elif "mountain" in lower or "hill" in lower:
            features.append("Topography: Elevated mountain / hilly relief")
        else:
            features.append("Surface: Bare soil / exposed rocky terrain")

    if any(k in lower for k in ("cloud", "haze", "fog", "shadow")):
        features.append("Atmospheric Conditions: Cloud cover or atmospheric haze present")

    return features


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
    key_visual_features: list[str] = []
    interpretation_parts: list[str] = []
    evidence: list[str] = []
    limitations: list[str] = []
    sources: list[str] = []
    confidence = "Moderate"
    status = "success"
    geochat_text = ""
    is_uncertain_geochat = False

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
                key_visual_features.append(f"Water: Surface water bodies covering {cov:.1f}% of analyzed scene")
                interpretation_parts.append(f"The scene exhibits {cov:.1f}% surface water presence based on Green and NIR spectral reflectance.")
                confidence = "High — derived from true multispectral Green and NIR bands"
            elif meta.get("method") == "rgb_water_proxy":
                cov = meta.get("coverage_percent", 0.0)
                observations.append(f"Estimated water-like surface area is approximately {cov:.1f}% based on RGB color proxy.")
                observations.append("Near-infrared (NIR) data is absent; calculation is a visual proxy rather than true NDWI.")
                evidence.append("RGB color heuristic proxy (heuristic approximation)")
                key_visual_features.append(f"Water-like Features: Approximately {cov:.1f}% RGB color proxy coverage")
                interpretation_parts.append(f"Visual color proxy suggests approximately {cov:.1f}% water-like dark/cyan surface tone.")
                limitations.append("Near-infrared (NIR) band absent; water calculation is an RGB color proxy rather than true NDWI.")
                confidence = "Moderate — estimated via RGB proxy without near-infrared (NIR) telemetry"
            elif meta.get("method") == "missing_bands":
                status = "insufficient_evidence"
                observations.append(f"Required spectral bands missing: {', '.join(meta.get('missing_bands', []))}.")
                limitations.append("Insufficient spectral bands for authoritative index computation.")
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
                key_visual_features.append(f"Vegetation Canopy: Dense/moderate foliage covering {cov:.1f}% of scene")
                interpretation_parts.append(f"Vegetation health and density is robust with {cov:.1f}% active photosynthesizing canopy.")
                confidence = "High — calculated from true multispectral Red and NIR bands"
            elif meta.get("method") == "rgb_greenness":
                cov = meta.get("coverage_percent", 0.0)
                observations.append(f"Green vegetation-like canopy estimated at {cov:.1f}% using RGB visual greenness proxy.")
                observations.append("Near-infrared (NIR) band missing; result is an RGB color proxy rather than true NDVI.")
                evidence.append("RGB visual greenness proxy")
                key_visual_features.append(f"Greenness Proxy: Estimated {cov:.1f}% green surface tone")
                interpretation_parts.append(f"RGB color analysis highlights approximately {cov:.1f}% green surface area.")
                limitations.append("Near-infrared (NIR) band missing; result is an RGB color proxy rather than true NDVI.")
                confidence = "Moderate — visual greenness proxy without NIR spectral data"
            elif meta.get("method") == "missing_bands":
                status = "insufficient_evidence"
                observations.append("Missing NIR band required for true NDVI calculation.")
                limitations.append("Near-infrared (NIR) band required for true NDVI.")
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
                key_visual_features.append(f"Built-up Structures: Developed urban area covering {cov:.1f}% of scene")
                interpretation_parts.append(f"Built-up and impervious structures comprise {cov:.1f}% of the monitored scene.")
                confidence = "High — calculated from true SWIR and NIR bands"
            elif meta.get("method") == "rgb_urban_proxy":
                cov = meta.get("coverage_percent", 0.0)
                observations.append(f"Urban and impervious structures estimated at {cov:.1f}% via RGB proxy.")
                evidence.append("RGB urban proxy")
                key_visual_features.append(f"Urban Proxy: Estimated {cov:.1f}% built-up structural tone")
                interpretation_parts.append(f"Visual textural analysis estimates approximately {cov:.1f}% structural coverage.")
                limitations.append("SWIR/NIR bands missing; urban estimate is an RGB proxy rather than true NDBI.")
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
                limitations.append("Secondary (after) satellite image was not provided.")
                confidence = "None — second image missing"
            else:
                observations.append(f"Bi-temporal change detected across {chg:.2f}% of the monitored region.")
                evidence.append("ChangeFormer deep Transformer bi-temporal change detection")
                if meta.get("alignment_applied"):
                    evidence.append(f"Pair spatial alignment: {meta.get('aligned_from')} → {meta.get('aligned_to')} px")
                    observations.append(
                        f"Secondary image automatically aligned from {meta.get('aligned_from')} to {meta.get('aligned_to')} px for bi-temporal comparison."
                    )
                elif "alignment_method" in meta:
                    evidence.append(f"Pair alignment method: {meta['alignment_method']}")
                key_visual_features.append(f"Temporal Change: {chg:.2f}% surface modification between bi-temporal acquisitions")
                interpretation_parts.append(f"Bi-temporal analysis confirms structural or land-cover modification across {chg:.2f}% of the scene.")
                confidence = "High — ChangeFormer Transformer architecture"

        # Multi-modal Optical + SAR (BIFOLD)
        elif "bifold" in res.tool_used.lower() or "optical_sar" in res.tool_used.lower():
            observations.append("Multi-modal Optical and Synthetic Aperture Radar (SAR) fusion analyzed.")
            evidence.append("BIFOLD RDNet Base 12-channel Sentinel-1 + Sentinel-2 classification")
            key_visual_features.append("Multi-modal Fusion: 12-channel Sentinel-1 (SAR) and Sentinel-2 (Optical) land cover")
            interpretation_parts.append("Cross-sensor fusion leverages SAR polarimetric backscatter alongside optical bands for robust surface classification.")
            confidence = "High — 12-band multi-modal Optical+SAR model"

        # GeoChat / VLM visual reasoning
        elif "geochat" in res.tool_used.lower() or "vlm" in res.tool_used.lower() or "image_description" in res.tool_used.lower():
            raw_text = res.answer.strip() if res.answer else ""
            geochat_text = raw_text
            evidence.append("GeoChat-7B remote GPU visual reasoning")

            # Check for uncertainty in GeoChat output
            if any(w in raw_text.lower() for w in ("blurry", "low quality", "unclear", "low-contrast", "difficult", "uncertain", "not clear")):
                is_uncertain_geochat = True
                observations.append(f"GeoChat-7B visual inspection noted visual constraint: {raw_text}")
                limitations.append("Visual features are constrained by image contrast, blur, or atmospheric conditions.")
                confidence = "Low — visual inspection noted observational ambiguity"
            else:
                observations.append(f"GeoChat-7B visual inspection: {raw_text}")
                if confidence == "Moderate":
                    confidence = "Moderate — visual inspection from GeoChat-7B"

            extracted_features = _extract_visual_features_from_text(raw_text)
            for feat in extracted_features:
                if feat not in key_visual_features:
                    key_visual_features.append(feat)

            if not interpretation_parts and raw_text:
                interpretation_parts.append(
                    f"Visual features suggest a complex landscape with visible spatial patterns as noted by GeoChat-7B."
                )

        # Other / Generic
        else:
            if res.answer and res.answer not in observations:
                observations.append(res.answer.strip())

    # Build primary scene overview (summary)
    if geochat_text and not is_uncertain_geochat:
        # Expand GeoChat summary cleanly
        clean_text = geochat_text.rstrip(".")
        if clean_text.lower().startswith("the image shows ") or clean_text.lower().startswith("the image depicts "):
            summary = f"The satellite image captures {clean_text[16:]}."
        elif clean_text.lower().startswith("this is ") or clean_text.lower().startswith("it is "):
            summary = f"The satellite scene shows {clean_text[8:]}."
        else:
            summary = f"The analyzed satellite imagery reveals {clean_text}."
    elif len(tool_results) == 1 and tool_results[0].answer:
        summary = tool_results[0].answer
    elif observations:
        summary = f"Satellite image analysis completed with {len(observations)} key findings across the scene."
    else:
        summary = "Analysis completed for the provided satellite imagery."

    # Standard remote sensing optical disclosures
    if not limitations:
        limitations.append("Optical visual interpretation; fine-scale biochemical or sub-surface properties require multispectral NIR/SWIR bands.")
        limitations.append("Sensor resolution and atmospheric conditions may influence feature boundary precision.")

    # Deduplicate observations and evidence preserving order
    dedup_obs = []
    seen_obs = set()
    for obs in observations:
        if obs not in seen_obs and obs.strip():
            seen_obs.add(obs)
            dedup_obs.append(obs)

    dedup_ev = []
    seen_ev = set()
    for ev in evidence:
        if ev not in seen_ev and ev.strip():
            seen_ev.add(ev)
            dedup_ev.append(ev)

    dedup_feat = []
    seen_feat = set()
    for feat in key_visual_features:
        if feat not in seen_feat and feat.strip():
            seen_feat.add(feat)
            dedup_feat.append(feat)

    interp_text = " ".join(interpretation_parts) if interpretation_parts else "Land-cover features identified across the scene represent key surface characteristics of the monitored region."

    synth = StructuredSynthesis(
        status=status,
        intent=plan.intent,
        summary=summary,
        observations=dedup_obs,
        key_visual_features=dedup_feat,
        interpretation=interp_text,
        evidence=dedup_ev,
        confidence=confidence,
        limitations=limitations,
        sources=list(dict.fromkeys(sources)),
    )
    synth.answer = synth.format_markdown()
    return synth
