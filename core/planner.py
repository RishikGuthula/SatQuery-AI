"""
Query planner / router.

Maps a user query string to an Intent, selecting the appropriate tool.
Currently uses deterministic keyword matching; designed to be replaced
by an LLM/VLM-based planner without changing the rest of the architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.models import Intent

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    """Output of the query planner."""
    intent: Intent
    confidence: float  # Planner confidence in the routing, not analysis confidence
    reasoning: str


# Out-of-scope indicators (general knowledge, coding, weather, entertainment, chitchat)
_OUT_OF_SCOPE_PATTERNS = [
    "who is", "who was", "president", "prime minister", "capital of",
    "write a python", "write me a python", "write python", "write code", "write a program", "write a script",
    "weather today", "weather tomorrow", "weather forecast", "temperature today", "will it rain today",
    "tell me a joke", "tell a joke", "make me laugh", "write a poem", "write a story", "sing a song",
    "how to bake", "recipe for", "how to cook", "translate to", "translate into",
    "what is the capital", "who won", "how are you", "who are you", "what is your name",
]

# In-domain remote sensing keywords
_GEOSPATIAL_DOMAIN_TERMS = [
    "satellite", "remote sensing", "earth observation", "imagery", "image", "scene",
    "water", "river", "lake", "ocean", "flood", "pond", "reservoir", "coast", "canal", "stream",
    "vegetation", "forest", "tree", "crop", "agriculture", "plant", "canopy", "green cover",
    "built", "urban", "city", "building", "structure", "construction", "impervious", "road",
    "change", "difference", "compare", "temporal", "deforestation", "urbanization",
    "ndvi", "ndwi", "ndbi", "optical", "sar", "radar", "sentinel", "landsat", "geotiff",
    "land cover", "land use", "terrain", "geography", "bifold", "changeformer",
]

# Specific Keyword rules: (keywords, intent)
_ROUTING_RULES: list[tuple[list[str], Intent]] = [
    # Change detection (bi-temporal change between two images)
    (
        ["change", "difference", "comparison", "compare", "before", "after",
         "temporal", "land use change", "deforestation", "urbanization",
         "detect change", "changes between", "what changed", "show areas that changed",
         "area change", "changeformer"],
        Intent.CHANGE_DETECTION,
    ),
    # Water detection
    (
        ["water", "river", "lake", "ocean", "flood", "pond", "reservoir",
         "ndwi", "water bod", "water body", "water detection", "water area"],
        Intent.WATER_DETECTION,
    ),
    # Vegetation detection
    (
        ["vegetation", "forest", "tree", "green cover", "agriculture",
         "crop", "plant", "ndvi", "vegetation detection", "vegetation area",
         "vegetation index"],
        Intent.VEGETATION_DETECTION,
    ),
    # Built-up detection
    (
        ["built", "building", "urban", "city", "construction", "ndbi",
         "built-up", "built up", "impervious", "structure"],
        Intent.BUILTUP_DETECTION,
    ),
    # Multi-modal Optical + SAR / Land-cover classification (BIFOLD RDNet)
    (
        [
            "optical and sar", "optical + sar", "sentinel-1 and sentinel-2",
            "sentinel-1 + sentinel-2", "s1 and s2", "s1 + s2", "bifold",
            "rdnet", "sar analysis", "sar imagery", "land-cover classification",
            "land cover classification", "classify land cover", "radar and optical",
            "sar and optical", "bigearthnet",
        ],
        Intent.OPTICAL_SAR_ANALYSIS,
    ),
    # General description of image / scene / visual features
    (
        [
            "describe this image", "describe the image", "describe image",
            "describe this scene", "describe the scene", "describe scene",
            "describe", "what do you see", "what is in this image",
            "what is in this scene", "what is in this photo", "what do you see in this photo",
            "what type of land use is visible", "what is visible", "explain this image",
            "explain the scene", "image overview", "scene overview", "summary of the image",
            "tell me about this image", "tell me about this scene",
            "show me features", "identify features", "identify objects", "identify",
            "visual overview", "scene analysis", "overview", "photo",
        ],
        Intent.IMAGE_DESCRIPTION,
    ),

]


def is_out_of_scope_query(query: str) -> bool:
    """Check if query is clearly outside the scope of satellite and remote sensing analysis."""
    q_lower = query.strip().lower()
    if not q_lower:
        return False

    # Direct match on known out-of-scope patterns
    for pat in _OUT_OF_SCOPE_PATTERNS:
        if pat in q_lower:
            return True

    return False


def is_geospatial_relevant(query: str) -> bool:
    """Check if query contains any relevance to satellite, Earth observation, or geospatial analysis."""
    q_lower = query.strip().lower()
    return any(term in q_lower for term in _GEOSPATIAL_DOMAIN_TERMS)


def plan_query(query: str, has_second_image: bool = False) -> PlanResult:
    """
    Route a user query to the appropriate intent with domain validation.

    Args:
        query: The user's natural language query.
        has_second_image: Whether a second image was provided.

    Returns:
        PlanResult with the matched intent.
    """
    query_lower = query.strip().lower()

    if not query_lower:
        return PlanResult(
            intent=Intent.UNSUPPORTED,
            confidence=1.0,
            reasoning="Empty query.",
        )

    # 1. Check for explicit out-of-scope queries
    if is_out_of_scope_query(query_lower):
        logger.info(f"Out-of-scope query detected: '{query}'")
        return PlanResult(
            intent=Intent.OUT_OF_SCOPE,
            confidence=1.0,
            reasoning="Query is unrelated to satellite imagery and remote sensing.",
        )

    # 2. Check rule-based intent matching
    best_intent = Intent.UNSUPPORTED
    best_score = 0.0
    best_rule = ""

    for keywords, intent in _ROUTING_RULES:
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > best_score:
            best_score = score
            best_intent = intent
            best_rule = ", ".join(kw for kw in keywords if kw in query_lower)

    if best_score == 0:
        # Check if the query mentioned satellite/geospatial domain but no specific tool matched
        if is_geospatial_relevant(query_lower):
            logger.info(f"Geospatial query with no specific tool match: '{query}'")
            return PlanResult(
                intent=Intent.UNSUPPORTED,
                confidence=1.0,
                reasoning=f"Geospatial query without matching tool: '{query}'.",
            )
        else:
            logger.info(f"Unrelated query classified as out-of-scope: '{query}'")
            return PlanResult(
                intent=Intent.OUT_OF_SCOPE,
                confidence=1.0,
                reasoning=f"Query '{query}' is unrelated to satellite imagery analysis.",
            )

    # Change detection requires two images
    if best_intent == Intent.CHANGE_DETECTION and not has_second_image:
        return PlanResult(
            intent=Intent.CHANGE_DETECTION,
            confidence=1.0,
            reasoning="Change detection selected, but only one image provided. "
                      "A second image is required.",
        )

    confidence = min(best_score / 3.0, 1.0)  # Normalize
    logger.info(
        f"Planned intent={best_intent.value} (score={best_score}, "
        f"matched='{best_rule}') for query='{query}'"
    )

    return PlanResult(
        intent=best_intent,
        confidence=confidence,
        reasoning=f"Matched keywords: {best_rule}",
    )

