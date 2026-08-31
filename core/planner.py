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


# Keyword rules: (keywords, intent)
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
    # General description
    (
        ["describe", "what is", "what do you see", "explain", "summary",
         "overview", "tell me about", "show me", "identify", "object"],
        Intent.IMAGE_DESCRIPTION,
    ),
]


def plan_query(query: str, has_second_image: bool = False) -> PlanResult:
    """
    Route a user query to the appropriate intent.

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
        logger.info(f"No routing match for query: '{query}'")
        return PlanResult(
            intent=Intent.UNSUPPORTED,
            confidence=1.0,
            reasoning=f"No matching tool for query: '{query}'.",
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
