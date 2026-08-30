"""
Agent controller.

Orchestrates the full pipeline:
    1. Load / validate images
    2. Plan the query (route to intent)
    3. Execute the appropriate tool
    4. Generate evidence
    5. Return structured result
"""

from __future__ import annotations

import logging
import time

from core.image_loader import RasterImage, load_from_bytes, ImageLoadError
from core.models import AnalysisResult, Intent
from core.planner import plan_query, PlanResult
from tools.registry import SINGLE_IMAGE_TOOLS
from tools.change_detection import detect_changes, ChangeDetectionError
from vlm.base import get_vlm

logger = logging.getLogger(__name__)


def process_query(
    query: str,
    image1_bytes: bytes,
    image2_bytes: bytes | None = None,
    filename1: str = "image1",
    filename2: str = "image2",
) -> AnalysisResult:
    """
    Main entry point for the agent.

    Args:
        query: User's natural language query.
        image1_bytes: Raw bytes of the primary image.
        image2_bytes: Raw bytes of optional second image.
        filename1: Filename hint for the primary image.
        filename2: Filename hint for the secondary image.

    Returns:
        AnalysisResult with answer, evidence, metadata.
    """
    t_start = time.time()
    logger.info(f"Processing query: '{query}'")

    # --- Step 1: Load images ---
    try:
        raster1 = load_from_bytes(image1_bytes, filename1)
    except ImageLoadError as e:
        return AnalysisResult(
            answer=f"❌ Error loading primary image: {e}",
            evidence=None,
            tool_used="error_handler",
            metadata={"error": str(e)},
        )

    raster2 = None
    if image2_bytes:
        try:
            raster2 = load_from_bytes(image2_bytes, filename2)
        except ImageLoadError as e:
            return AnalysisResult(
                answer=f"❌ Error loading secondary image: {e}",
                evidence=None,
                tool_used="error_handler",
                metadata={"error": str(e)},
            )

    logger.info(
        f"Image 1 loaded: {raster1.width}x{raster1.height}, "
        f"{raster1.num_bands} bands, sensor={raster1.sensor_type.value}"
    )
    if raster2:
        logger.info(
            f"Image 2 loaded: {raster2.width}x{raster2.height}, "
            f"{raster2.num_bands} bands, sensor={raster2.sensor_type.value}"
        )

    # --- Step 2: Plan query ---
    plan = plan_query(query, has_second_image=raster2 is not None)
    logger.info(f"Planned intent: {plan.intent.value}")

    # --- Step 3: Execute tool ---
    try:
        result = _execute_tool(plan, raster1, raster2)
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        return AnalysisResult(
            answer=f"❌ Analysis failed: {e}",
            evidence=None,
            tool_used="error_handler",
            metadata={"error": str(e), "intent": plan.intent.value},
        )

    # --- Step 4: Enrich with VLM if available ---
    vlm = get_vlm()
    if vlm.is_available():
        enriched_answer = vlm.analyze(query, raster1, result)
        if enriched_answer and enriched_answer != result.answer:
            result.answer = enriched_answer

    # --- Step 5: Add timing metadata ---
    elapsed = time.time() - t_start
    result.metadata["processing_time_seconds"] = round(elapsed, 3)
    result.metadata["image1_dimensions"] = f"{raster1.width}x{raster1.height}"
    result.metadata["image1_sensor_type"] = raster1.sensor_type.value
    result.metadata["image1_bands"] = raster1.bands
    if raster2:
        result.metadata["image2_dimensions"] = f"{raster2.width}x{raster2.height}"
        result.metadata["image2_sensor_type"] = raster2.sensor_type.value

    logger.info(f"Analysis completed in {elapsed:.3f}s. Tool: {result.tool_used}")
    return result


def _execute_tool(
    plan: PlanResult,
    raster1: RasterImage,
    raster2: RasterImage | None,
) -> AnalysisResult:
    """Execute the tool matching the planned intent."""
    intent = plan.intent

    # --- Change detection (needs two images) ---
    if intent == Intent.CHANGE_DETECTION:
        if raster2 is None:
            return AnalysisResult(
                answer=(
                    "⚠️ Change detection requires two images.\n\n"
                    "Please upload a second image and try again.\n\n"
                    f"Your query: matched '{intent.value}' but only one image was provided."
                ),
                tool_used="change_detection (blocked)",
                metadata={"reason": "missing_second_image"},
            )
        try:
            return detect_changes(raster1, raster2)
        except ChangeDetectionError as e:
            return AnalysisResult(
                answer=f"❌ Change detection failed: {e}",
                tool_used="change_detection (error)",
                metadata={"error": str(e)},
            )

    # --- Single-image tools ---
    if intent in SINGLE_IMAGE_TOOLS:
        tool_fn = SINGLE_IMAGE_TOOLS[intent]
        return tool_fn(raster1)

    # --- Image description ---
    if intent == Intent.IMAGE_DESCRIPTION:
        vlm = get_vlm()
        answer = vlm.analyze("", raster1)
        return AnalysisResult(
            answer=answer,
            tool_used="vlm_description",
            metadata={"method": "rule_based" if not vlm.is_available() else "vlm"},
        )

    # --- Unsupported ---
    if intent == Intent.UNSUPPORTED:
        return AnalysisResult(
            answer=(
                f"⚠️ Unsupported query.\n\n"
                f"The system cannot process: '{plan.reasoning}'\n\n"
                f"**Currently supported queries:**\n"
                f"- Water detection (rivers, lakes, floods)\n"
                f"- Vegetation detection (forests, agriculture)\n"
                f"- Built-up area detection (urban, construction)\n"
                f"- Change detection (requires two images)\n"
                f"- General image description\n\n"
                f"Future versions will support more analysis types via "
                f"Vision-Language Model integration."
            ),
            tool_used="unsupported_handler",
            metadata={"reason": plan.reasoning},
        )

    # Fallback
    return AnalysisResult(
        answer=f"⚠️ No tool found for intent: {intent.value}",
        tool_used="fallback",
        metadata={"intent": intent.value},
    )
