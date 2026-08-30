"""
LLM Structured Query Planner with Pydantic Schema Validation.

Takes a natural language query and image metadata, producing a structured
execution plan composed strictly of valid registered capabilities.
Falls back safely to the deterministic planner if LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from pydantic import BaseModel, Field

from core.models import RasterImage, Intent
from core.planner import plan_query as deterministic_plan_query
from core.registry import get_registry
from llm.base import LLMMessage
from llm.client import get_llm_client, LLMClient

logger = logging.getLogger(__name__)


class TaskItem(BaseModel):
    """A single atomic capability task within the plan."""
    capability: str = Field(description="Name of registered capability to execute")
    reason: str = Field(description="Why this capability is selected based on query and data")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Optional tool parameters")


class ExecutionPlan(BaseModel):
    """Structured execution plan produced by the planner."""
    intent: str = Field(description="Overall intent category")
    reasoning: str = Field(description="Detailed plan rationale")
    tasks: list[TaskItem] = Field(description="Ordered list of capability tasks to run")
    synthesis_required: bool = Field(default=True, description="Whether LLM synthesis is needed")
    planner_used: str = Field(default="llm", description="'llm' or 'deterministic'")


_SYSTEM_PLANNER_PROMPT = """You are the Lead Planning Agent for SatQuery AI, an expert remote-sensing assistant.
Your job is to analyze the user's satellite query and available image bands to generate a strict JSON execution plan.

REGISTERED CAPABILITIES:
{capabilities_prompt}

RULES:
1. ONLY select capability names from the list above. DO NOT invent arbitrary tool names.
2. Multispectral vs RGB rules:
   - For NDVI: Requires NIR and Red bands. If image is RGB only, use 'vegetation_detection' which provides an RGB greenness proxy with honest disclosures.
   - For NDWI: Requires Green and NIR bands. If image is RGB only, use 'water_detection' (RGB water proxy).
   - For NDBI: Requires SWIR and NIR bands. If image is RGB only, use 'builtup_detection' (RGB urban proxy).
3. Visual Reasoning:
   - If user asks for general description, scene analysis, feature identification, or explanation alongside a tool, include 'geochat' if available.
4. Change Detection:
   - If user asks to compare/change detection between two images, use 'change_detection'.
5. Multi-feature queries:
   - If user asks for multiple features (e.g., "find water and vegetation and explain the scene"), decompose into multiple tasks: ['water_detection', 'vegetation_detection', 'geochat'].

OUTPUT FORMAT (JSON ONLY):
{{
  "intent": "water_detection" | "vegetation_detection" | "builtup_detection" | "change_detection" | "image_description" | "multi_feature_analysis" | "unsupported",
  "reasoning": "Brief explanation of plan",
  "tasks": [
    {{
      "capability": "capability_name",
      "reason": "Why this capability is needed",
      "parameters": {{}}
    }}
  ],
  "synthesis_required": true
}}
"""


def plan_with_llm(
    query: str,
    image1: RasterImage,
    image2: RasterImage | None = None,
    session_context: Any = None,
    llm_client: LLMClient | None = None,
) -> ExecutionPlan:
    """
    Generate an execution plan using the Online LLM, with automatic fallback
    to the deterministic planner if the LLM is offline or unconfigured.
    """
    client = llm_client or get_llm_client()
    registry = get_registry()

    # If LLM is not available, immediately use deterministic fallback
    if not client.is_available:
        logger.info("LLM is unavailable. Using deterministic planner fallback.")
        return _fallback_to_deterministic(query, image1, image2)

    # Build contextual metadata prompt
    has_second = image2 is not None
    img1_info = (
        f"Primary Image: {image1.width}x{image1.height} px, {image1.num_bands} bands: {image1.bands}, "
        f"sensor: {image1.sensor_type.value}"
    )
    img2_info = (
        f"Secondary Image: {image2.width}x{image2.height} px, {image2.num_bands} bands: {image2.bands}"
        if has_second and image2
        else "No secondary image provided."
    )

    capabilities_doc = registry.get_capabilities_prompt()
    system_prompt = _SYSTEM_PLANNER_PROMPT.format(capabilities_prompt=capabilities_doc)

    user_content = f"""USER QUERY: "{query}"

IMAGE METADATA:
- {img1_info}
- {img2_info}
"""

    if session_context and getattr(session_context, "history", None):
        history_summary = [f"Prev Q: {t.query} -> Tool: {t.tool_used}" for t in session_context.history[-3:]]
        user_content += f"\nCONVERSATION CONTEXT:\n" + "\n".join(history_summary)

    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_content),
    ]

    try:
        response = client.generate(
            messages=messages,
            temperature=0.1,
            max_tokens=600,
            response_format="json",
        )

        if not response or not response.content.strip():
            logger.warning("Empty LLM response received. Falling back to deterministic plan.")
            return _fallback_to_deterministic(query, image1, image2)

        data = json.loads(response.content)
        plan = ExecutionPlan.model_validate(data)

        # Validate that all planned tasks exist in registry
        valid_tasks = []
        for task in plan.tasks:
            cap_name = task.capability.lower().strip()
            if registry.is_valid_capability(cap_name):
                valid_tasks.append(task)
            else:
                logger.warning(f"Planner proposed unregistered capability '{task.capability}'. Discarding.")

        if not valid_tasks:
            logger.warning("No valid capabilities in LLM plan. Using deterministic fallback.")
            return _fallback_to_deterministic(query, image1, image2)

        plan.tasks = valid_tasks
        plan.planner_used = "llm"
        logger.info(f"LLM Plan generated successfully with {len(plan.tasks)} task(s). Intent: {plan.intent}")
        return plan

    except Exception as e:
        logger.warning(f"LLM planning failed with error: {e}. Falling back to deterministic planner.")
        return _fallback_to_deterministic(query, image1, image2)


def _fallback_to_deterministic(
    query: str,
    image1: RasterImage,
    image2: RasterImage | None,
) -> ExecutionPlan:
    """Create a structured ExecutionPlan from the deterministic router."""
    plan_res = deterministic_plan_query(query, has_second_image=image2 is not None)
    intent = plan_res.intent

    tasks: list[TaskItem] = []

    if intent == Intent.WATER_DETECTION:
        tasks.append(TaskItem(capability="water_detection", reason="Deterministic keyword match for water"))
    elif intent == Intent.VEGETATION_DETECTION:
        tasks.append(TaskItem(capability="vegetation_detection", reason="Deterministic keyword match for vegetation"))
    elif intent == Intent.BUILTUP_DETECTION:
        tasks.append(TaskItem(capability="builtup_detection", reason="Deterministic keyword match for built-up"))
    elif intent == Intent.CHANGE_DETECTION:
        tasks.append(TaskItem(capability="change_detection", reason="Deterministic keyword match for change detection"))
    elif intent == Intent.IMAGE_DESCRIPTION:
        # Prefer GeoChat if registered & available, else image_description
        registry = get_registry()
        geochat_cap = registry.get("geochat")
        if geochat_cap and geochat_cap.is_available():
            tasks.append(TaskItem(capability="geochat", reason="Visual reasoning for image description"))
        else:
            tasks.append(TaskItem(capability="image_description", reason="Rule-based baseline description"))
    elif intent == Intent.UNSUPPORTED:
        tasks.append(TaskItem(capability="unsupported", reason=plan_res.reasoning))

    return ExecutionPlan(
        intent=intent.value,
        reasoning=plan_res.reasoning,
        tasks=tasks,
        synthesis_required=False,
        planner_used="deterministic",
    )
