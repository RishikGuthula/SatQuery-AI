"""
Unified SatQuery Agent Controller.

Orchestrates the single-agent pipeline:
    1. Ingest & validate images (RGB / Multispectral GeoTIFF)
    2. Retrieve / update session context
    3. Plan query with structured LLM planner (with deterministic fallback)
    4. Execute tasks via CapabilityRegistry with execution tracing
    5. Generate layered visual evidence maps
    6. Grounded LLM synthesis (authoritative scientific metrics + visual reasoning)
    7. Return unified AnalysisResult with execution trace and metadata
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.image_loader import RasterImage, load_from_bytes, ImageLoadError
from core.models import AnalysisResult, AgentTrace, SessionContext, Intent
from core.planner import plan_query as deterministic_plan_query, PlanResult
from core.registry import get_registry, ExecutionContext
from llm.planner import plan_with_llm, ExecutionPlan, TaskItem
from llm.synthesis import synthesize_results, StructuredSynthesis
from tools.registry import SINGLE_IMAGE_TOOLS
from tools.changeformer_tool import detect_changes_changeformer, ChangeFormerError
from vlm.client import get_vlm

logger = logging.getLogger(__name__)


def process_query(
    query: str,
    image1_bytes: bytes,
    image2_bytes: bytes | None = None,
    filename1: str = "image1",
    filename2: str = "image2",
    session_context: SessionContext | None = None,
) -> AnalysisResult:
    """
    Main entry point for the SatQuery unified agent.

    Args:
        query: User's natural language question.
        image1_bytes: Raw bytes of the primary image.
        image2_bytes: Raw bytes of optional second image.
        filename1: Filename hint for primary image.
        filename2: Filename hint for secondary image.
        session_context: Optional session context for multi-turn dialogue.

    Returns:
        AnalysisResult containing unified answer, visual evidence, metadata, and execution trace.
    """
    t_start = time.time()
    trace = AgentTrace()
    ctx = session_context or SessionContext()

    logger.info(f"SatQuery Agent: Processing query: '{query}'")

    # --- Step 1: Ingest & validate images ---
    try:
        t_load = time.time()
        raster1 = load_from_bytes(image1_bytes, filename1)
        trace.add_step(
            capability="image_loader",
            status="success",
            description=f"Loaded primary image ({raster1.width}x{raster1.height} px, {raster1.num_bands} bands)",
            duration=time.time() - t_load,
        )
    except ImageLoadError as e:
        trace.add_step(
            capability="image_loader",
            status="failed",
            description=f"Primary image loading failed: {e}",
            duration=time.time() - t_start,
        )
        return AnalysisResult(
            answer=f"❌ Error loading primary image: {e}",
            evidence=None,
            tool_used="error_handler",
            metadata={"error": str(e)},
            trace=trace,
            session_context=ctx,
            status="error",
            summary=f"Error loading primary image: {e}",
        )

    raster2 = None
    if image2_bytes:
        try:
            t_load2 = time.time()
            raster2 = load_from_bytes(image2_bytes, filename2)
            trace.add_step(
                capability="image_loader",
                status="success",
                description=f"Loaded secondary image ({raster2.width}x{raster2.height} px, {raster2.num_bands} bands)",
                duration=time.time() - t_load2,
            )
        except ImageLoadError as e:
            trace.add_step(
                capability="image_loader",
                status="failed",
                description=f"Secondary image loading failed: {e}",
            )
            return AnalysisResult(
                answer=f"❌ Error loading secondary image: {e}",
                evidence=None,
                tool_used="error_handler",
                metadata={"error": str(e)},
                trace=trace,
                session_context=ctx,
                status="error",
                summary=f"Error loading secondary image: {e}",
            )

    # --- Step 2: Plan Query ---
    t_plan = time.time()
    plan = plan_with_llm(
        query=query,
        image1=raster1,
        image2=raster2,
        session_context=ctx,
    )
    plan_duration = time.time() - t_plan

    trace.planner_type = plan.planner_used
    trace.planned_intent = plan.intent
    trace.plan_reasoning = plan.reasoning
    trace.add_step(
        capability=f"planner ({plan.planner_used})",
        status="success" if plan.status != "out_of_scope" else "skipped",
        description=f"Planned intent '{plan.intent}' with {len(plan.tasks)} task(s): {plan.reasoning}",
        duration=plan_duration,
    )

    # Check for early out-of-scope validation
    if plan.status == "out_of_scope" or plan.intent == "out_of_scope":
        elapsed = time.time() - t_start
        trace.total_duration_seconds = round(elapsed, 3)
        out_of_scope_msg = (
            "🛰️ **SatQuery AI is designed for satellite imagery and Earth observation analysis.**\n\n"
            "Your query appears to be unrelated to remote sensing or satellite analysis.\n\n"
            "**Supported analysis includes:**\n"
            "• Water detection & NDWI index (rivers, lakes, floods)\n"
            "• Vegetation canopy & NDVI index (agriculture, forests)\n"
            "• Built-up area detection & NDBI index (urban structures)\n"
            "• Bi-temporal change detection (with two temporal images)\n"
            "• Optical + SAR multi-modal classification (BIFOLD RDNet)\n"
            "• Natural language visual reasoning (GeoChat-7B)\n\n"
            "Please provide a question related to your satellite imagery."
        )
        return AnalysisResult(
            answer=out_of_scope_msg,
            evidence=None,
            tool_used="domain_validator",
            metadata={"query": query, "processing_time_seconds": round(elapsed, 3)},
            trace=trace,
            session_context=ctx,
            status="out_of_scope",
            intent="out_of_scope",
            summary="Query is outside the scope of satellite and Earth observation analysis.",
            observations=[],
            evidence_sources=[],
            confidence_level="Not Applicable",
            sources=[],
            structured_output={
                "status": "out_of_scope",
                "intent": "out_of_scope",
                "summary": "Query is outside the scope of satellite and Earth observation analysis.",
                "observations": [],
                "evidence": [],
                "confidence": "Not Applicable",
                "sources": [],
            },
        )

    # Check for early missing second image for change detection
    if plan.status == "insufficient_evidence" and plan.intent == "change_detection" and raster2 is None:
        elapsed = time.time() - t_start
        trace.total_duration_seconds = round(elapsed, 3)
        insufficient_msg = (
            "⚠️ **Change detection requires two images.**\n\n"
            "Please upload a second image (secondary/after image) to detect temporal land-cover changes.\n\n"
            "Your query matched bi-temporal change detection, but only one image was provided."
        )
        return AnalysisResult(
            answer=insufficient_msg,
            evidence=None,
            tool_used="change_detection (blocked)",
            metadata={"reason": "missing_second_image", "processing_time_seconds": round(elapsed, 3)},
            trace=trace,
            session_context=ctx,
            status="insufficient_evidence",
            intent="change_detection",
            summary="Change detection requires two temporal satellite images.",
            observations=["A second image is required to calculate bi-temporal changes."],
            evidence_sources=[f"Uploaded primary image ({raster1.width}x{raster1.height} px)"],
            confidence_level="None — second image missing",
            sources=["change_detection"],
            structured_output={
                "status": "insufficient_evidence",
                "intent": "change_detection",
                "summary": "Change detection requires two temporal satellite images.",
                "observations": ["A second image is required to calculate bi-temporal changes."],
                "evidence": [f"Uploaded primary image ({raster1.width}x{raster1.height} px)"],
                "confidence": "None — second image missing",
                "sources": ["change_detection"],
            },
        )

    # --- Step 3: Execute Planned Tasks ---
    registry = get_registry()
    task_results: list[AnalysisResult] = []
    primary_evidence = None
    primary_mask = None
    primary_index_name = None

    exec_context = ExecutionContext(
        image1=raster1,
        image2=raster2,
        query=query,
        session_context=ctx,
    )

    for task in plan.tasks:
        cap_name = task.capability.lower().strip()
        capability = registry.get(cap_name)

        if capability is None:
            logger.warning(f"Task capability '{cap_name}' not found in registry.")
            trace.add_step(
                capability=cap_name,
                status="failed",
                description=f"Capability '{cap_name}' is not registered.",
            )
            continue

        # Check availability
        if not capability.is_available():
            logger.info(f"Capability '{cap_name}' is currently unavailable. Logging fallback.")
            trace.add_step(
                capability=cap_name,
                status="skipped",
                description=f"Capability '{cap_name}' is offline or unconfigured. Skipped.",
            )
            continue

        # Execute capability
        t_exec = time.time()
        try:
            res = capability.execute(exec_context)
            duration = time.time() - t_exec

            task_results.append(res)
            trace.add_step(
                capability=cap_name,
                status="success",
                description=f"Executed {cap_name} ({task.reason})",
                duration=duration,
                summary=res.answer[:120] + "..." if len(res.answer) > 120 else res.answer,
            )

            # Keep track of visual evidence
            if res.evidence is not None and primary_evidence is None:
                primary_evidence = res.evidence
                primary_mask = res.mask
                primary_index_name = res.index_name

        except Exception as e:
            duration = time.time() - t_exec
            logger.error(f"Execution of '{cap_name}' failed: {e}", exc_info=True)
            trace.add_step(
                capability=cap_name,
                status="failed",
                description=f"Execution failed: {str(e)}",
                duration=duration,
            )

    # If no tasks succeeded or returned results, run fallback execution
    if not task_results:
        logger.info("No task results produced. Running fallback executor.")
        fallback_res = _execute_legacy_fallback(plan, raster1, raster2)
        task_results.append(fallback_res)
        if fallback_res.evidence is not None:
            primary_evidence = fallback_res.evidence
            primary_mask = fallback_res.mask
            primary_index_name = fallback_res.index_name

    # --- Step 4: Synthesize Final Answer ---
    t_synth = time.time()
    synth: StructuredSynthesis = synthesize_results(
        query=query,
        plan=plan,
        tool_results=task_results,
        image1=raster1,
        session_context=ctx,
    )
    synth_duration = time.time() - t_synth
    trace.add_step(
        capability="synthesis",
        status="success",
        description="Synthesized multi-tool results into a structured grounded answer.",
        duration=synth_duration,
    )

    final_answer = synth.answer or synth.format_markdown()

    # --- Step 5: Consolidate Metadata & Session Context ---
    elapsed = time.time() - t_start
    trace.total_duration_seconds = round(elapsed, 3)

    consolidated_metadata: dict[str, Any] = {
        "processing_time_seconds": round(elapsed, 3),
        "image1_dimensions": f"{raster1.width}x{raster1.height}",
        "image1_sensor_type": raster1.sensor_type.value,
        "image1_bands": raster1.bands,
        "capabilities_executed": [t.capability for t in plan.tasks],
        "planner_type": plan.planner_used,
        "tasks_count": len(task_results),
    }

    # Merge tool-specific metrics
    for res in task_results:
        if res.metadata:
            for k, v in res.metadata.items():
                if k not in consolidated_metadata:
                    consolidated_metadata[k] = v

    if raster2:
        consolidated_metadata["image2_dimensions"] = f"{raster2.width}x{raster2.height}"
        consolidated_metadata["image2_sensor_type"] = raster2.sensor_type.value

    # Update session context
    primary_tool_name = task_results[0].tool_used if task_results else plan.intent
    ctx.add_turn(
        query=query,
        answer=final_answer,
        tool_used=primary_tool_name,
        metadata=consolidated_metadata,
    )

    final_result = AnalysisResult(
        answer=final_answer,
        evidence=primary_evidence,
        mask=primary_mask,
        index_name=primary_index_name,
        confidence=None,
        tool_used=primary_tool_name,
        metadata=consolidated_metadata,
        trace=trace,
        session_context=ctx,
        status=synth.status,
        intent=synth.intent or plan.intent,
        summary=synth.summary,
        observations=synth.observations,
        key_visual_features=synth.key_visual_features,
        interpretation=synth.interpretation,
        evidence_sources=synth.evidence,
        confidence_level=synth.confidence,
        limitations=synth.limitations,
        sources=synth.sources or [primary_tool_name],
        structured_output=synth.model_dump(),
    )

    logger.info(f"Analysis completed in {elapsed:.3f}s. Primary tool: {final_result.tool_used}")
    return final_result


def _execute_legacy_fallback(
    plan: ExecutionPlan,
    raster1: RasterImage,
    raster2: RasterImage | None,
) -> AnalysisResult:
    """Legacy deterministic tool executor for fallback scenarios."""
    intent_str = plan.intent

    # Map string intent to Enum
    try:
        intent = Intent(intent_str)
    except ValueError:
        intent = Intent.UNSUPPORTED

    # Out of scope
    if intent == Intent.OUT_OF_SCOPE:
        from tools.registry import OutOfScopeCapability
        return OutOfScopeCapability().execute(ExecutionContext(image1=raster1, query=plan.reasoning))

    # Insufficient evidence
    if intent == Intent.INSUFFICIENT_EVIDENCE:
        from tools.registry import InsufficientEvidenceCapability
        return InsufficientEvidenceCapability().execute(ExecutionContext(image1=raster1, query=plan.reasoning))

    # Change detection
    if intent == Intent.CHANGE_DETECTION:
        if raster2 is None:
            return AnalysisResult(
                answer=(
                    "⚠️ Change detection requires two images.\n\n"
                    "Please upload a second image and try again.\n\n"
                    f"Your query matched change detection, but only one image was provided."
                ),
                status="insufficient_evidence",
                intent="change_detection",
                summary="Change detection requires two images.",
                tool_used="change_detection (blocked)",
                metadata={"reason": "missing_second_image"},
            )
        try:
            return detect_changes_changeformer(raster1, raster2)
        except ChangeFormerError as e:
            return AnalysisResult(
                answer=f"❌ Change detection failed: {e}",
                status="error",
                tool_used="changeformer (error)",
                metadata={"error": str(e)},
            )

    # Single-image tools
    if intent in SINGLE_IMAGE_TOOLS:
        tool_fn = SINGLE_IMAGE_TOOLS[intent]
        try:
            return tool_fn(raster1, raster2, query=plan.reasoning)
        except TypeError:
            try:
                return tool_fn(raster1, raster2)
            except TypeError:
                try:
                    return tool_fn(raster1, query=plan.reasoning)
                except TypeError:
                    return tool_fn(raster1)

    # Multi-modal Optical + SAR
    if intent == Intent.OPTICAL_SAR_ANALYSIS:
        from tools.bifold_tool import analyze_optical_sar_bifold, BIFOLDToolError
        try:
            return analyze_optical_sar_bifold(raster1, query=plan.reasoning)
        except BIFOLDToolError as e:
            return AnalysisResult(
                answer=f"❌ BIFOLD analysis failed: {e}",
                status="error",
                tool_used="bifold_rdnet (error)",
                metadata={"error": str(e)},
            )

    # Image description
    if intent == Intent.IMAGE_DESCRIPTION:
        vlm = get_vlm()
        answer = vlm.analyze("", raster1)
        return AnalysisResult(
            answer=answer,
            status="success",
            intent="image_description",
            summary=answer,
            tool_used="vlm_description",
            metadata={"method": "rule_based" if not vlm.is_available() else "vlm"},
        )

    # Unsupported
    if intent == Intent.UNSUPPORTED:
        return AnalysisResult(
            answer=(
                f"⚠️ Unsupported query.\n\n"
                f"The system cannot process: '{plan.reasoning}'\n\n"
                f"**Currently supported queries:**\n"
                f"- Water detection (rivers, lakes, floods)\n"
                "- Vegetation detection (forests, agriculture)\n"
                "- Built-up area detection (urban, construction)\n"
                "- Change detection (requires two images)\n"
                "- General image description\n"
            ),
            status="insufficient_evidence",
            intent="unsupported",
            summary=f"Unsupported query: {plan.reasoning}",
            tool_used="unsupported_handler",
            metadata={"reason": plan.reasoning},
        )

    return AnalysisResult(
        answer=f"⚠️ No tool found for intent: {intent_str}",
        tool_used="fallback",
        metadata={"intent": intent_str},
    )


def _execute_tool(
    plan: PlanResult,
    raster1: RasterImage,
    raster2: RasterImage | None,
) -> AnalysisResult:
    """Backward compatibility helper for unit tests."""
    exec_plan = ExecutionPlan(
        status="ready" if plan.intent != Intent.OUT_OF_SCOPE else "out_of_scope",
        intent=plan.intent.value,
        reasoning=plan.reasoning,
        tasks=[TaskItem(capability=plan.intent.value, reason=plan.reasoning)],
        synthesis_required=False,
        planner_used="deterministic",
    )
    return _execute_legacy_fallback(exec_plan, raster1, raster2)
