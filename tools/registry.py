"""
Tool registry and capability registration for SatQuery AI.

Maps Intent categories and capability names to their execution functions,
and registers all capabilities into the central CapabilityRegistry.
"""

from __future__ import annotations

from typing import Callable

from core.models import AnalysisResult, Intent, RasterImage
from core.registry import (
    Capability,
    ExecutionContext,
    FunctionCapability,
    get_registry,
)
from tools.water_detection import detect_water
from tools.vegetation_detection import detect_vegetation
from tools.builtup_detection import detect_builtup
from tools.change_detection import detect_changes
from tools.changeformer_tool import detect_changes_changeformer
from tools.bifold_tool import analyze_optical_sar_bifold
from vlm.client import get_vlm

# Type aliases
SingleImageTool = Callable[[RasterImage], AnalysisResult]
DualImageTool = Callable[[RasterImage, RasterImage], AnalysisResult]

# Legacy dictionary lookups for backward compatibility
SINGLE_IMAGE_TOOLS: dict[Intent, SingleImageTool] = {
    Intent.WATER_DETECTION: detect_water,
    Intent.VEGETATION_DETECTION: detect_vegetation,
    Intent.BUILTUP_DETECTION: detect_builtup,
}

DUAL_IMAGE_TOOLS: dict[Intent, DualImageTool] = {
    Intent.CHANGE_DETECTION: detect_changes_changeformer,
}

TOOLS: dict[str, callable] = {
    "water_detection": detect_water,
    "vegetation_detection": detect_vegetation,
    "builtup_detection": detect_builtup,
    "change_detection": detect_changes_changeformer,
    "changeformer": detect_changes_changeformer,
    "bifold": analyze_optical_sar_bifold,
    "bifold_rdnet": analyze_optical_sar_bifold,
    "optical_sar": analyze_optical_sar_bifold,
}


class GeoChatCapability(Capability):
    """Registered capability for remote GeoChat-7B visual reasoning."""

    name = "geochat"
    description = (
        "Remote GPU-hosted GeoChat-7B vision-language model for natural language "
        "satellite image interpretation and visual reasoning."
    )
    supported_inputs = ["single_image"]
    required_bands = ["red", "green", "blue"]
    requires_gpu = True
    requires_external_api = True

    def is_available(self) -> bool:
        vlm = get_vlm()
        return vlm.is_available()

    def health_check(self) -> dict:
        vlm = get_vlm()
        return vlm.health_check()

    def execute(self, context: ExecutionContext) -> AnalysisResult:
        vlm = get_vlm()
        answer = vlm.analyze(query=context.query, image=context.image1)
        return AnalysisResult(
            answer=answer,
            tool_used="geochat (remote_gpu)",
            metadata={"model": "geochat-7b"},
        )


class ImageDescriptionCapability(Capability):
    """Registered capability for baseline image description."""

    name = "image_description"
    description = "Provides metadata overview and structural baseline description of the image."
    supported_inputs = ["single_image"]
    required_bands = []
    requires_gpu = False
    requires_external_api = False

    def is_available(self) -> bool:
        return True

    def execute(self, context: ExecutionContext) -> AnalysisResult:
        vlm = get_vlm()
        answer = vlm.analyze(query=context.query, image=context.image1)
        return AnalysisResult(
            answer=answer,
            tool_used="image_description",
            metadata={"bands": context.image1.bands, "dimensions": f"{context.image1.width}x{context.image1.height}"},
        )


class OutOfScopeCapability(Capability):
    """Handler for out-of-scope non-geospatial queries."""

    name = "out_of_scope"
    description = "Handles queries unrelated to satellite imagery and Earth observation."
    supported_inputs = ["single_image", "dual_image"]

    def is_available(self) -> bool:
        return True

    def execute(self, context: ExecutionContext) -> AnalysisResult:
        answer = (
            "🛰️ **SatQuery AI is designed for satellite and Earth observation questions.**\n\n"
            "Your query appears to be unrelated to remote sensing or satellite imagery.\n\n"
            "**Supported topics include:**\n"
            "• Water detection & NDWI index\n"
            "• Vegetation canopy & NDVI index\n"
            "• Built-up area detection & NDBI index\n"
            "• Temporal change detection (with two images)\n"
            "• Optical + SAR multi-modal classification (BIFOLD RDNet)\n"
            "• Visual reasoning and scene description (GeoChat-7B)\n\n"
            "Please enter a question about your satellite image."
        )
        return AnalysisResult(
            answer=answer,
            status="out_of_scope",
            intent="out_of_scope",
            summary="Query is outside the scope of satellite and Earth observation analysis.",
            observations=[],
            evidence_sources=[],
            confidence_level="Not Applicable",
            sources=[],
            tool_used="domain_validator",
            metadata={"query": context.query, "reason": "out_of_scope"},
        )


class InsufficientEvidenceCapability(Capability):
    """Handler for queries where required evidence/images are missing."""

    name = "insufficient_evidence"
    description = "Handles queries lacking necessary input data or secondary imagery."
    supported_inputs = ["single_image", "dual_image"]

    def is_available(self) -> bool:
        return True

    def execute(self, context: ExecutionContext) -> AnalysisResult:
        return AnalysisResult(
            answer="⚠️ The available data is insufficient to answer this query. Please provide the required imagery or spectral bands.",
            status="insufficient_evidence",
            intent="insufficient_evidence",
            summary="Available evidence is insufficient.",
            tool_used="evidence_validator",
            metadata={"query": context.query},
        )


class UnsupportedCapability(Capability):
    """Handler for unsupported remote-sensing queries."""

    name = "unsupported"
    description = "Handles queries outside the current scope of remote-sensing tools."
    supported_inputs = ["single_image", "dual_image"]

    def is_available(self) -> bool:
        return True

    def execute(self, context: ExecutionContext) -> AnalysisResult:
        return AnalysisResult(
            answer=(
                "⚠️ Unsupported query.\n\n"
                "The requested analysis is outside the current scope of remote-sensing tools.\n\n"
                "**Supported capabilities:**\n"
                "- Water detection (NDWI / rivers, lakes, oceans)\n"
                "- Vegetation detection (NDVI / forest, agriculture, greenness)\n"
                "- Built-up area detection (NDBI / urban, structures)\n"
                "- Change detection (between two temporal satellite images)\n"
                "- Multi-modal Optical + SAR land-cover classification via BIFOLD RDNet Base\n"
                "- Remote visual interpretation via GeoChat-7B\n"
            ),
            status="insufficient_evidence",
            intent="unsupported",
            summary="Query is not currently supported by registered remote-sensing tools.",
            tool_used="unsupported_handler",
            metadata={"query": context.query},
        )


def register_all_capabilities() -> None:
    """Register all available tools and models into the global registry."""
    reg = get_registry()

    # Scientific tools
    reg.register(
        FunctionCapability(
            name="water_detection",
            description="Detects water bodies. Computes true NDWI (Green+NIR) or RGB water proxy.",
            fn=detect_water,
            supported_inputs=["single_image"],
            required_bands=[],
        )
    )
    reg.register(
        FunctionCapability(
            name="vegetation_detection",
            description="Detects vegetation and green canopy. Computes true NDVI (NIR+Red) or RGB greenness proxy.",
            fn=detect_vegetation,
            supported_inputs=["single_image"],
            required_bands=[],
        )
    )
    reg.register(
        FunctionCapability(
            name="builtup_detection",
            description="Detects urban and built-up areas. Computes true NDBI (SWIR+NIR) or RGB urban proxy.",
            fn=detect_builtup,
            supported_inputs=["single_image"],
            required_bands=[],
        )
    )
    reg.register(
        FunctionCapability(
            name="change_detection",
            description="Detects land-cover changes between two bi-temporal satellite images using ChangeFormer deep-learning architecture.",
            fn=detect_changes_changeformer,
            supported_inputs=["dual_image"],
        )
    )
    reg.register(
        FunctionCapability(
            name="changeformer",
            description="ChangeFormer bi-temporal Transformer model for remote sensing satellite change detection.",
            fn=detect_changes_changeformer,
            supported_inputs=["dual_image"],
        )
    )

    # Multi-modal AI Models
    reg.register(
        FunctionCapability(
            name="bifold",
            description="BIFOLD RDNet Base model for 12-channel Sentinel-1 (VV, VH) + Sentinel-2 multispectral land-cover classification.",
            fn=analyze_optical_sar_bifold,
            supported_inputs=["single_image"],
            required_bands=["vv", "vh", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b11", "b12"],
        )
    )
    reg.register(
        FunctionCapability(
            name="bifold_rdnet",
            description="Alias for BIFOLD RDNet Base multi-modal optical+SAR classification.",
            fn=analyze_optical_sar_bifold,
            supported_inputs=["single_image"],
            required_bands=["vv", "vh", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b11", "b12"],
        )
    )

    # Aliases
    reg.register(
        FunctionCapability(
            name="ndvi",
            description="Alias for vegetation_detection (calculates NDVI index).",
            fn=detect_vegetation,
            supported_inputs=["single_image"],
            required_bands=["nir", "red"],
        )
    )
    reg.register(
        FunctionCapability(
            name="ndwi",
            description="Alias for water_detection (calculates NDWI index).",
            fn=detect_water,
            supported_inputs=["single_image"],
            required_bands=["green", "nir"],
        )
    )
    reg.register(
        FunctionCapability(
            name="ndbi",
            description="Alias for builtup_detection (calculates NDBI index).",
            fn=detect_builtup,
            supported_inputs=["single_image"],
            required_bands=["swir1", "nir"],
        )
    )

    # AI Models & Descriptions
    reg.register(GeoChatCapability())
    reg.register(ImageDescriptionCapability())
    reg.register(OutOfScopeCapability())
    reg.register(InsufficientEvidenceCapability())
    reg.register(UnsupportedCapability())



# Automatically register on module import
register_all_capabilities()
