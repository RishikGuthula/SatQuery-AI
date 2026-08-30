"""
Tool registry.

Maps Intent values to their handler functions.
"""

from __future__ import annotations

from typing import Callable

from core.models import AnalysisResult, Intent, RasterImage
from tools.water_detection import detect_water
from tools.vegetation_detection import detect_vegetation
from tools.builtup_detection import detect_builtup
from tools.change_detection import detect_changes

# Type alias for single-image tool functions
SingleImageTool = Callable[[RasterImage], AnalysisResult]
# Type alias for dual-image tool functions
DualImageTool = Callable[[RasterImage, RasterImage], AnalysisResult]

# Registry of tools by intent
SINGLE_IMAGE_TOOLS: dict[Intent, SingleImageTool] = {
    Intent.WATER_DETECTION: detect_water,
    Intent.VEGETATION_DETECTION: detect_vegetation,
    Intent.BUILTUP_DETECTION: detect_builtup,
}

DUAL_IMAGE_TOOLS: dict[Intent, DualImageTool] = {
    Intent.CHANGE_DETECTION: detect_changes,
}

# All tools
TOOLS: dict[str, callable] = {
    "water_detection": detect_water,
    "vegetation_detection": detect_vegetation,
    "builtup_detection": detect_builtup,
    "change_detection": detect_changes,
}
