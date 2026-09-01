"""
Vision-Language Model abstraction for SatQuery AI.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from core.models import AnalysisResult, RasterImage

logger = logging.getLogger(__name__)


class VisionLanguageModel(ABC):
    """Abstract interface for a vision-language model backend."""

    @abstractmethod
    def analyze(
        self,
        query: str,
        image: RasterImage,
        analysis_result: AnalysisResult | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Perform multimodal visual reasoning on the image and query.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the VLM service is configured and reachable."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Check the health of the VLM backend."""
        return {"available": self.is_available()}


class RuleBasedVLM(VisionLanguageModel):
    """Rule-based fallback when no remote VLM is available."""

    def analyze(
        self,
        query: str,
        image: RasterImage,
        analysis_result: AnalysisResult | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        if analysis_result is not None:
            return analysis_result.answer

        return (
            f"[Visual Baseline — GeoChat VLM Offline / Not Configured]\n\n"
            f"Image properties: {image.width}x{image.height} px, "
            f"{image.num_bands} band(s) ({', '.join(image.bands)}), "
            f"sensor type: {image.sensor_type.value}.\n\n"
            f"ℹ️ For detailed visual AI interpretation, connect the remote "
            f"GeoChat-7B GPU inference service via GEOCHAT_API_URL."
        )

    def is_available(self) -> bool:
        return False
