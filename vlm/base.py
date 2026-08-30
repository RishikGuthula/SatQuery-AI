"""
Vision-Language Model abstraction.

Defines the interface for integrating a VLM (e.g., GPT-4V, LLaVA, etc.).
Currently provides a rule-based fallback that does NOT claim to be AI reasoning.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from core.models import AnalysisResult, RasterImage

logger = logging.getLogger(__name__)


class VisionLanguageModel(ABC):
    """
    Abstract interface for a vision-language model.

    When a real VLM is available, implement this interface.
    The VLM is responsible for:
      - Query understanding
      - Natural language explanation of results
      - Contextual interpretation

    It must NOT fabricate numerical remote-sensing measurements.
    """

    @abstractmethod
    def analyze(
        self,
        query: str,
        image: RasterImage,
        analysis_result: AnalysisResult | None = None,
    ) -> str:
        """
        Generate a natural language explanation for an analysis result.

        Args:
            query: The user's original query.
            image: The input image.
            analysis_result: Optional result from a spectral tool.

        Returns:
            Natural language string explaining the result.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if a real VLM is configured and operational."""
        ...


class RuleBasedVLM(VisionLanguageModel):
    """
    Rule-based fallback when no real VLM is available.

    Does NOT pretend to be AI. Clearly labels itself as rule-based.
    """

    def analyze(
        self,
        query: str,
        image: RasterImage,
        analysis_result: AnalysisResult | None = None,
    ) -> str:
        if analysis_result is not None:
            # The tool already produced a good answer; pass it through.
            return analysis_result.answer

        # For image description / unsupported queries
        return (
            f"[Rule-based baseline — no VLM configured]\n\n"
            f"Image info: {image.width}x{image.height} pixels, "
            f"{image.num_bands} band(s), "
            f"sensor type: {image.sensor_type.value}.\n\n"
            f"To enable AI-powered analysis, configure a Vision-Language Model "
            f"(e.g., GPT-4V, LLaVA, or similar) by implementing the "
            f"VisionLanguageModel interface in vlm/base.py."
        )

    def is_available(self) -> bool:
        return False


def get_vlm() -> VisionLanguageModel:
    """
    Return the configured VLM instance.

    Currently returns the rule-based fallback.
    Replace this to plug in a real VLM.
    """
    return RuleBasedVLM()
