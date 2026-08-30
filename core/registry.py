"""
Capability Registry for SatQuery AI.

Provides a unified interface for registering, inspecting, and executing
both scientific tools (NDVI, NDWI, NDBI, Change Detection) and remote
AI models (GeoChat-7B, Future Foundation Models).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from core.models import AnalysisResult, RasterImage

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Input payload passed to any capability during execution."""
    image1: RasterImage
    image2: RasterImage | None = None
    query: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    session_context: Any = None


class Capability(ABC):
    """Abstract Base Class for all SatQuery capabilities and tools."""

    name: str
    description: str
    supported_inputs: list[str]  # e.g. ["single_image"], ["dual_image"]
    required_bands: list[str] = []
    requires_gpu: bool = False
    requires_external_api: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the capability is operational and its dependencies are met."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Return diagnostic health information."""
        return {
            "name": self.name,
            "available": self.is_available(),
            "requires_gpu": self.requires_gpu,
            "requires_external_api": self.requires_external_api,
        }

    @abstractmethod
    def execute(self, context: ExecutionContext) -> AnalysisResult:
        """Execute the capability against the provided context."""
        ...


class FunctionCapability(Capability):
    """Wraps a standard Python function into a registered capability."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., AnalysisResult],
        supported_inputs: list[str] | None = None,
        required_bands: list[str] | None = None,
        requires_gpu: bool = False,
        requires_external_api: bool = False,
        is_available_fn: Callable[[], bool] | None = None,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.supported_inputs = supported_inputs or ["single_image"]
        self.required_bands = required_bands or []
        self.requires_gpu = requires_gpu
        self.requires_external_api = requires_external_api
        self._is_available_fn = is_available_fn

    def is_available(self) -> bool:
        if self._is_available_fn is not None:
            return self._is_available_fn()
        return True

    def execute(self, context: ExecutionContext) -> AnalysisResult:
        if "dual_image" in self.supported_inputs:
            if context.image2 is None:
                return AnalysisResult(
                    answer=f"⚠️ Capability '{self.name}' requires two images.",
                    tool_used=f"{self.name} (blocked)",
                    metadata={"error": "missing_second_image"},
                )
            return self.fn(context.image1, context.image2)
        return self.fn(context.image1)


class CapabilityRegistry:
    """Central registry holding all operational tools and model capabilities."""

    def __init__(self):
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        """Register a new capability."""
        name = capability.name.lower().strip()
        self._capabilities[name] = capability
        logger.info(f"Registered capability: '{name}'")

    def get(self, name: str) -> Capability | None:
        """Retrieve capability by name (case-insensitive)."""
        return self._capabilities.get(name.lower().strip())

    def list_all(self) -> list[Capability]:
        """List all registered capabilities."""
        return list(self._capabilities.values())

    def list_available(self) -> list[Capability]:
        """List only currently available and healthy capabilities."""
        return [c for c in self._capabilities.values() if c.is_available()]

    def is_valid_capability(self, name: str) -> bool:
        """Validate if a capability name is known."""
        return name.lower().strip() in self._capabilities

    def get_capabilities_prompt(self) -> str:
        """
        Generate a formatted description of registered capabilities
        to be injected into the LLM planner prompt.
        """
        lines = []
        for cap in self.list_all():
            avail_str = "AVAILABLE" if cap.is_available() else "UNAVAILABLE"
            bands_str = f" (Required bands: {', '.join(cap.required_bands)})" if cap.required_bands else ""
            inputs_str = ", ".join(cap.supported_inputs)
            lines.append(
                f"- **{cap.name}** [{avail_str}]: {cap.description} "
                f"[Inputs: {inputs_str}]{bands_str}"
            )
        return "\n".join(lines)


# Global registry singleton
_registry = CapabilityRegistry()


def get_registry() -> CapabilityRegistry:
    """Return the global capability registry singleton, ensuring built-ins are registered."""
    if not _registry._capabilities:
        try:
            import tools.registry  # noqa: F401
        except Exception:
            pass
    return _registry
