"""
Tests for Capability Registry.
"""

import pytest
from core.registry import (
    CapabilityRegistry,
    Capability,
    FunctionCapability,
    ExecutionContext,
    get_registry,
)
from core.models import AnalysisResult, RasterImage, SensorType
import numpy as np


class DummyCapability(Capability):
    name = "dummy_tool"
    description = "A dummy capability for testing."
    supported_inputs = ["single_image"]
    required_bands = ["red", "green"]

    def __init__(self, available: bool = True):
        self._avail = available

    def is_available(self) -> bool:
        return self._avail

    def execute(self, context: ExecutionContext) -> AnalysisResult:
        return AnalysisResult(answer="Dummy executed successfully", tool_used="dummy_tool")


class TestCapabilityRegistry:
    """Test registry registration and query logic."""

    def test_register_and_get(self):
        reg = CapabilityRegistry()
        dummy = DummyCapability(available=True)
        reg.register(dummy)

        assert reg.get("dummy_tool") is dummy
        assert reg.get("DUMMY_TOOL") is dummy  # Case-insensitive
        assert reg.is_valid_capability("dummy_tool") is True
        assert reg.is_valid_capability("unknown_tool") is False

    def test_list_all_and_available(self):
        reg = CapabilityRegistry()
        active = DummyCapability(available=True)
        active.name = "active_tool"
        inactive = DummyCapability(available=False)
        inactive.name = "inactive_tool"

        reg.register(active)
        reg.register(inactive)

        assert len(reg.list_all()) == 2
        avail = reg.list_available()
        assert len(avail) == 1
        assert avail[0].name == "active_tool"

    def test_get_capabilities_prompt(self):
        reg = CapabilityRegistry()
        dummy = DummyCapability(available=True)
        reg.register(dummy)

        prompt = reg.get_capabilities_prompt()
        assert "dummy_tool" in prompt
        assert "AVAILABLE" in prompt
        assert "red, green" in prompt

    def test_function_capability_single_image(self, rgb_image):
        def sample_fn(img: RasterImage) -> AnalysisResult:
            return AnalysisResult(answer=f"Processed {img.width}x{img.height}", tool_used="sample")

        cap = FunctionCapability(
            name="sample_fn",
            description="Sample function capability",
            fn=sample_fn,
            supported_inputs=["single_image"],
        )

        ctx = ExecutionContext(image1=rgb_image)
        res = cap.execute(ctx)
        assert res.answer == f"Processed {rgb_image.width}x{rgb_image.height}"
        assert res.tool_used == "sample"

    def test_function_capability_missing_second_image(self, rgb_image):
        def sample_dual(a: RasterImage, b: RasterImage) -> AnalysisResult:
            return AnalysisResult(answer="Dual processed", tool_used="dual")

        cap = FunctionCapability(
            name="sample_dual",
            description="Dual image capability",
            fn=sample_dual,
            supported_inputs=["dual_image"],
        )

        ctx = ExecutionContext(image1=rgb_image, image2=None)
        res = cap.execute(ctx)
        assert "requires two images" in res.answer
