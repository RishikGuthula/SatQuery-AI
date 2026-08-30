"""
Tests for LLM Structured Planner and Fallbacks.
"""

import json
import pytest
from unittest.mock import MagicMock

from core.models import RasterImage, SensorType
from llm.base import LLMResponse, LLMProvider
from llm.client import LLMClient
from llm.planner import plan_with_llm, ExecutionPlan, TaskItem


class MockLLMProvider(LLMProvider):
    """Mock LLM Provider for unit testing."""

    def __init__(self, response_payload: str | None = None, available: bool = True):
        super().__init__(api_key="mock-key", model="mock-model")
        self.response_payload = response_payload
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def generate(self, messages, temperature=0.2, max_tokens=1000, response_format=None):
        if not self._available or self.response_payload is None:
            raise RuntimeError("LLM Provider unavailable")
        return LLMResponse(content=self.response_payload, model="mock-model")


class TestLLMPlanner:
    """Test LLM structured planning and schema validation."""

    def test_plan_with_valid_llm_json(self, rgb_image):
        valid_json = json.dumps({
            "intent": "water_detection",
            "reasoning": "User asked for water bodies",
            "tasks": [
                {"capability": "water_detection", "reason": "Detect water in RGB image", "parameters": {}}
            ],
            "synthesis_required": False
        })
        provider = MockLLMProvider(response_payload=valid_json, available=True)
        client = LLMClient(provider=provider)

        plan = plan_with_llm("Find water bodies", image1=rgb_image, llm_client=client)

        assert isinstance(plan, ExecutionPlan)
        assert plan.intent == "water_detection"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].capability == "water_detection"
        assert plan.planner_used == "llm"

    def test_plan_filters_unregistered_capabilities(self, rgb_image):
        """Planner must reject hallucinated/unregistered capability names."""
        invalid_json = json.dumps({
            "intent": "magic_analysis",
            "reasoning": "Hallucinated capability",
            "tasks": [
                {"capability": "magic_satellite_detector", "reason": "Fake tool"},
                {"capability": "vegetation_detection", "reason": "Real tool"}
            ],
            "synthesis_required": True
        })
        provider = MockLLMProvider(response_payload=invalid_json, available=True)
        client = LLMClient(provider=provider)

        plan = plan_with_llm("Analyze vegetation with magic", image1=rgb_image, llm_client=client)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].capability == "vegetation_detection"

    def test_fallback_when_llm_unavailable(self, rgb_image):
        """When LLM is offline, deterministic planner must kick in seamlessly."""
        provider = MockLLMProvider(available=False)
        client = LLMClient(provider=provider)

        plan = plan_with_llm("Calculate NDVI", image1=rgb_image, llm_client=client)

        assert isinstance(plan, ExecutionPlan)
        assert plan.planner_used == "deterministic"
        assert plan.intent == "vegetation_detection"
        assert len(plan.tasks) >= 1

    def test_fallback_on_corrupt_llm_json(self, rgb_image):
        """When LLM returns non-JSON or invalid schema, fallback must trigger."""
        provider = MockLLMProvider(response_payload="I cannot output JSON today!", available=True)
        client = LLMClient(provider=provider)

        plan = plan_with_llm("Find water", image1=rgb_image, llm_client=client)

        assert isinstance(plan, ExecutionPlan)
        assert plan.planner_used == "deterministic"
        assert plan.intent == "water_detection"
