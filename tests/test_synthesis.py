"""
Tests for LLM Synthesis and Grounding.
"""

import pytest
from core.models import AnalysisResult, RasterImage, SensorType
from llm.planner import ExecutionPlan, TaskItem
from llm.synthesis import synthesize_results, _fallback_rule_based_synthesis
from llm.client import LLMClient
from tests.test_llm_planner import MockLLMProvider


class TestSynthesis:
    """Test multi-result synthesis and grounding."""

    def test_single_tool_without_llm_returns_direct_answer(self, rgb_image):
        plan = ExecutionPlan(
            intent="water_detection",
            reasoning="Water query",
            tasks=[TaskItem(capability="water_detection", reason="find water")],
            synthesis_required=False,
        )
        res = AnalysisResult(answer="Water coverage is 15.4%", tool_used="water_detection")
        client = LLMClient(provider=None)

        final = synthesize_results("Find water", plan, [res], rgb_image, llm_client=client)
        assert final == "Water coverage is 15.4%"

    def test_multi_tool_rule_based_synthesis(self, rgb_image):
        plan = ExecutionPlan(
            intent="multi_feature",
            reasoning="Multi feature query",
            tasks=[
                TaskItem(capability="water_detection", reason="find water"),
                TaskItem(capability="vegetation_detection", reason="find veg"),
            ],
            synthesis_required=True,
        )
        res1 = AnalysisResult(answer="Water coverage is 12.0%", tool_used="water_detection")
        res2 = AnalysisResult(answer="Vegetation coverage is 45.5%", tool_used="vegetation_detection")
        client = LLMClient(provider=None)

        final = synthesize_results("Analyze scene", plan, [res1, res2], rgb_image, llm_client=client)
        assert "water_detection" in final
        assert "vegetation_detection" in final
        assert "12.0%" in final
        assert "45.5%" in final

    def test_llm_synthesis_success(self, rgb_image):
        plan = ExecutionPlan(
            intent="multi_feature",
            reasoning="Multi feature query",
            tasks=[
                TaskItem(capability="water_detection", reason="find water"),
                TaskItem(capability="geochat", reason="visual reasoning"),
            ],
            synthesis_required=True,
        )
        res1 = AnalysisResult(
            answer="Water coverage: 15.0%",
            tool_used="water_detection",
            metadata={"coverage_percent": 15.0},
        )
        res2 = AnalysisResult(
            answer="A river flows through an open plain.",
            tool_used="geochat",
        )

        mock_synth_text = "The scene contains a prominent river covering 15.0% of the land area flowing across an open plain."
        provider = MockLLMProvider(response_payload=mock_synth_text, available=True)
        client = LLMClient(provider=provider)

        final = synthesize_results("Find water and describe", plan, [res1, res2], rgb_image, llm_client=client)
        assert "15.0%" in final
        assert "open plain" in final
