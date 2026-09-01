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

    def test_geochat_oneline_response_expanded_explanation(self, rgb_image):
        plan = ExecutionPlan(
            intent="image_description",
            reasoning="Describe the satellite scene",
            tasks=[TaskItem(capability="geochat", reason="visual reasoning")],
            synthesis_required=False,
        )
        res = AnalysisResult(
            answer="A small lake surrounded by green farmland and agricultural plots.",
            tool_used="geochat",
        )
        client = LLMClient(provider=None)

        synth = synthesize_results("Describe this satellite scene", plan, [res], rgb_image, llm_client=client)

        assert synth.status == "success"
        assert synth.summary != ""
        assert len(synth.observations) > 0
        assert any("lake" in obs.lower() or "farmland" in obs.lower() for obs in synth.observations)
        assert len(synth.key_visual_features) > 0
        assert any("water" in feat.lower() for feat in synth.key_visual_features)
        assert any("vegetation" in feat.lower() or "agriculture" in feat.lower() for feat in synth.key_visual_features)
        assert synth.interpretation != ""
        assert len(synth.limitations) > 0
        assert "GeoChat-7B" in synth.evidence[0] or "GeoChat-7B" in synth.evidence[1]
        assert "Scene Overview" in synth.answer
        assert "Visual Features" in synth.answer

    def test_geochat_observations_plus_spectral_combined_grounded_answer(self, rgb_image):
        plan = ExecutionPlan(
            intent="water_detection",
            reasoning="Water detection with visual context",
            tasks=[
                TaskItem(capability="water_detection", reason="spectral NDWI"),
                TaskItem(capability="geochat", reason="visual description"),
            ],
            synthesis_required=True,
        )
        res_spectral = AnalysisResult(
            answer="Water coverage is 18.5%",
            tool_used="water_detection",
            metadata={"method": "true_ndwi", "coverage_percent": 18.5, "water_pixels": 1850, "total_pixels": 10000, "threshold": 0.0},
        )
        res_geochat = AnalysisResult(
            answer="The image shows a river flowing through agricultural fields and sparse vegetation.",
            tool_used="geochat",
        )
        client = LLMClient(provider=None)

        synth = synthesize_results("Find water bodies and describe the surrounding scene", plan, [res_spectral, res_geochat], rgb_image, llm_client=client)

        assert "18.5%" in synth
        assert any("18.5%" in obs for obs in synth.observations)
        assert any("river" in obs.lower() or "agricultural" in obs.lower() for obs in synth.observations)
        assert any("water" in feat.lower() for feat in synth.key_visual_features)
        assert any("vegetation" in feat.lower() or "agriculture" in feat.lower() for feat in synth.key_visual_features)
        assert "High" in synth.confidence
        assert any("NDWI" in ev for ev in synth.evidence)
        assert any("GeoChat-7B" in ev for ev in synth.evidence)

    def test_geochat_uncertain_response_preserved(self, rgb_image):
        plan = ExecutionPlan(
            intent="image_description",
            reasoning="Describe scene",
            tasks=[TaskItem(capability="geochat", reason="visual reasoning")],
            synthesis_required=False,
        )
        res = AnalysisResult(
            answer="The image is blurry and low-contrast with heavy cloud shadow.",
            tool_used="geochat",
        )
        client = LLMClient(provider=None)

        synth = synthesize_results("What do you see in this image?", plan, [res], rgb_image, llm_client=client)

        assert "Low" in synth.confidence
        assert any("blurry" in obs.lower() or "low-contrast" in obs.lower() for obs in synth.observations)
        assert any("contrast" in lim.lower() or "blur" in lim.lower() or "atmospheric" in lim.lower() for lim in synth.limitations)

    def test_no_geochat_evidence_no_fabricated_visual_observations(self, rgb_image):
        plan = ExecutionPlan(
            intent="vegetation_detection",
            reasoning="Vegetation NDVI calculation",
            tasks=[TaskItem(capability="vegetation_detection", reason="NDVI")],
            synthesis_required=False,
        )
        res = AnalysisResult(
            answer="Vegetation canopy coverage: 34.2%",
            tool_used="vegetation_detection",
            metadata={"method": "true_ndvi", "coverage_percent": 34.2, "veg_pixels": 3420, "total_pixels": 10000},
        )
        client = LLMClient(provider=None)

        synth = synthesize_results("Calculate vegetation coverage", plan, [res], rgb_image, llm_client=client)

        assert "34.2%" in synth
        assert not any("GeoChat" in ev for ev in synth.evidence)
        assert not any("highway" in feat.lower() for feat in synth.key_visual_features)

    def test_out_of_scope_query_synthesis(self, rgb_image):
        plan = ExecutionPlan(
            status="out_of_scope",
            intent="out_of_scope",
            reasoning="Out of scope query",
            tasks=[],
            synthesis_required=False,
        )
        res = AnalysisResult(answer="Out of scope", tool_used="router")
        client = LLMClient(provider=None)

        synth = synthesize_results("Who won the cricket world cup?", plan, [res], rgb_image, llm_client=client)

        assert synth.status == "out_of_scope"
        assert synth.intent == "out_of_scope"
        assert "SatQuery AI is designed for satellite" in synth.answer

    def test_insufficient_evidence_synthesis(self, rgb_image):
        plan = ExecutionPlan(
            status="insufficient_evidence",
            intent="change_detection",
            reasoning="Change detection missing second image",
            tasks=[],
            synthesis_required=False,
        )
        client = LLMClient(provider=None)

        synth = synthesize_results("Compare before and after", plan, [], rgb_image, llm_client=client)

        assert synth.status == "insufficient_evidence"
        assert "No analysis results" in synth.summary

    def test_malformed_llm_json_fallback(self, rgb_image):
        plan = ExecutionPlan(
            intent="image_description",
            reasoning="Describe image with LLM",
            tasks=[TaskItem(capability="geochat", reason="visual reasoning")],
            synthesis_required=True,
        )
        res = AnalysisResult(
            answer="Agricultural crop fields bordered by a small creek.",
            tool_used="geochat",
        )
        provider = MockLLMProvider(response_payload="{ corrupted json invalid syntax ...", available=True)
        client = LLMClient(provider=provider)

        synth = synthesize_results("Describe scene", plan, [res], rgb_image, llm_client=client)

        assert synth.status == "success"
        assert synth.summary != ""
        assert len(synth.observations) > 0
        assert "Scene Overview" in synth.answer
