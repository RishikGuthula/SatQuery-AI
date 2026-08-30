"""
Tests for the query planner / router.
"""

import pytest
from core.planner import plan_query, PlanResult
from core.models import Intent


class TestQueryPlanner:
    """Test query routing logic."""

    def test_water_detection_intent(self):
        result = plan_query("Find water bodies")
        assert result.intent == Intent.WATER_DETECTION

    def test_river_query(self):
        result = plan_query("Show the river area")
        assert result.intent == Intent.WATER_DETECTION

    def test_ndwi_query(self):
        result = plan_query("Calculate NDWI")
        assert result.intent == Intent.WATER_DETECTION

    def test_flood_query(self):
        result = plan_query("Detect flood areas")
        assert result.intent == Intent.WATER_DETECTION

    def test_vegetation_detection_intent(self):
        result = plan_query("Find vegetation")
        assert result.intent == Intent.VEGETATION_DETECTION

    def test_forest_query(self):
        result = plan_query("Detect forest cover")
        assert result.intent == Intent.VEGETATION_DETECTION

    def test_ndvi_query(self):
        result = plan_query("Calculate NDVI")
        assert result.intent == Intent.VEGETATION_DETECTION

    def test_agriculture_query(self):
        result = plan_query("Show agriculture areas")
        assert result.intent == Intent.VEGETATION_DETECTION

    def test_builtup_detection_intent(self):
        result = plan_query("Find built-up areas")
        assert result.intent == Intent.BUILTUP_DETECTION

    def test_urban_query(self):
        result = plan_query("Show urban regions")
        assert result.intent == Intent.BUILTUP_DETECTION

    def test_ndbi_query(self):
        result = plan_query("Calculate NDBI")
        assert result.intent == Intent.BUILTUP_DETECTION

    def test_change_detection_with_two_images(self):
        result = plan_query("Detect changes", has_second_image=True)
        assert result.intent == Intent.CHANGE_DETECTION

    def test_change_detection_without_second_image(self):
        result = plan_query("Detect changes", has_second_image=False)
        assert result.intent == Intent.CHANGE_DETECTION
        assert "second image" in result.reasoning.lower()

    def test_compare_query(self):
        result = plan_query("Compare these images", has_second_image=True)
        assert result.intent == Intent.CHANGE_DETECTION

    def test_describe_query(self):
        result = plan_query("Describe this image")
        assert result.intent == Intent.IMAGE_DESCRIPTION

    def test_unsupported_query(self):
        result = plan_query("Predict next year's financial returns from satellite")
        assert result.intent == Intent.UNSUPPORTED

    def test_empty_query(self):
        result = plan_query("")
        assert result.intent == Intent.UNSUPPORTED

    def test_whitespace_query(self):
        result = plan_query("   ")
        assert result.intent == Intent.UNSUPPORTED

    def test_plan_result_has_confidence(self):
        result = plan_query("Find water")
        assert 0.0 <= result.confidence <= 1.0

    def test_plan_result_has_reasoning(self):
        result = plan_query("Find vegetation")
        assert result.reasoning  # Non-empty
