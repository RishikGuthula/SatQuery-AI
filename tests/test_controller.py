"""
Tests for the agent controller (end-to-end pipeline).
"""

import io
import numpy as np
import pytest
from PIL import Image

from agent.controller import process_query
from core.models import AnalysisResult


def _make_png_bytes(w: int = 32, h: int = 32, color=(100, 150, 80)) -> bytes:
    """Helper to create small PNG bytes."""
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestController:
    """Test the full query → tool → result pipeline."""

    def test_water_query_returns_result(self, rgb_bytes):
        result = process_query("Find water", rgb_bytes)
        assert isinstance(result, AnalysisResult)
        assert result.answer  # Non-empty
        assert "water" in result.tool_used.lower() or "water" in result.answer.lower()

    def test_vegetation_query_returns_result(self, rgb_bytes):
        result = process_query("Find vegetation", rgb_bytes)
        assert isinstance(result, AnalysisResult)
        assert result.answer

    def test_builtup_query_returns_result(self, rgb_bytes):
        result = process_query("Show built-up areas", rgb_bytes)
        assert isinstance(result, AnalysisResult)
        assert result.answer

    def test_unsupported_query(self, rgb_bytes):
        result = process_query("Predict financial returns from satellite data", rgb_bytes)
        assert isinstance(result, AnalysisResult)
        assert "unsupported" in result.tool_used.lower() or "Unsupported" in result.answer

    def test_change_detection_needs_two_images(self, rgb_bytes):
        result = process_query("Detect changes", rgb_bytes)
        assert "two images" in result.answer.lower() or "second image" in result.answer.lower()

    def test_change_detection_with_two_images(self, two_rgb_images):
        img1_bytes, img2_bytes = two_rgb_images
        result = process_query("Compare these images", img1_bytes, img2_bytes)
        assert isinstance(result, AnalysisResult)
        assert "change_detection" in result.tool_used.lower()
        assert result.metadata.get("changed_percent") is not None

    def test_invalid_image_returns_error(self):
        result = process_query("Find water", b"not an image")
        assert "error" in result.answer.lower() or "❌" in result.answer

    def test_empty_image_returns_error(self):
        result = process_query("Find water", b"")
        assert "error" in result.answer.lower() or "❌" in result.answer

    def test_metadata_has_timing(self, rgb_bytes):
        result = process_query("Find water", rgb_bytes)
        assert "processing_time_seconds" in result.metadata
        assert result.metadata["processing_time_seconds"] >= 0

    def test_metadata_has_image_info(self, rgb_bytes):
        result = process_query("Find water", rgb_bytes)
        assert "image1_dimensions" in result.metadata
        assert "image1_sensor_type" in result.metadata
        assert "image1_bands" in result.metadata

    def test_evidence_is_pil_image(self, rgb_bytes):
        result = process_query("Find water", rgb_bytes)
        if result.evidence is not None:
            assert isinstance(result.evidence, Image.Image)

    def test_describe_query(self, rgb_bytes):
        result = process_query("Describe this image", rgb_bytes)
        assert isinstance(result, AnalysisResult)
        assert result.answer  # Has some answer

    def test_result_has_tool_used(self, rgb_bytes):
        result = process_query("Find water", rgb_bytes)
        assert result.tool_used  # Non-empty
