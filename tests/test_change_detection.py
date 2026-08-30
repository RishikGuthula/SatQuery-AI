"""
Tests for change detection.
"""

import numpy as np
import pytest

from core.models import RasterImage, SensorType
from tools.change_detection import (
    detect_changes,
    validate_pair,
    ChangeDetectionError,
    _resize_to_match,
    _normalize,
)


@pytest.fixture
def pair_rgb():
    """Two compatible RGB RasterImages."""
    np.random.seed(0)
    a = RasterImage(
        data=np.random.rand(32, 32, 3).astype(np.float32) * 255,
        bands=["red", "green", "blue"],
        sensor_type=SensorType.RGB,
    )
    b = RasterImage(
        data=np.random.rand(32, 32, 3).astype(np.float32) * 255,
        bands=["red", "green", "blue"],
        sensor_type=SensorType.RGB,
    )
    return a, b


@pytest.fixture
def identical_pair():
    """Two identical RGB RasterImages."""
    np.random.seed(0)
    data = np.random.rand(32, 32, 3).astype(np.float32) * 255
    a = RasterImage(data=data.copy(), bands=["red", "green", "blue"], sensor_type=SensorType.RGB)
    b = RasterImage(data=data.copy(), bands=["red", "green", "blue"], sensor_type=SensorType.RGB)
    return a, b


class TestValidatePair:
    """Test image pair validation."""

    def test_compatible_pair(self, pair_rgb):
        a, b = pair_rgb
        validate_pair(a, b)  # Should not raise

    def test_sar_optical_mismatch(self):
        a = RasterImage(
            data=np.zeros((10, 10, 1), dtype=np.float32),
            bands=["gray"],
            sensor_type=SensorType.SAR,
        )
        b = RasterImage(
            data=np.zeros((10, 10, 3), dtype=np.float32),
            bands=["red", "green", "blue"],
            sensor_type=SensorType.RGB,
        )
        with pytest.raises(ChangeDetectionError, match="modality"):
            validate_pair(a, b)

    def test_optical_sar_mismatch(self):
        a = RasterImage(
            data=np.zeros((10, 10, 3), dtype=np.float32),
            bands=["red", "green", "blue"],
            sensor_type=SensorType.RGB,
        )
        b = RasterImage(
            data=np.zeros((10, 10, 1), dtype=np.float32),
            bands=["gray"],
            sensor_type=SensorType.SAR,
        )
        with pytest.raises(ChangeDetectionError, match="modality"):
            validate_pair(a, b)

    def test_empty_image(self):
        a = RasterImage(
            data=np.array([]).reshape(0, 0),
            bands=[],
            sensor_type=SensorType.RGB,
        )
        b = RasterImage(
            data=np.zeros((10, 10, 3), dtype=np.float32),
            bands=["red", "green", "blue"],
            sensor_type=SensorType.RGB,
        )
        with pytest.raises(ChangeDetectionError, match="empty"):
            validate_pair(a, b)


class TestResizeToMatch:
    """Test image alignment/resize."""

    def test_same_size_no_resize(self):
        a = np.random.rand(32, 32, 3).astype(np.float32)
        b = np.random.rand(32, 32, 3).astype(np.float32)
        ra, rb = _resize_to_match(a, b)
        assert ra.shape == (32, 32, 3)
        assert rb.shape == (32, 32, 3)

    def test_different_size_resize(self):
        a = np.random.rand(64, 64, 3).astype(np.float32) * 255
        b = np.random.rand(32, 32, 3).astype(np.float32) * 255
        ra, rb = _resize_to_match(a, b)
        assert ra.shape[0] == 32
        assert ra.shape[1] == 32
        assert rb.shape[0] == 32
        assert rb.shape[1] == 32


class TestNormalize:
    """Test normalization."""

    def test_basic(self):
        arr = np.array([0.0, 50.0, 100.0])
        result = _normalize(arr)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-5)

    def test_constant_array(self):
        arr = np.array([5.0, 5.0, 5.0])
        result = _normalize(arr)
        assert np.all(result == 0.0)


class TestDetectChanges:
    """Test full change detection pipeline."""

    def test_different_images(self, pair_rgb):
        a, b = pair_rgb
        result = detect_changes(a, b)
        assert result.tool_used == "change_detection (pixel-difference)"
        assert result.evidence is not None
        assert result.mask is not None
        assert "changed_percent" in result.metadata
        assert result.metadata["changed_percent"] >= 0

    def test_identical_images(self, identical_pair):
        a, b = identical_pair
        result = detect_changes(a, b)
        # Identical images should have very low change percentage
        assert result.metadata["changed_percent"] < 50.0  # Some due to normalization

    def test_returns_analysis_result(self, pair_rgb):
        a, b = pair_rgb
        result = detect_changes(a, b)
        assert hasattr(result, "answer")
        assert hasattr(result, "evidence")
        assert hasattr(result, "mask")
        assert hasattr(result, "metadata")

    def test_mask_dimensions(self, pair_rgb):
        a, b = pair_rgb
        result = detect_changes(a, b)
        assert result.mask.shape == (32, 32)

    def test_sar_optical_raises(self):
        a = RasterImage(
            data=np.random.rand(32, 32, 1).astype(np.float32),
            bands=["gray"],
            sensor_type=SensorType.SAR,
        )
        b = RasterImage(
            data=np.random.rand(32, 32, 3).astype(np.float32),
            bands=["red", "green", "blue"],
            sensor_type=SensorType.RGB,
        )
        with pytest.raises(ChangeDetectionError):
            detect_changes(a, b)
