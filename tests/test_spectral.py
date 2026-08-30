"""
Tests for spectral index calculations.
"""

import numpy as np
import pytest

from core.models import RasterImage, SensorType
from tools.spectral import (
    calculate_ndvi,
    calculate_ndwi,
    calculate_ndbi,
    can_compute_ndvi,
    can_compute_ndwi,
    can_compute_ndbi,
    rgb_greenness_index,
    rgb_water_proxy,
    rgb_urban_proxy,
    create_mask,
    _normalized_index,
)


class TestNormalizedIndex:
    """Test the base normalized index calculation."""

    def test_basic_calculation(self):
        a = np.array([[100.0, 200.0]])
        b = np.array([[50.0, 100.0]])
        result = _normalized_index(a, b)
        # (100-50)/(100+50) = 0.333..., (200-100)/(200+100) = 0.333...
        np.testing.assert_allclose(result[0, 0], 1 / 3, atol=1e-5)
        np.testing.assert_allclose(result[0, 1], 1 / 3, atol=1e-5)

    def test_zero_denominator(self):
        a = np.array([[0.0]])
        b = np.array([[0.0]])
        result = _normalized_index(a, b)
        assert np.isnan(result[0, 0])

    def test_range(self):
        """Values should be in [-1, 1] for valid inputs."""
        np.random.seed(0)
        a = np.random.rand(10, 10) * 1000
        b = np.random.rand(10, 10) * 1000
        result = _normalized_index(a, b)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= -1.0)
        assert np.all(result[valid] <= 1.0)

    def test_nodata_handling(self):
        a = np.array([[100.0, -9999.0]])
        b = np.array([[50.0, 50.0]])
        result = _normalized_index(a, b, nodata=-9999.0)
        assert np.isnan(result[0, 1])
        assert not np.isnan(result[0, 0])

    def test_dimensions_preserved(self):
        a = np.random.rand(32, 64)
        b = np.random.rand(32, 64)
        result = _normalized_index(a, b)
        assert result.shape == (32, 64)


class TestCanCompute:
    """Test band availability checks."""

    def test_ndvi_multispectral(self, multispectral_image):
        assert can_compute_ndvi(multispectral_image) is True

    def test_ndvi_rgb(self, rgb_image):
        assert can_compute_ndvi(rgb_image) is False

    def test_ndwi_multispectral(self, multispectral_image):
        assert can_compute_ndwi(multispectral_image) is True

    def test_ndwi_rgb(self, rgb_image):
        assert can_compute_ndwi(rgb_image) is False

    def test_ndbi_multispectral(self, multispectral_image):
        assert can_compute_ndbi(multispectral_image) is True

    def test_ndbi_rgb(self, rgb_image):
        assert can_compute_ndbi(rgb_image) is False


class TestCalculateNDVI:
    """Test NDVI calculation."""

    def test_multispectral_ndvi(self, multispectral_image):
        result = calculate_ndvi(multispectral_image)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= -1.0)
        assert np.all(result[valid] <= 1.0)

    def test_rgb_raises_error(self, rgb_image):
        with pytest.raises(ValueError, match="lacks required"):
            calculate_ndvi(rgb_image)


class TestCalculateNDWI:
    """Test NDWI calculation."""

    def test_multispectral_ndwi(self, multispectral_image):
        result = calculate_ndwi(multispectral_image)
        assert result.shape == (64, 64)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= -1.0)
        assert np.all(result[valid] <= 1.0)

    def test_rgb_raises_error(self, rgb_image):
        with pytest.raises(ValueError, match="lacks required"):
            calculate_ndwi(rgb_image)


class TestCalculateNDBI:
    """Test NDBI calculation."""

    def test_multispectral_ndbi(self, multispectral_image):
        result = calculate_ndbi(multispectral_image)
        assert result.shape == (64, 64)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= -1.0)
        assert np.all(result[valid] <= 1.0)

    def test_rgb_raises_error(self, rgb_image):
        with pytest.raises(ValueError, match="lacks required"):
            calculate_ndbi(rgb_image)


class TestRGBProxies:
    """Test RGB visual proxy calculations."""

    def test_greenness_index(self, rgb_image):
        result = rgb_greenness_index(rgb_image)
        assert result.shape == (64, 64)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= -1.0)
        assert np.all(result[valid] <= 1.0)

    def test_water_proxy(self, rgb_image):
        result = rgb_water_proxy(rgb_image)
        assert result.shape == (64, 64)

    def test_urban_proxy(self, rgb_image):
        result = rgb_urban_proxy(rgb_image)
        assert result.shape == (64, 64)

    def test_greenness_on_vegetation_like(self):
        """Green pixels should have positive greenness."""
        data = np.zeros((10, 10, 3), dtype=np.float32)
        data[:, :, 0] = 50   # Red
        data[:, :, 1] = 200  # Green (high)
        data[:, :, 2] = 30   # Blue
        image = RasterImage(
            data=data, bands=["red", "green", "blue"], sensor_type=SensorType.RGB
        )
        result = rgb_greenness_index(image)
        assert np.all(result > 0)


class TestCreateMask:
    """Test mask generation."""

    def test_basic_mask(self):
        index = np.array([[0.5, -0.3], [0.1, 0.8]])
        mask = create_mask(index, threshold=0.2, above=True)
        assert mask[0, 0] == 255  # 0.5 > 0.2
        assert mask[0, 1] == 0    # -0.3 < 0.2
        assert mask[1, 0] == 0    # 0.1 < 0.2
        assert mask[1, 1] == 255  # 0.8 > 0.2

    def test_mask_dtype(self):
        index = np.zeros((5, 5))
        mask = create_mask(index)
        assert mask.dtype == np.uint8

    def test_mask_below(self):
        index = np.array([[0.5, -0.3], [0.1, 0.8]])
        mask = create_mask(index, threshold=0.0, above=False)
        assert mask[0, 0] == 0    # 0.5 > 0 (not below)
        assert mask[0, 1] == 255  # -0.3 < 0 (below)

    def test_nan_handling(self):
        index = np.array([[0.5, np.nan]])
        mask = create_mask(index, threshold=0.0, above=True)
        assert mask[0, 0] == 255
        assert mask[0, 1] == 0  # NaN is not above threshold
