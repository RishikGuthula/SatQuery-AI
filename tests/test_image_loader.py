"""
Tests for the image loading layer.
"""

import io
import numpy as np
import pytest
from PIL import Image

from core.image_loader import (
    load_from_bytes,
    load_from_pil_image,
    ImageLoadError,
)
from core.models import SensorType


class TestLoadFromBytes:
    """Test loading images from raw bytes."""

    def test_load_rgb_png(self, rgb_bytes):
        raster = load_from_bytes(rgb_bytes, "test.png")
        assert raster.width == 32
        assert raster.height == 32
        assert raster.num_bands == 3
        assert raster.sensor_type == SensorType.RGB
        assert "red" in raster.bands
        assert "green" in raster.bands
        assert "blue" in raster.bands

    def test_load_rgb_jpeg(self):
        img = Image.new("RGB", (50, 50), color=(100, 200, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raster = load_from_bytes(buf.getvalue(), "test.jpg")
        assert raster.width == 50
        assert raster.height == 50
        assert raster.num_bands == 3

    def test_load_rgba_png(self):
        img = Image.new("RGBA", (30, 30), color=(100, 200, 50, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raster = load_from_bytes(buf.getvalue(), "test.png")
        assert raster.num_bands == 3  # Alpha dropped

    def test_load_grayscale(self):
        img = Image.new("L", (40, 40), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raster = load_from_bytes(buf.getvalue(), "test.png")
        assert raster.num_bands == 1
        assert raster.bands == ["gray"]

    def test_empty_bytes_raises(self, empty_bytes):
        with pytest.raises(ImageLoadError, match="Empty file"):
            load_from_bytes(empty_bytes, "empty.png")

    def test_invalid_bytes_raises(self):
        with pytest.raises(ImageLoadError):
            load_from_bytes(b"not an image at all", "bad.dat")

    def test_data_shape(self, rgb_bytes):
        raster = load_from_bytes(rgb_bytes, "test.png")
        assert raster.data.shape == (32, 32, 3)
        assert raster.data.dtype == np.float32

    def test_to_rgb(self, rgb_bytes):
        raster = load_from_bytes(rgb_bytes, "test.png")
        rgb = raster.to_rgb()
        assert rgb.shape == (32, 32, 3)
        assert rgb.dtype == np.uint8


class TestLoadFromPILImage:
    """Test loading from PIL Image objects."""

    def test_basic_pil_load(self):
        img = Image.new("RGB", (20, 20), color=(50, 100, 150))
        raster = load_from_pil_image(img, "test.png")
        assert raster.width == 20
        assert raster.height == 20

    def test_preserves_colors(self):
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        raster = load_from_pil_image(img)
        # Red channel should be max
        assert raster.data[0, 0, 0] == 255.0


class TestRasterImageModel:
    """Test RasterImage data model methods."""

    def test_has_band(self, rgb_image):
        assert rgb_image.has_band("red") is True
        assert rgb_image.has_band("nir") is False

    def test_get_band(self, rgb_image):
        red = rgb_image.get_band("red")
        assert red is not None
        assert red.shape == (64, 64)

    def test_get_band_missing(self, rgb_image):
        nir = rgb_image.get_band("nir")
        assert nir is None

    def test_get_band_index(self, multispectral_image):
        idx = multispectral_image.get_band_index("nir")
        assert idx == 6  # nir is at index 6

    def test_num_bands(self, rgb_image):
        assert rgb_image.num_bands == 3

    def test_dimensions(self, rgb_image):
        assert rgb_image.width == 64
        assert rgb_image.height == 64
