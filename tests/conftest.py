"""
Shared test fixtures and utilities.
"""

import io
import numpy as np
import pytest
from PIL import Image

from core.models import RasterImage, SensorType


@pytest.fixture
def rgb_image() -> RasterImage:
    """Create a synthetic RGB RasterImage."""
    data = np.random.rand(64, 64, 3).astype(np.float32) * 255
    return RasterImage(
        data=data,
        bands=["red", "green", "blue"],
        sensor_type=SensorType.RGB,
        dtype="float32",
    )


@pytest.fixture
def multispectral_image() -> RasterImage:
    """Create a synthetic multispectral image with NIR, SWIR bands."""
    # 8 bands: coastal, blue, green, red, rededge1, rededge2, nir, swir1
    np.random.seed(42)
    data = np.random.rand(64, 64, 8).astype(np.float32) * 10000
    return RasterImage(
        data=data,
        bands=["coastal", "blue", "green", "red", "rededge1", "rededge2", "nir", "swir1"],
        sensor_type=SensorType.MULTISPECTRAL,
        dtype="float32",
    )


@pytest.fixture
def rgb_bytes() -> bytes:
    """Create PNG bytes for a small RGB image."""
    img = Image.new("RGB", (32, 32), color=(100, 150, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def rgb_vegetation_bytes() -> bytes:
    """Create PNG bytes with a green vegetation region."""
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    # Green vegetation area
    img_array[10:40, 10:40, 1] = 200  # Green channel
    img_array[10:40, 10:40, 0] = 50   # Low red
    img_array[10:40, 10:40, 2] = 30   # Low blue
    # Water area (blue)
    img_array[45:60, 45:60, 2] = 200  # Blue channel
    img_array[45:60, 45:60, 0] = 30
    img_array[45:60, 45:60, 1] = 50
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def empty_bytes() -> bytes:
    """Empty file bytes."""
    return b""


@pytest.fixture
def two_rgb_images() -> tuple[bytes, bytes]:
    """Two slightly different RGB images for change detection testing."""
    np.random.seed(0)
    img1 = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    img2 = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

    buf1 = io.BytesIO()
    img1.save(buf1, format="PNG")
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    return buf1.getvalue(), buf2.getvalue()


@pytest.fixture
def identical_images() -> tuple[bytes, bytes]:
    """Two identical RGB images for change detection testing."""
    np.random.seed(0)
    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    buf1 = io.BytesIO()
    img.save(buf1, format="PNG")
    buf2 = io.BytesIO()
    img.save(buf2, format="PNG")
    return buf1.getvalue(), buf2.getvalue()
