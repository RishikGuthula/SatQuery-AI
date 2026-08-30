"""
Regression tests for NDWI/NDVI explicit routing, missing-band handling,
GeoChat response cleaning, and TIFF display integrity.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image

from core.image_loader import load_from_bytes
from core.models import RasterImage, SensorType
from tools.water_detection import detect_water
from tools.vegetation_detection import detect_vegetation
from vlm.response_cleaner import clean_vlm_response
from app import get_display_image


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def rgb_sample_image() -> RasterImage:
    """RGB image with all 3 channels."""
    data = np.zeros((30, 30, 3), dtype=np.float32)
    data[:, :, 0] = 50.0   # Red
    data[:, :, 1] = 180.0  # Green
    data[:, :, 2] = 200.0  # Blue
    return RasterImage(
        data=data,
        bands=["red", "green", "blue"],
        sensor_type=SensorType.RGB,
        dtype="float32",
    )


@pytest.fixture
def green_only_image() -> RasterImage:
    """Single-band green (B03) image without NIR or full RGB."""
    data = np.full((30, 30, 1), 1500.0, dtype=np.float32)
    return RasterImage(
        data=data,
        bands=["green"],
        sensor_type=SensorType.MULTISPECTRAL,
        dtype="float32",
    )


@pytest.fixture
def nir_only_image() -> RasterImage:
    """Single-band NIR (B08) image."""
    data = np.full((30, 30, 1), 4000.0, dtype=np.float32)
    return RasterImage(
        data=data,
        bands=["nir"],
        sensor_type=SensorType.MULTISPECTRAL,
        dtype="float32",
    )


@pytest.fixture
def red_only_image() -> RasterImage:
    """Single-band Red (B04) image."""
    data = np.full((30, 30, 1), 1200.0, dtype=np.float32)
    return RasterImage(
        data=data,
        bands=["red"],
        sensor_type=SensorType.MULTISPECTRAL,
        dtype="float32",
    )


@pytest.fixture
def multispectral_full_image() -> RasterImage:
    """Multispectral image with Red, Green, Blue, NIR, SWIR."""
    data = np.zeros((30, 30, 5), dtype=np.float32)
    data[:, :, 0] = 1000.0  # Blue
    data[:, :, 1] = 1200.0  # Green
    data[:, :, 2] = 1100.0  # Red
    data[:, :, 3] = 4500.0  # NIR
    data[:, :, 4] = 800.0   # SWIR
    return RasterImage(
        data=data,
        bands=["blue", "green", "red", "nir", "swir1"],
        sensor_type=SensorType.MULTISPECTRAL,
        dtype="float32",
    )


# =========================================================================
# Part 1 & 2: Spectral & Water/Vegetation Routing Tests
# =========================================================================

def test_1_rgb_image_generic_water_detection(rgb_sample_image):
    """1. RGB image + generic water detection → rgb_water_proxy()."""
    res = detect_water(rgb_sample_image, query="Find water bodies in the image")
    assert res.metadata["method"] == "rgb_water_proxy"
    assert "RGB Water Proxy" in res.index_name
    assert res.evidence is not None


def test_2_green_and_nir_ndwi_request(multispectral_full_image):
    """2. Green + NIR + NDWI request → true NDWI (never rgb_water_proxy)."""
    res = detect_water(multispectral_full_image, query="Calculate the true NDWI")
    assert res.metadata["method"] == "true_ndwi"
    assert "NDWI (McFeeters" in res.index_name
    assert "water_detection (NDWI)" in res.tool_used


def test_2b_dual_band_geotiff_ndwi(green_only_image, nir_only_image):
    """2b. Separate Green (B03) + NIR (B08) GeoTIFFs → true NDWI."""
    res = detect_water(green_only_image, nir_only_image, query="Calculate NDWI")
    assert res.metadata["method"] == "true_ndwi"
    assert "Dual Band GeoTIFF" in res.index_name


def test_3_green_only_ndwi_request_no_crash(green_only_image):
    """3. Green-only + NDWI request → clear missing-NIR response, no exception."""
    res = detect_water(green_only_image, query="Calculate NDWI")
    assert res.metadata["method"] == "missing_bands"
    assert "requires Green (B03) and NIR (B08)" in res.answer
    assert "missing" in res.answer.lower()
    assert res.evidence is None


def test_3b_green_only_generic_water_request_no_crash(green_only_image):
    """3b. Green-only + generic water request → clear explanation without crashing on RGB proxy."""
    res = detect_water(green_only_image, query="Find water")
    assert res.metadata["method"] == "missing_bands"
    assert "cannot be performed with the available data" in res.answer


def test_4_red_and_nir_ndvi_request(multispectral_full_image):
    """4. Red + NIR + NDVI request → true NDVI."""
    res = detect_vegetation(multispectral_full_image, query="Calculate NDVI")
    assert res.metadata["method"] == "true_ndvi"
    assert "NDVI (Rouse" in res.index_name
    assert "vegetation_detection (NDVI)" in res.tool_used


def test_4b_dual_band_geotiff_ndvi(red_only_image, nir_only_image):
    """4b. Separate Red (B04) + NIR (B08) GeoTIFFs → true NDVI."""
    res = detect_vegetation(red_only_image, nir_only_image, query="Calculate NDVI")
    assert res.metadata["method"] == "true_ndvi"
    assert "Dual Band GeoTIFF" in res.index_name


def test_5_rgb_image_ndvi_request_without_nir(rgb_sample_image):
    """5. RGB image + explicit NDVI request without NIR → clear missing-NIR response."""
    res = detect_vegetation(rgb_sample_image, query="Calculate true NDVI for this scene")
    assert res.metadata["method"] == "missing_bands"
    assert "True NDVI cannot be calculated" in res.answer
    assert "NIR (B08) data are required" in res.answer


def test_5b_rgb_image_generic_vegetation_proxy(rgb_sample_image):
    """5b. RGB image + generic vegetation request → RGB greenness proxy."""
    res = detect_vegetation(rgb_sample_image, query="Find green trees and vegetation")
    assert res.metadata["method"] == "rgb_greenness"
    assert "RGB Greenness Proxy" in res.index_name


# =========================================================================
# Part 4 & 5: GeoChat Output Cleaning Tests
# =========================================================================

def test_6_geochat_malformed_tokenizer_output():
    """6. GeoChat malformed tokenizer output → clean readable English."""
    bad_output = (
        "In the image , there are some buildings { < 4 8 >< 5 3 >< 5 2 >< 5 7 > | < 9 0 > } "
        "< del im >{ < 4 8 >< 4 9 >< 5 2 >< 5 3 > | < 9 0 > } located close to each other at the center of the scene ."
    )
    cleaned = clean_vlm_response(bad_output)
    assert "{" not in cleaned
    assert "}" not in cleaned
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "del im" not in cleaned
    assert "In the image, there are some buildings located close to each other at the center of the scene." == cleaned


def test_7_geochat_normal_output_unchanged():
    """7. GeoChat normal output → remains unchanged except harmless whitespace normalization."""
    normal_text = "The satellite image shows an agricultural area with rectangular fields and irrigation canals."
    cleaned = clean_vlm_response(normal_text)
    assert cleaned == normal_text


def test_8_geochat_special_tokens_removed():
    """8. GeoChat special tokens (<s>, </s>, <unk>, <image>, ▁) removed."""
    dirty_text = "<s> <image> ▁The port contains several cargo vessels docked at the pier. </s> <unk>"
    cleaned = clean_vlm_response(dirty_text)
    assert cleaned == "The port contains several cargo vessels docked at the pier."


# =========================================================================
# Part 9 & 10: Streamlit TIFF Display & Data Integrity Tests
# =========================================================================

class MockUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._buf = io.BytesIO(data)

    def read(self, *args):
        return self._buf.read(*args)

    def seek(self, pos: int):
        self._buf.seek(pos)


def test_9_streamlit_tiff_display_no_mode_i_error():
    """9. Streamlit TIFF display: scientific Mode I TIFF displayed without 'OSError: cannot write mode I as JPEG'."""
    raw_16bit = np.random.randint(500, 5000, size=(64, 64), dtype=np.uint16)
    img_mode_i = Image.fromarray(raw_16bit, mode="I;16")
    buf = io.BytesIO()
    img_mode_i.save(buf, format="TIFF")
    tiff_bytes = buf.getvalue()

    upload = MockUploadedFile("HLS.S30.T18TYN.B04.tif", tiff_bytes)
    disp = get_display_image(upload)

    assert disp is not None
    assert disp.mode == "RGB"

    # Must be serializable as JPEG for Streamlit frontend without OSError
    jpeg_buf = io.BytesIO()
    disp.save(jpeg_buf, format="JPEG")
    assert len(jpeg_buf.getvalue()) > 0


def test_10_scientific_tiff_original_values_preserved():
    """10. Scientific TIFF original values: remain untouched for calculations."""
    exact_values = np.array([[1234, 5678], [9101, 4321]], dtype=np.uint16)
    img = Image.fromarray(exact_values, mode="I;16")
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    tiff_bytes = buf.getvalue()

    raster = load_from_bytes(tiff_bytes, filename="B04_exact.tif")
    assert raster.data.shape == (2, 2, 1)
    assert float(raster.data[0, 0, 0]) == 1234.0
    assert float(raster.data[0, 1, 0]) == 5678.0
    assert float(raster.data[1, 0, 0]) == 9101.0
    assert float(raster.data[1, 1, 0]) == 4321.0
