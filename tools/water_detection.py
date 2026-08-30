"""
Water detection tool.

Uses true NDWI when multispectral bands are available,
otherwise falls back to an RGB color-based water proxy.
"""

from __future__ import annotations

import logging

import numpy as np

from core.models import AnalysisResult, RasterImage
from tools.spectral import (
    can_compute_ndwi,
    calculate_ndwi,
    rgb_water_proxy,
    create_mask,
)
from evidence.engine import create_evidence_image

logger = logging.getLogger(__name__)

# McFeeters NDWI threshold for water (commonly used)
NDWI_WATER_THRESHOLD = 0.0
# RGB proxy threshold (heuristic)
RGB_WATER_THRESHOLD = 0.1


def detect_water(image: RasterImage, image2: RasterImage | None = None) -> AnalysisResult:
    """
    Detect water bodies in the image.

    For multispectral images with green + NIR bands (or separate B03/B08 GeoTIFF uploads):
        Uses true NDWI (McFeeters 1996).
    For RGB-only images:
        Uses a color-based water proxy (clearly labeled as heuristic).
    """
    logger.info(f"Water detection: sensor_type={image.sensor_type}, bands={image.bands}")

    # Check single image with both Green and NIR
    if can_compute_ndwi(image):
        index = calculate_ndwi(image)
        mask = create_mask(index, threshold=NDWI_WATER_THRESHOLD, above=True)
        index_name = "NDWI (McFeeters 1996)"
        method = "true_ndwi"
        answer_text = _format_water_answer(mask, image, method="NDWI")
        tool_name = "water_detection (NDWI)"
    # Check separate GeoTIFF band uploads (e.g. B03.tif as image1 and B08.tif as image2)
    elif image2 is not None and (
        (image.has_band("green") and image2.has_band("nir"))
        or (image.has_band("nir") and image2.has_band("green"))
    ):
        green_band = image.get_band("green") if image.has_band("green") else image2.get_band("green")
        nir_band = image2.get_band("nir") if image2.has_band("nir") else image.get_band("nir")
        if green_band is not None and nir_band is not None and green_band.shape == nir_band.shape:
            from tools.spectral import _normalized_index
            index = _normalized_index(green_band, nir_band, image.nodata)
            mask = create_mask(index, threshold=NDWI_WATER_THRESHOLD, above=True)
            index_name = "NDWI (McFeeters 1996 — Dual Band GeoTIFF)"
            method = "true_ndwi"
            answer_text = _format_water_answer(mask, image, method="NDWI")
            tool_name = "water_detection (NDWI)"
        else:
            index = rgb_water_proxy(image)
            mask = create_mask(index, threshold=RGB_WATER_THRESHOLD, above=True)
            index_name = "RGB Water Proxy (heuristic — NOT NDWI)"
            method = "rgb_water_proxy"
            answer_text = _format_water_answer(mask, image, method="RGB water proxy")
            tool_name = "water_detection (RGB proxy)"
    else:
        # RGB proxy
        index = rgb_water_proxy(image)
        mask = create_mask(index, threshold=RGB_WATER_THRESHOLD, above=True)
        index_name = "RGB Water Proxy (heuristic — NOT NDWI)"
        method = "rgb_water_proxy"
        answer_text = _format_water_answer(mask, image, method="RGB water proxy")
        tool_name = "water_detection (RGB proxy)"

    # Generate evidence visualization
    rgb = image.to_rgb()
    evidence = create_evidence_image(rgb, mask, color=(0, 100, 255), alpha=0.5)

    # Compute statistics
    total_pixels = mask.size
    water_pixels = int(np.sum(mask > 0))
    coverage_pct = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

    return AnalysisResult(
        answer=answer_text,
        evidence=evidence,
        mask=mask,
        index_name=index_name,
        confidence=None,  # Not fabricated
        tool_used=tool_name,
        metadata={
            "method": method,
            "water_pixels": water_pixels,
            "total_pixels": total_pixels,
            "coverage_percent": round(coverage_pct, 2),
            "threshold": NDWI_WATER_THRESHOLD if method == "true_ndwi" else RGB_WATER_THRESHOLD,
            "requires_multispectral": method != "true_ndwi",
        },
    )


def _format_water_answer(mask: np.ndarray, image: RasterImage, method: str) -> str:
    total_pixels = mask.size
    water_pixels = int(np.sum(mask > 0))
    coverage = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

    if method == "NDWI":
        return (
            f"Water detection using NDWI (McFeeters 1996).\n\n"
            f"Water coverage: {coverage:.1f}% of image area "
            f"({water_pixels:,} of {total_pixels:,} pixels).\n\n"
            f"Threshold: > {NDWI_WATER_THRESHOLD:.2f}\n"
            f"Method: True spectral index using Green and NIR bands."
        )
    else:
        return (
            f"Water detection using RGB color heuristic (visual proxy).\n\n"
            f"Estimated water-like area: {coverage:.1f}% of image area "
            f"({water_pixels:,} of {total_pixels:,} pixels).\n\n"
            f"⚠️ This is NOT a true NDWI calculation. "
            f"True NDWI requires near-infrared (NIR) spectral data. "
            f"The current image has bands: {image.bands}. "
            f"This result is a color-based approximation only."
        )
