"""
Vegetation detection tool.

Uses true NDVI when multispectral bands are available,
otherwise falls back to an RGB greenness proxy.
"""

from __future__ import annotations

import logging

import numpy as np

from core.models import AnalysisResult, RasterImage
from tools.spectral import (
    can_compute_ndvi,
    calculate_ndvi,
    rgb_greenness_index,
    create_mask,
)
from evidence.engine import create_evidence_image

logger = logging.getLogger(__name__)

# NDVI threshold for vegetation (commonly used)
NDVI_VEGETATION_THRESHOLD = 0.2
# RGB proxy threshold (heuristic)
RGB_GREENNESS_THRESHOLD = 0.1


def detect_vegetation(image: RasterImage, image2: RasterImage | None = None) -> AnalysisResult:
    """
    Detect vegetation in the image.

    For multispectral images with NIR + Red bands (or separate B04/B08 GeoTIFF uploads):
        Uses true NDVI (Rouse et al. 1974).
    For RGB-only images:
        Uses a greenness color proxy (clearly labeled as heuristic).
    """
    logger.info(f"Vegetation detection: sensor_type={image.sensor_type}, bands={image.bands}")

    # Check single image with both NIR and Red
    if can_compute_ndvi(image):
        index = calculate_ndvi(image)
        mask = create_mask(index, threshold=NDVI_VEGETATION_THRESHOLD, above=True)
        index_name = "NDVI (Rouse et al. 1974)"
        method = "true_ndvi"
        answer_text = _format_vegetation_answer(mask, image, method="NDVI")
        tool_name = "vegetation_detection (NDVI)"
    # Check separate GeoTIFF band uploads (e.g. B04.tif as image1 and B08.tif as image2)
    elif image2 is not None and (
        (image.has_band("red") and image2.has_band("nir"))
        or (image.has_band("nir") and image2.has_band("red"))
    ):
        red_band = image.get_band("red") if image.has_band("red") else image2.get_band("red")
        nir_band = image2.get_band("nir") if image2.has_band("nir") else image.get_band("nir")
        if red_band is not None and nir_band is not None and red_band.shape == nir_band.shape:
            from tools.spectral import _normalized_index
            index = _normalized_index(nir_band, red_band, image.nodata)
            mask = create_mask(index, threshold=NDVI_VEGETATION_THRESHOLD, above=True)
            index_name = "NDVI (Rouse et al. 1974 — Dual Band GeoTIFF)"
            method = "true_ndvi"
            answer_text = _format_vegetation_answer(mask, image, method="NDVI")
            tool_name = "vegetation_detection (NDVI)"
        else:
            index = rgb_greenness_index(image)
            mask = create_mask(index, threshold=RGB_GREENNESS_THRESHOLD, above=True)
            index_name = "RGB Greenness Proxy (heuristic — NOT NDVI)"
            method = "rgb_greenness"
            answer_text = _format_vegetation_answer(mask, image, method="RGB greenness")
            tool_name = "vegetation_detection (RGB proxy)"
    else:
        # RGB proxy
        index = rgb_greenness_index(image)
        mask = create_mask(index, threshold=RGB_GREENNESS_THRESHOLD, above=True)
        index_name = "RGB Greenness Proxy (heuristic — NOT NDVI)"
        method = "rgb_greenness"
        answer_text = _format_vegetation_answer(mask, image, method="RGB greenness")
        tool_name = "vegetation_detection (RGB proxy)"

    # Generate evidence visualization
    rgb = image.to_rgb()
    evidence = create_evidence_image(rgb, mask, color=(0, 180, 0), alpha=0.5)

    # Statistics
    total_pixels = mask.size
    veg_pixels = int(np.sum(mask > 0))
    coverage_pct = (veg_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

    return AnalysisResult(
        answer=answer_text,
        evidence=evidence,
        mask=mask,
        index_name=index_name,
        confidence=None,
        tool_used=tool_name,
        metadata={
            "method": method,
            "vegetation_pixels": veg_pixels,
            "total_pixels": total_pixels,
            "coverage_percent": round(coverage_pct, 2),
            "threshold": NDVI_VEGETATION_THRESHOLD if method == "true_ndvi" else RGB_GREENNESS_THRESHOLD,
            "requires_multispectral": method != "true_ndvi",
        },
    )


def _format_vegetation_answer(mask: np.ndarray, image: RasterImage, method: str) -> str:
    total_pixels = mask.size
    veg_pixels = int(np.sum(mask > 0))
    coverage = (veg_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

    if method == "NDVI":
        return (
            f"Vegetation detection using NDVI (Rouse et al. 1974).\n\n"
            f"Vegetation coverage: {coverage:.1f}% of image area "
            f"({veg_pixels:,} of {total_pixels:,} pixels).\n\n"
            f"Threshold: > {NDVI_VEGETATION_THRESHOLD:.2f}\n"
            f"Method: True spectral index using NIR and Red bands."
        )
    else:
        return (
            f"Vegetation detection using RGB greenness heuristic (visual proxy).\n\n"
            f"Estimated green area: {coverage:.1f}% of image area "
            f"({veg_pixels:,} of {total_pixels:,} pixels).\n\n"
            f"⚠️ This is NOT a true NDVI calculation. "
            f"True NDVI requires near-infrared (NIR) spectral data. "
            f"The current image has bands: {image.bands}. "
            f"This result is a color-based approximation only."
        )
