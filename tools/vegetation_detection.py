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


def detect_vegetation(image: RasterImage) -> AnalysisResult:
    """
    Detect vegetation in the image.

    For multispectral images with NIR + Red bands:
        Uses true NDVI (Rouse et al. 1974).
    For RGB-only images:
        Uses a greenness color proxy (clearly labeled as heuristic).
    """
    logger.info(f"Vegetation detection: sensor_type={image.sensor_type}, bands={image.bands}")

    if can_compute_ndvi(image):
        # True NDVI
        index = calculate_ndvi(image)
        mask = create_mask(index, threshold=NDVI_VEGETATION_THRESHOLD, above=True)
        index_name = "NDVI (Rouse et al. 1974)"
        method = "true_ndvi"
        answer_text = _format_vegetation_answer(mask, image, method="NDVI")
        tool_name = "vegetation_detection (NDVI)"
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
