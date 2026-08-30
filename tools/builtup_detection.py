"""
Built-up / urban area detection tool.

Uses true NDBI when SWIR + NIR bands are available,
otherwise falls back to an RGB color proxy.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from core.models import AnalysisResult, RasterImage
from tools.spectral import (
    can_compute_ndbi,
    calculate_ndbi,
    rgb_urban_proxy,
    create_mask,
)
from evidence.engine import create_evidence_image

logger = logging.getLogger(__name__)

NDBI_THRESHOLD = 0.0
RGB_URBAN_THRESHOLD = 0.1


def detect_builtup(image: RasterImage) -> AnalysisResult:
    """
    Detect built-up / urban areas.

    For multispectral images with SWIR + NIR bands:
        Uses true NDBI (Zha et al. 2003).
    For RGB-only images:
        Uses a color-based urban proxy.
    """
    logger.info(f"Built-up detection: sensor_type={image.sensor_type}, bands={image.bands}")

    if can_compute_ndbi(image):
        index = calculate_ndbi(image)
        mask = create_mask(index, threshold=NDBI_THRESHOLD, above=True)
        index_name = "NDBI (Zha et al. 2003)"
        method = "true_ndbi"
        answer_text = _format_builtup_answer(mask, image, method="NDBI")
        tool_name = "builtup_detection (NDBI)"
    else:
        index = rgb_urban_proxy(image)
        mask = create_mask(index, threshold=RGB_URBAN_THRESHOLD, above=True)
        index_name = "RGB Urban Proxy (heuristic — NOT NDBI)"
        method = "rgb_urban"
        answer_text = _format_builtup_answer(mask, image, method="RGB proxy")
        tool_name = "builtup_detection (RGB proxy)"

    rgb = image.to_rgb()
    evidence = create_evidence_image(rgb, mask, color=(200, 50, 50), alpha=0.5)

    total_pixels = mask.size
    builtup_pixels = int(np.sum(mask > 0))
    coverage_pct = (builtup_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

    return AnalysisResult(
        answer=answer_text,
        evidence=evidence,
        mask=mask,
        index_name=index_name,
        confidence=None,
        tool_used=tool_name,
        metadata={
            "method": method,
            "builtup_pixels": builtup_pixels,
            "total_pixels": total_pixels,
            "coverage_percent": round(coverage_pct, 2),
            "threshold": NDBI_THRESHOLD if method == "true_ndbi" else RGB_URBAN_THRESHOLD,
            "requires_multispectral": method != "true_ndbi",
        },
    )


def _format_builtup_answer(mask: np.ndarray, image: RasterImage, method: str) -> str:
    total_pixels = mask.size
    builtup_pixels = int(np.sum(mask > 0))
    coverage = (builtup_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

    if method == "NDBI":
        return (
            f"Built-up area detection using NDBI (Zha et al. 2003).\n\n"
            f"Built-up coverage: {coverage:.1f}% of image area "
            f"({builtup_pixels:,} of {total_pixels:,} pixels).\n\n"
            f"Threshold: > {NDBI_THRESHOLD:.2f}\n"
            f"Method: True spectral index using SWIR and NIR bands."
        )
    else:
        return (
            f"Built-up area detection using RGB color heuristic (visual proxy).\n\n"
            f"Estimated urban-like area: {coverage:.1f}% of image area "
            f"({builtup_pixels:,} of {total_pixels:,} pixels).\n\n"
            f"⚠️ This is NOT a true NDBI calculation. "
            f"True NDBI requires SWIR and NIR spectral data. "
            f"The current image has bands: {image.bands}. "
            f"This result is a color-based approximation only."
        )
