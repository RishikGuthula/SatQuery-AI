"""
Water detection tool.

Uses true NDWI when multispectral bands are available,
otherwise falls back to an RGB color-based water proxy (for generic visual requests only).
When required bands are missing or explicit NDWI cannot be computed, returns a clear explanation.
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


def is_explicit_ndwi_request(query: str) -> bool:
    """Check if the user explicitly requested NDWI or spectral water index."""
    q = query.lower()
    keywords = [
        "ndwi",
        "true ndwi",
        "water index",
        "water spectral index",
        "spectral water",
        "spectral index",
        "b03",
        "b08",
        "band 3",
        "band 8",
    ]
    return any(k in q for k in keywords)


def _has_rgb(image: RasterImage) -> bool:
    """Check if image has all three RGB bands needed for RGB water proxy."""
    return (
        image.has_band("red")
        and image.has_band("green")
        and image.has_band("blue")
    )


def detect_water(
    image: RasterImage,
    image2: RasterImage | None = None,
    query: str = "",
) -> AnalysisResult:
    """
    Detect water bodies in the image.

    Routing logic:
    1. If Green (B03) + NIR (B08) are available (single or dual image):
       → Compute TRUE NDWI (McFeeters 1996).
    2. If Green + NIR are NOT available:
       a. If user explicitly requested NDWI/spectral index:
          → Do NOT call rgb_water_proxy(). Return clear missing-NIR/Green explanation.
       b. If generic visual request AND RGB bands exist:
          → Compute RGB water proxy (clearly labeled as heuristic).
       c. Otherwise:
          → Return clear missing-band explanation (never crash).
    """
    logger.info(f"Water detection: sensor_type={image.sensor_type}, bands={image.bands}, query='{query}'")

    # --- Route 1: Single image with both Green and NIR ---
    if can_compute_ndwi(image):
        index = calculate_ndwi(image)
        mask = create_mask(index, threshold=NDWI_WATER_THRESHOLD, above=True)
        return _build_result(
            mask,
            image,
            index_name="NDWI (McFeeters 1996)",
            method="true_ndwi",
            tool_name="water_detection (NDWI)",
            answer_text=_format_water_answer(mask, image, method="NDWI"),
        )

    # --- Route 2: Separate GeoTIFF band uploads (e.g. B03.tif + B08.tif) ---
    if image2 is not None:
        combined_has_green = image.has_band("green") or image2.has_band("green")
        combined_has_nir = image.has_band("nir") or image2.has_band("nir")
        if combined_has_green and combined_has_nir:
            green_band = (
                image.get_band("green")
                if image.has_band("green")
                else image2.get_band("green")
            )
            nir_band = (
                image2.get_band("nir")
                if image2.has_band("nir")
                else image.get_band("nir")
            )
            if (
                green_band is not None
                and nir_band is not None
                and green_band.shape == nir_band.shape
            ):
                from tools.spectral import _normalized_index

                index = _normalized_index(green_band, nir_band, image.nodata)
                mask = create_mask(index, threshold=NDWI_WATER_THRESHOLD, above=True)
                return _build_result(
                    mask,
                    image,
                    index_name="NDWI (McFeeters 1996 — Dual Band GeoTIFF)",
                    method="true_ndwi",
                    tool_name="water_detection (NDWI)",
                    answer_text=_format_water_answer(mask, image, method="NDWI"),
                )

    available = ", ".join(image.bands) if image.bands else "none"

    # --- Route 3: Explicit NDWI requested, but required bands missing ---
    if is_explicit_ndwi_request(query):
        missing = []
        if not image.has_band("green"):
            missing.append("Green (B03)")
        if not image.has_band("nir"):
            missing.append("NIR (B08)")
        missing_str = " and ".join(missing) if missing else "NIR (B08)"

        answer = (
            f"⚠️ True NDWI requires Green (B03) and NIR (B08) bands.\n\n"
            f"The uploaded data does not contain all required bands ({missing_str} is missing).\n"
            f"**Available bands:** {available}\n\n"
            "Please upload both Green (B03) and NIR (B08) GeoTIFFs to calculate true NDWI."
        )
        return AnalysisResult(
            answer=answer,
            evidence=None,
            mask=None,
            index_name=None,
            confidence=None,
            tool_used="water_detection (missing_bands)",
            metadata={
                "method": "missing_bands",
                "available_bands": image.bands,
                "missing_bands": missing,
                "error": "insufficient_spectral_bands_for_ndwi",
            },
        )

    # --- Route 4: Generic visual request with RGB available ---
    if _has_rgb(image):
        index = rgb_water_proxy(image)
        mask = create_mask(index, threshold=RGB_WATER_THRESHOLD, above=True)
        return _build_result(
            mask,
            image,
            index_name="RGB Water Proxy (heuristic — NOT NDWI)",
            method="rgb_water_proxy",
            tool_name="water_detection (RGB proxy)",
            answer_text=_format_water_answer(mask, image, method="RGB water proxy"),
        )

    # --- Route 5: Generic request with insufficient bands (e.g. single-band B03 without RGB or NIR) ---
    return AnalysisResult(
        answer=(
            f"⚠️ Water detection cannot be performed with the available data.\n\n"
            f"**Available bands:** {available}\n\n"
            "**True NDWI** requires Green (B03) and NIR (B08) bands.\n"
            "**RGB water proxy** requires Red, Green, and Blue bands.\n\n"
            "Please upload either a standard RGB image or both Green and NIR multispectral bands."
        ),
        evidence=None,
        mask=None,
        index_name=None,
        confidence=None,
        tool_used="water_detection (missing_bands)",
        metadata={
            "method": "missing_bands",
            "available_bands": image.bands,
            "error": "insufficient_spectral_bands",
        },
    )


def _build_result(
    mask: np.ndarray,
    image: RasterImage,
    index_name: str,
    method: str,
    tool_name: str,
    answer_text: str,
) -> AnalysisResult:
    """Build a complete AnalysisResult with evidence visualization."""
    rgb = image.to_rgb()
    evidence = create_evidence_image(rgb, mask, color=(0, 100, 255), alpha=0.5)

    total_pixels = mask.size
    water_pixels = int(np.sum(mask > 0))
    coverage_pct = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

    return AnalysisResult(
        answer=answer_text,
        evidence=evidence,
        mask=mask,
        index_name=index_name,
        confidence=None,
        tool_used=tool_name,
        metadata={
            "method": method,
            "water_pixels": water_pixels,
            "total_pixels": total_pixels,
            "coverage_percent": round(coverage_pct, 2),
            "threshold": (
                NDWI_WATER_THRESHOLD
                if method == "true_ndwi"
                else RGB_WATER_THRESHOLD
            ),
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
