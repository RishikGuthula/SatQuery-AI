"""
Vegetation detection tool.

Uses true NDVI when multispectral bands are available,
otherwise falls back to an RGB greenness proxy (for generic visual requests only).
When required bands are missing or explicit NDVI cannot be computed, returns a clear explanation.
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


def is_explicit_ndvi_request(query: str) -> bool:
    """Check if the user explicitly requested NDVI or spectral vegetation index."""
    q = query.lower()
    keywords = [
        "ndvi",
        "true ndvi",
        "vegetation index",
        "vegetation spectral index",
        "spectral vegetation",
        "spectral index",
        "b04",
        "b08",
        "band 4",
        "band 8",
    ]
    return any(k in q for k in keywords)


def _has_red_and_green(image: RasterImage) -> bool:
    """Check if image has red and green bands needed for RGB greenness proxy."""
    return image.has_band("red") and image.has_band("green")


def detect_vegetation(
    image: RasterImage,
    image2: RasterImage | None = None,
    query: str = "",
) -> AnalysisResult:
    """
    Detect vegetation in the image.

    Routing logic:
    1. If Red (B04) + NIR (B08) are available (single or dual image):
       → Compute TRUE NDVI (Rouse et al. 1974).
    2. If Red + NIR are NOT available:
       a. If user explicitly requested NDVI/spectral index:
          → Do NOT call rgb_greenness_index(). Return clear missing-NIR/Red explanation.
       b. If generic visual request AND Red + Green bands exist:
          → Compute RGB greenness proxy (clearly labeled as heuristic).
       c. Otherwise:
          → Return clear missing-band explanation (never crash).
    """
    logger.info(f"Vegetation detection: sensor_type={image.sensor_type}, bands={image.bands}, query='{query}'")

    # --- Route 1: Single image with both NIR and Red ---
    if can_compute_ndvi(image):
        index = calculate_ndvi(image)
        mask = create_mask(index, threshold=NDVI_VEGETATION_THRESHOLD, above=True)
        return _build_result(
            mask,
            image,
            index_name="NDVI (Rouse et al. 1974)",
            method="true_ndvi",
            tool_name="vegetation_detection (NDVI)",
            answer_text=_format_vegetation_answer(mask, image, method="NDVI"),
        )

    # --- Route 2: Separate GeoTIFF band uploads (e.g. B04.tif + B08.tif) ---
    if image2 is not None:
        combined_has_red = image.has_band("red") or image2.has_band("red")
        combined_has_nir = image.has_band("nir") or image2.has_band("nir")
        if combined_has_red and combined_has_nir:
            red_band = (
                image.get_band("red")
                if image.has_band("red")
                else image2.get_band("red")
            )
            nir_band = (
                image2.get_band("nir")
                if image2.has_band("nir")
                else image.get_band("nir")
            )
            if (
                red_band is not None
                and nir_band is not None
                and red_band.shape == nir_band.shape
            ):
                from tools.spectral import _normalized_index

                index = _normalized_index(nir_band, red_band, image.nodata)
                mask = create_mask(index, threshold=NDVI_VEGETATION_THRESHOLD, above=True)
                return _build_result(
                    mask,
                    image,
                    index_name="NDVI (Rouse et al. 1974 — Dual Band GeoTIFF)",
                    method="true_ndvi",
                    tool_name="vegetation_detection (NDVI)",
                    answer_text=_format_vegetation_answer(mask, image, method="NDVI"),
                )

    available = ", ".join(image.bands) if image.bands else "none"

    # --- Route 3: Explicit NDVI requested, but required bands missing ---
    if is_explicit_ndvi_request(query):
        missing = []
        if not image.has_band("red"):
            missing.append("Red (B04)")
        if not image.has_band("nir"):
            missing.append("NIR (B08)")
        missing_str = " and ".join(missing) if missing else "NIR (B08)"

        answer = (
            f"⚠️ True NDVI cannot be calculated from this image because Red (B04) and NIR (B08) data are required.\n\n"
            f"The uploaded data does not contain all required bands ({missing_str} is missing).\n"
            f"**Available bands:** {available}\n\n"
            "Please upload both Red (B04) and NIR (B08) GeoTIFFs to calculate true NDVI."
        )
        return AnalysisResult(
            answer=answer,
            evidence=None,
            mask=None,
            index_name=None,
            confidence=None,
            tool_used="vegetation_detection (missing_bands)",
            metadata={
                "method": "missing_bands",
                "available_bands": image.bands,
                "missing_bands": missing,
                "error": "insufficient_spectral_bands_for_ndvi",
            },
        )

    # --- Route 4: Generic visual request with Red + Green available ---
    if _has_red_and_green(image):
        index = rgb_greenness_index(image)
        mask = create_mask(index, threshold=RGB_GREENNESS_THRESHOLD, above=True)
        return _build_result(
            mask,
            image,
            index_name="RGB Greenness Proxy (heuristic — NOT NDVI)",
            method="rgb_greenness",
            tool_name="vegetation_detection (RGB proxy)",
            answer_text=_format_vegetation_answer(mask, image, method="RGB greenness"),
        )

    # --- Route 5: Generic request with insufficient bands ---
    return AnalysisResult(
        answer=(
            f"⚠️ Vegetation detection cannot be performed with the available data.\n\n"
            f"**Available bands:** {available}\n\n"
            "**True NDVI** requires Red (B04) and NIR (B08) bands.\n"
            "**RGB greenness proxy** requires Red and Green bands.\n\n"
            "Please upload either a standard RGB image or both Red and NIR multispectral bands."
        ),
        evidence=None,
        mask=None,
        index_name=None,
        confidence=None,
        tool_used="vegetation_detection (missing_bands)",
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
    evidence = create_evidence_image(rgb, mask, color=(0, 180, 0), alpha=0.5)

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
            "threshold": (
                NDVI_VEGETATION_THRESHOLD
                if method == "true_ndvi"
                else RGB_GREENNESS_THRESHOLD
            ),
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
