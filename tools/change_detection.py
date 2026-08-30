"""
Change detection tool.

Implements a robust change-detection pipeline for two co-registered
or auto-aligned images. Supports:
  - Geospatial metadata alignment (when CRS/transform exist)
  - Dimension normalization fallback
  - Normalized color-space Euclidean difference mapping
  - Adaptive percentile thresholding
  - Visual evidence and grounded metrics
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from core.models import AnalysisResult, RasterImage, SensorType

logger = logging.getLogger(__name__)


class ChangeDetectionError(Exception):
    """Raised when change detection cannot be performed."""
    pass


def validate_pair(img_a: RasterImage, img_b: RasterImage) -> None:
    """
    Validate that two images are compatible for change detection.
    Raises ChangeDetectionError if incompatible.
    """
    if img_a.sensor_type == SensorType.SAR and img_b.sensor_type != SensorType.SAR:
        raise ChangeDetectionError(
            "Cannot compare SAR imagery with optical imagery. "
            "Both images must be the same modality."
        )
    if img_b.sensor_type == SensorType.SAR and img_a.sensor_type != SensorType.SAR:
        raise ChangeDetectionError(
            "Cannot compare optical imagery with SAR imagery. "
            "Both images must be the same modality."
        )
    if img_a.data.size == 0 or img_b.data.size == 0:
        raise ChangeDetectionError("One or both images are empty.")


def _align_pair(img_a: RasterImage, img_b: RasterImage) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Align two images using geospatial reprojection if rasterio/CRS is available,
    otherwise resize to matching dimensions.
    """
    # Check for geospatial metadata
    if img_a.crs and img_b.crs and img_a.transform and img_b.transform:
        try:
            import rasterio.warp
            # If CRS or transforms differ, reproject b to match a
            logger.info("Aligning images using geospatial reprojection.")
            rgb_a = img_a.to_rgb().astype(np.float32)
            rgb_b = img_b.to_rgb().astype(np.float32)

            dest = np.zeros_like(rgb_a)
            for c in range(3):
                rasterio.warp.reproject(
                    source=rgb_b[:, :, c],
                    destination=dest[:, :, c],
                    src_transform=img_b.transform,
                    src_crs=img_b.crs,
                    dst_transform=img_a.transform,
                    dst_crs=img_a.crs,
                    resampling=rasterio.warp.Resampling.bilinear,
                )
            return rgb_a, dest, "geospatial_reproject"
        except Exception as e:
            logger.debug(f"Geospatial reprojection fallback to resize: {e}")

    # Fallback to dimension alignment
    rgb_a = img_a.to_rgb().astype(np.float32)
    rgb_b = img_b.to_rgb().astype(np.float32)
    ra, rb = _resize_to_match(rgb_a, rgb_b)
    return ra, rb, "bilinear_resize"


def _resize_to_match(arr_a: np.ndarray, arr_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize both arrays to the smaller of the two dimensions using PIL."""
    h_a, w_a = arr_a.shape[:2]
    h_b, w_b = arr_b.shape[:2]

    if h_a == h_b and w_a == w_b:
        return arr_a, arr_b

    # Target: minimum dimensions
    target_h = min(h_a, h_b)
    target_w = min(w_a, w_b)

    def _resize(arr, th, tw):
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr, mode="L")
        else:
            pil_img = Image.fromarray(arr.astype(np.uint8) if arr.dtype != np.uint8 else arr)
        pil_img = pil_img.resize((tw, th), Image.BILINEAR)
        return np.array(pil_img).astype(np.float32)

    logger.info(f"Resizing images from ({h_a}x{w_a}, {h_b}x{w_b}) to ({target_h}x{target_w})")
    return _resize(arr_a, target_h, target_w), _resize(arr_b, target_h, target_w)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    amin, amax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if amax - amin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - amin) / (amax - amin)).astype(np.float32)


def detect_changes(
    image_a: RasterImage,
    image_b: RasterImage,
    threshold_percentile: float = 85.0,
) -> AnalysisResult:
    """
    Detect changes between two images.
    """
    logger.info("Starting change detection pipeline.")

    # Step 1: Validate
    validate_pair(image_a, image_b)

    # Step 2: Align images (geospatial or resize)
    rgb_a, rgb_b, alignment_method = _align_pair(image_a, image_b)

    # Step 3: Normalize
    norm_a = _normalize(rgb_a)
    norm_b = _normalize(rgb_b)

    # Step 4: Compute Euclidean difference in color space
    diff = np.sqrt(np.sum((norm_a - norm_b) ** 2, axis=-1))
    diff = _normalize(diff)

    # Step 5: Adaptive threshold using percentile
    valid_diff = diff[~np.isnan(diff)]
    if len(valid_diff) == 0:
        raise ChangeDetectionError("Difference map is entirely NaN.")

    threshold = float(np.percentile(valid_diff, threshold_percentile))
    mask = (diff > threshold).astype(np.uint8) * 255

    # Step 6: Metrics
    total_pixels = mask.size
    changed_pixels = int(np.sum(mask > 0))
    changed_pct = (changed_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
    mean_diff = float(np.nanmean(diff))

    # Step 7: Visual evidence overlay
    evidence = _create_change_evidence(norm_a, diff, mask)

    answer = (
        f"Change detection between two images.\n\n"
        f"Detected changed area: {changed_pct:.1f}% of image "
        f"({changed_pixels:,} of {total_pixels:,} pixels).\n"
        f"Mean difference: {mean_diff:.4f}\n"
        f"Threshold (p{threshold_percentile:.0f}): {threshold:.4f}\n"
        f"Alignment method: {alignment_method}.\n\n"
        f"Red regions in the visual evidence map highlight detected changes."
    )

    return AnalysisResult(
        answer=answer,
        evidence=evidence,
        mask=mask,
        index_name="Change Magnitude",
        confidence=None,
        tool_used="change_detection (pixel-difference)",
        metadata={
            "method": "color_euclidean_distance",
            "alignment_method": alignment_method,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "changed_percent": round(changed_pct, 2),
            "mean_difference": round(mean_diff, 4),
            "threshold": round(threshold, 4),
            "threshold_percentile": threshold_percentile,
            "input_a_size": (image_a.height, image_a.width),
            "input_b_size": (image_b.height, image_b.width),
        },
    )


def _create_change_evidence(
    base_rgb: np.ndarray,
    diff_map: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    """Create a change-detection evidence image."""
    base_uint8 = (base_rgb * 255).clip(0, 255).astype(np.uint8)
    if base_uint8.ndim == 2:
        base_uint8 = np.stack([base_uint8] * 3, axis=-1)

    overlay = base_uint8.copy()
    changed = mask > 0

    # Dim unchanged areas
    overlay[~changed] = (overlay[~changed] * 0.4).astype(np.uint8)
    # Color changed areas red
    overlay[changed, 0] = 255
    overlay[changed, 1] = 50
    overlay[changed, 2] = 50

    return Image.fromarray(overlay)
