"""
ChangeFormer Tool Interface for SatQuery AI.

Provides the high-level tool wrapper and bi-temporal validation for running
ChangeFormer neural change detection on satellite image pairs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from core.models import AnalysisResult, RasterImage, SensorType
from models.changeformer import ChangeFormerAdapter

logger = logging.getLogger(__name__)


class ChangeFormerError(Exception):
    """Raised when ChangeFormer change detection cannot be performed."""
    pass


# Global singleton adapter for reuse across requests
_adapter_instance: ChangeFormerAdapter | None = None


def get_changeformer_adapter() -> ChangeFormerAdapter:
    """Retrieve or initialize the global ChangeFormerAdapter singleton."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = ChangeFormerAdapter()
    return _adapter_instance


def validate_bitemporal_pair(img_a: RasterImage, img_b: RasterImage) -> None:
    """
    Validate that two images are compatible for ChangeFormer bi-temporal change detection.

    Raises ChangeFormerError with clear user-facing messages if invalid.
    """
    if img_a is None or img_b is None:
        raise ChangeFormerError("Both primary (T1) and secondary (T2) images are required for ChangeFormer change detection.")

    if img_a.data is None or img_a.data.size == 0:
        raise ChangeFormerError("Primary image (T1) contains no readable image data.")

    if img_b.data is None or img_b.data.size == 0:
        raise ChangeFormerError("Secondary image (T2) contains no readable image data.")

    # Modality check
    if img_a.sensor_type == SensorType.SAR and img_b.sensor_type != SensorType.SAR:
        raise ChangeFormerError("Cannot compare SAR imagery with optical imagery. Both images must share the same modality.")
    if img_b.sensor_type == SensorType.SAR and img_a.sensor_type != SensorType.SAR:
        raise ChangeFormerError("Cannot compare optical imagery with SAR imagery. Both images must share the same modality.")

    # Spatial dimension validation
    dim_a = (img_a.height, img_a.width)
    dim_b = (img_b.height, img_b.width)

    if dim_a != dim_b:
        # Check if geospatial alignment is available via CRS/transform metadata
        if img_a.crs and img_b.crs and img_a.transform and img_b.transform:
            logger.info("Spatial dimensions differ, but geospatial CRS/transform metadata is present for reprojection.")
        else:
            raise ChangeFormerError(
                f"Dimension mismatch for ChangeFormer analysis: Primary image (T1) is {dim_a[1]}x{dim_a[0]} px, "
                f"but secondary image (T2) is {dim_b[1]}x{dim_b[0]} px. "
                f"Bi-temporal images must have matching spatial dimensions."
            )


def create_change_evidence(
    base_rgb: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    """
    Create a visual overlay evidence image highlighting detected changes in red.
    """
    base = base_rgb.copy()
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    elif base.shape[2] == 1:
        base = np.concatenate([base] * 3, axis=-1)

    overlay = base.copy().astype(np.uint8)
    changed = mask > 0

    # Dim unchanged background pixels slightly for contrast
    overlay[~changed] = (overlay[~changed] * 0.5).astype(np.uint8)

    # Highlight changed pixels in vibrant red
    overlay[changed, 0] = 255
    overlay[changed, 1] = 40
    overlay[changed, 2] = 40

    return Image.fromarray(overlay)


def detect_changes_changeformer(
    image_a: RasterImage,
    image_b: RasterImage,
    query: str = "",
    adapter: ChangeFormerAdapter | None = None,
) -> AnalysisResult:
    """
    Run ChangeFormer bi-temporal change detection on image pair (T1, T2).

    Args:
        image_a: Primary / T1 (Before) image.
        image_b: Secondary / T2 (After) image.
        query: Optional user query string.
        adapter: Optional custom ChangeFormerAdapter instance (for testing).

    Returns:
        AnalysisResult containing natural language summary, visual evidence, mask, and metrics.
    """
    logger.info("Executing ChangeFormer bi-temporal change detection tool.")

    # Step 1: Validate input pair
    validate_bitemporal_pair(image_a, image_b)

    # Step 2: Acquire adapter & execute model inference
    changeformer = adapter or get_changeformer_adapter()
    pred = changeformer.predict(image_a, image_b)

    mask = pred["mask"]
    change_pct = pred["change_percentage"]
    changed_pixels = pred["changed_pixels"]
    total_pixels = pred["total_pixels"]
    confidence = pred["confidence"]
    device_name = pred["device"]

    # Step 3: Generate visual evidence map
    base_rgb = image_a.to_rgb()
    evidence_img = create_change_evidence(base_rgb, mask)

    # Step 4: Deterministic NL Response Formatting (Phase 12)
    answer = (
        f"ChangeFormer detected changes across approximately {change_pct:.1f}% of the analyzed area.\n\n"
        f"**Detailed Breakdown:**\n"
        f"- Changed area: {change_pct:.2f}% ({changed_pixels:,} of {total_pixels:,} pixels)\n"
        f"- Mean model confidence: {confidence:.2%}\n"
        f"- Execution device: `{device_name}`\n"
        f"- Input dimensions: {image_a.width}x{image_a.height} px\n\n"
        f"The visual evidence map highlights detected change regions in bright red."
    )

    metadata: dict[str, Any] = {
        "model": "ChangeFormer-BiTemporal",
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percentage": change_pct,
        "changed_percent": change_pct,
        "confidence": confidence,
        "device": device_name,
        "checkpoint_loaded": pred["checkpoint_loaded"],
        "input_t1_size": (image_a.height, image_a.width),
        "input_t2_size": (image_b.height, image_b.width),
    }

    return AnalysisResult(
        answer=answer,
        evidence=evidence_img,
        mask=mask,
        index_name="ChangeFormer Binary Mask",
        confidence=confidence,
        tool_used="changeformer (change_detection)",
        metadata=metadata,
    )
