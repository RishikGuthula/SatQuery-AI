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


def align_and_validate_bitemporal_pair(
    img_a: RasterImage,
    img_b: RasterImage,
) -> tuple[RasterImage, RasterImage, dict[str, Any]]:
    """
    Validate and spatially align a bi-temporal image pair for ChangeFormer analysis.

    If primary (T1) and secondary (T2) spatial dimensions differ, T2 is automatically
    aligned/resized to T1's spatial dimensions using high-quality bilinear interpolation.

    Returns:
        tuple of (aligned_t1, aligned_t2, alignment_info_dict)

    Raises:
        ChangeFormerError: If images are missing, empty, or share incompatible modalities.
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

    dim_a = (img_a.height, img_a.width)
    dim_b = (img_b.height, img_b.width)

    if dim_a != dim_b:
        logger.info(
            f"ChangeFormer: Spatial dimension difference detected. "
            f"Auto-aligning secondary image (T2) from {dim_b[1]}x{dim_b[0]} px "
            f"to primary image (T1) dimensions {dim_a[1]}x{dim_a[0]} px."
        )
        aligned_b = img_b.resize(img_a.width, img_a.height)
        alignment_info = {
            "aligned": True,
            "original_t2_size": (dim_b[0], dim_b[1]),
            "target_size": (dim_a[0], dim_a[1]),
            "aligned_from": f"{dim_b[1]}x{dim_b[0]}",
            "aligned_to": f"{dim_a[1]}x{dim_a[0]}",
        }
    else:
        aligned_b = img_b
        alignment_info = {
            "aligned": False,
            "original_t2_size": (dim_b[0], dim_b[1]),
            "target_size": (dim_a[0], dim_a[1]),
            "aligned_from": f"{dim_b[1]}x{dim_b[0]}",
            "aligned_to": f"{dim_a[1]}x{dim_a[0]}",
        }

    # Strict safety assertion post-alignment
    if (img_a.height, img_a.width) != (aligned_b.height, aligned_b.width):
        raise ChangeFormerError(
            f"Post-alignment dimension mismatch: T1 is {img_a.width}x{img_a.height} px, "
            f"but aligned T2 is {aligned_b.width}x{aligned_b.height} px."
        )

    return img_a, aligned_b, alignment_info


def validate_bitemporal_pair(img_a: RasterImage, img_b: RasterImage) -> None:
    """
    Validate that two images are structurally compatible for ChangeFormer change detection.
    Raises ChangeFormerError with clear user-facing messages if invalid.
    """
    align_and_validate_bitemporal_pair(img_a, img_b)


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

    # Step 1: Validate & spatially align input pair
    aligned_a, aligned_b, align_info = align_and_validate_bitemporal_pair(image_a, image_b)

    # Step 2: Acquire adapter & execute model inference
    changeformer = adapter or get_changeformer_adapter()
    pred = changeformer.predict(aligned_a, aligned_b)

    mask = pred["mask"]
    change_pct = pred["change_percentage"]
    changed_pixels = pred["changed_pixels"]
    total_pixels = pred["total_pixels"]
    confidence = pred["confidence"]
    device_name = pred["device"]

    # Step 3: Generate visual evidence map
    base_rgb = aligned_a.to_rgb()
    evidence_img = create_change_evidence(base_rgb, mask)

    # Step 4: Formatting Natural Language Response and Breakdown
    alignment_line = ""
    if align_info.get("aligned"):
        alignment_line = (
            f"- Spatial alignment: Secondary image automatically aligned from "
            f"{align_info['aligned_from']} to {align_info['aligned_to']} px for bi-temporal comparison.\n"
        )

    answer = (
        f"ChangeFormer detected changes across approximately {change_pct:.1f}% of the analyzed area.\n\n"
        f"**Detailed Breakdown:**\n"
        f"- Changed area: {change_pct:.2f}% ({changed_pixels:,} of {total_pixels:,} pixels)\n"
        f"- Mean model confidence: {confidence:.2%}\n"
        f"- Execution device: `{device_name}`\n"
        f"- Input dimensions: {aligned_a.width}x{aligned_a.height} px\n"
        f"{alignment_line}\n"
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
        "aligned_t1_size": (aligned_a.height, aligned_a.width),
        "aligned_t2_size": (aligned_b.height, aligned_b.width),
        "alignment_applied": align_info.get("aligned", False),
        "aligned_from": align_info.get("aligned_from"),
        "aligned_to": align_info.get("aligned_to"),
    }

    evidence_sources = [f"Primary image T1 ({image_a.width}x{image_a.height} px)"]
    if align_info.get("aligned"):
        evidence_sources.append(
            f"Secondary image T2 (automatically aligned from {align_info['aligned_from']} to {align_info['aligned_to']} px)"
        )
    else:
        evidence_sources.append(f"Secondary image T2 ({image_b.width}x{image_b.height} px)")

    observations = [
        f"Bi-temporal change detected across {change_pct:.2f}% of the monitored region ({changed_pixels:,} pixels).",
    ]
    if align_info.get("aligned"):
        observations.append(
            f"Secondary image automatically aligned from {align_info['aligned_from']} to {align_info['aligned_to']} px for bi-temporal comparison."
        )

    return AnalysisResult(
        answer=answer,
        evidence=evidence_img,
        mask=mask,
        index_name="ChangeFormer Binary Mask",
        confidence=confidence,
        tool_used="changeformer (change_detection)",
        metadata=metadata,
        status="success",
        intent="change_detection",
        summary=f"ChangeFormer detected {change_pct:.2f}% surface change between bi-temporal images.",
        observations=observations,
        evidence_sources=evidence_sources,
        confidence_level=f"High — ChangeFormer Transformer architecture ({confidence:.1%})",
        sources=["changeformer", "bitemporal_alignment"],
        structured_output={
            "status": "success",
            "intent": "change_detection",
            "summary": f"ChangeFormer detected {change_pct:.2f}% surface change between bi-temporal images.",
            "observations": observations,
            "evidence": evidence_sources,
            "confidence": f"High — ChangeFormer Transformer architecture ({confidence:.1%})",
            "sources": ["changeformer", "bitemporal_alignment"],
        },
    )
