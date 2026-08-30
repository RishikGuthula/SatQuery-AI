"""
Evidence visualization engine.

Generates overlay images that clearly show what was detected,
where it was detected, and with what method.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def create_evidence_image(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
    label: str = "",
) -> Image.Image:
    """
    Create a visual evidence map by overlaying a detection mask on the base image.

    Args:
        base_rgb: (H, W, 3) uint8 array of the original RGB image.
        mask: (H, W) uint8 binary mask (0 = background, 255 = detected).
        color: RGB color for the overlay on detected areas.
        alpha: Overlay transparency (0-1).
        label: Optional label text to draw on the image.

    Returns:
        PIL Image of the evidence visualization.
    """
    # Ensure correct shapes
    h, w = mask.shape[:2]
    if base_rgb.shape[:2] != (h, w):
        # Resize base to match mask
        base_img = Image.fromarray(base_rgb)
        base_img = base_img.resize((w, h), Image.BILINEAR)
        base_rgb = np.array(base_img)

    # Ensure base is 3-channel uint8
    if base_rgb.dtype != np.uint8:
        bmin, bmax = float(base_rgb.min()), float(base_rgb.max())
        if bmax - bmin < 1e-8:
            base_rgb = np.zeros((*base_rgb.shape[:2], 3), dtype=np.uint8)
        else:
            base_rgb = ((base_rgb - bmin) / (bmax - bmin) * 255).astype(np.uint8)

    if base_rgb.ndim == 2:
        base_rgb = np.stack([base_rgb] * 3, axis=-1)
    elif base_rgb.shape[2] == 1:
        base_rgb = np.concatenate([base_rgb] * 3, axis=-1)
    elif base_rgb.shape[2] == 4:
        base_rgb = base_rgb[:, :, :3]

    # Create overlay
    overlay = base_rgb.copy()
    detected = mask > 127  # threshold at midpoint

    # Blend detected pixels with color
    overlay[detected, 0] = (
        base_rgb[detected, 0] * (1 - alpha) + color[0] * alpha
    ).astype(np.uint8)
    overlay[detected, 1] = (
        base_rgb[detected, 1] * (1 - alpha) + color[1] * alpha
    ).astype(np.uint8)
    overlay[detected, 2] = (
        base_rgb[detected, 2] * (1 - alpha) + color[2] * alpha
    ).astype(np.uint8)

    # Dim undetected pixels slightly
    undetected = ~detected
    overlay[undetected] = (overlay[undetected] * 0.7).astype(np.uint8)

    result = Image.fromarray(overlay)

    # Add label if provided
    if label:
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((10, 10), label, fill=(255, 255, 255), font=font)

    return result
