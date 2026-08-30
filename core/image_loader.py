"""
Image loading layer.

Handles RGB (PNG, JPEG, TIFF) and standard image inputs using PIL and NumPy,
returning a unified RasterImage data model without requiring rasterio/GDAL.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

from core.models import RasterImage, SensorType, SENTINEL2_BAND_NAMES

logger = logging.getLogger(__name__)

# Maximum allowed image dimension per side (pixels)
MAX_DIMENSION = 20000
# Maximum allowed file size (bytes) — 500 MB
MAX_FILE_SIZE = 500 * 1024 * 1024


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or is invalid."""
    pass


def _detect_sensor_type(n_bands: int, band_names: list[str] | None, metadata: dict) -> SensorType:
    """Infer sensor type from band count and metadata."""
    desc = str(metadata).lower()
    if "sentinel" in desc or "landsat" in desc:
        return SensorType.MULTISPECTRAL
    if n_bands > 3:
        return SensorType.MULTISPECTRAL
    if n_bands == 1:
        if "sar" in desc or "radar" in desc or "sigma0" in desc:
            return SensorType.SAR
        return SensorType.UNKNOWN
    return SensorType.RGB


def load_from_bytes(data: bytes, filename: str = "unknown") -> RasterImage:
    """
    Load an image from raw bytes using PIL (fallback-safe for web deployments).
    """
    if len(data) == 0:
        raise ImageLoadError("Empty file uploaded.")
    if len(data) > MAX_FILE_SIZE:
        raise ImageLoadError(
            f"File too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed: {MAX_FILE_SIZE // 1024 // 1024} MB."
        )

    return _load_with_pil(data, filename)


def _load_with_pil(data: bytes, filename: str) -> RasterImage:
    """Load image with PIL (RGB/PNG/JPEG/TIFF fallback)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # Force load to detect corruption
    except Exception as e:
        raise ImageLoadError(f"Cannot open image: {e}")

    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        raise ImageLoadError(
            f"Image dimensions ({img.width}x{img.height}) exceed "
            f"maximum {MAX_DIMENSION}px per side."
        )

    if img.width == 0 or img.height == 0:
        raise ImageLoadError("Image has zero dimensions.")

    # Convert to numpy and determine bands
    mode = img.mode
    if mode == "RGBA":
        arr = np.array(img)[:, :, :3]  # Drop alpha channel
        band_names = ["red", "green", "blue"]
    elif mode == "RGB":
        arr = np.array(img)
        band_names = ["red", "green", "blue"]
    elif mode == "L":
        arr = np.array(img)
        # Ensure 3D shape (H, W, 1) for grayscale
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=2)
        band_names = ["gray"]
    else:
        img = img.convert("RGB")
        arr = np.array(img)
        band_names = ["red", "green", "blue"]

    arr = arr.astype(np.float32)
    sensor = _detect_sensor_type(len(band_names), band_names, {})

    return RasterImage(
        data=arr,
        bands=band_names,
        sensor_type=sensor,
        dtype="float32",
        metadata={"source": "pil", "mode": mode, "filename": filename},
    )


def load_from_pil_image(img: Image.Image, filename: str = "unknown") -> RasterImage:
    """
    Load from a PIL Image object directly.
    """
    buf = io.BytesIO()
    fmt = "PNG"
    img.save(buf, format=fmt)
    buf.seek(0)
    return load_from_bytes(buf.read(), filename)