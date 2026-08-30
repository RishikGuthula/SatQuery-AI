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
    fname_lower = filename.lower()

    # Detect specific Sentinel-2 / Landsat / HLS band from filename
    detected_band_name = "gray"
    if any(k in fname_lower for k in ("b04", "b4.", "b4_", "red", "band4", "band 4")):
        detected_band_name = "red"
    elif any(k in fname_lower for k in ("b08", "b8.", "b8_", "b8a", "nir", "band8", "band 8")):
        detected_band_name = "nir"
    elif any(k in fname_lower for k in ("b03", "b3.", "b3_", "green", "band3", "band 3")):
        detected_band_name = "green"
    elif any(k in fname_lower for k in ("b02", "b2.", "b2_", "blue", "band2", "band 2")):
        detected_band_name = "blue"
    elif any(k in fname_lower for k in ("b11", "swir1", "swir_1", "band11")):
        detected_band_name = "swir1"
    elif any(k in fname_lower for k in ("b12", "swir2", "swir_2", "band12")):
        detected_band_name = "swir2"

    if mode == "RGBA":
        arr = np.array(img)[:, :, :3].astype(np.float32)  # Drop alpha channel
        band_names = ["red", "green", "blue"]
    elif mode == "RGB":
        arr = np.array(img, dtype=np.float32)
        band_names = ["red", "green", "blue"]
    elif mode in ("L", "I;16", "I", "F", "I;16B", "I;16L", "I;16S", "I;16BS", "16-bit uint") or mode.startswith("I"):
        # Scientific single-band (16-bit, 32-bit int, or float)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=2)
        band_names = [detected_band_name]
    else:
        # Multi-band or other TIFF mode
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=2)
            band_names = [detected_band_name]
        elif arr.ndim == 3:
            if arr.shape[2] == 3:
                band_names = ["red", "green", "blue"]
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
                band_names = ["red", "green", "blue"]
            else:
                band_names = [f"B{i+1}" for i in range(arr.shape[2])]
        else:
            try:
                rgb_img = img.convert("RGB")
                arr = np.array(rgb_img, dtype=np.float32)
                band_names = ["red", "green", "blue"]
            except Exception:
                arr = np.array(img, dtype=np.float32)
                band_names = [detected_band_name]

    sensor = _detect_sensor_type(len(band_names), band_names, {"filename": filename, "mode": mode})
    if detected_band_name in ("red", "nir", "green", "blue", "swir1", "swir2") and filename != "unknown":
        sensor = SensorType.MULTISPECTRAL

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