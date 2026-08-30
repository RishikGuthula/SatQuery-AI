"""
Image loading layer.

Handles RGB (PNG, JPEG) and multispectral (GeoTIFF) inputs,
returning a unified RasterImage data model.
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


def _detect_sensor_type(n_bands: int, band_names: list[str] | None,
                         metadata: dict) -> SensorType:
    """Infer sensor type from band count and metadata."""
    # Check for known satellite band names in metadata
    desc = str(metadata).lower()
    if "sentinel" in desc or "landsat" in desc:
        return SensorType.MULTISPECTRAL
    if n_bands > 3:
        return SensorType.MULTISPECTRAL
    if n_bands == 1:
        # Single-band could be SAR or grayscale
        if "sar" in desc or "radar" in desc or "sigma0" in desc:
            return SensorType.SAR
        return SensorType.UNKNOWN
    return SensorType.RGB


def _band_names_for_multispectral(n_bands: int) -> list[str]:
    """Generate default band names for multispectral data."""
    # Only use Sentinel-2 names for 4+ bands (likely satellite data)
    if n_bands > 3 and n_bands <= 12:
        return [SENTINEL2_BAND_NAMES.get(i, f"band_{i+1}") for i in range(n_bands)]
    return [f"band_{i+1}" for i in range(n_bands)]


def _standard_band_names(n_bands: int) -> list[str]:
    """Standard band names for common image formats."""
    if n_bands == 1:
        return ["gray"]
    if n_bands == 3:
        return ["red", "green", "blue"]
    if n_bands == 4:
        return ["red", "green", "blue", "alpha"]
    return [f"band_{i+1}" for i in range(n_bands)]


def load_from_bytes(data: bytes, filename: str = "unknown") -> RasterImage:
    """
    Load an image from raw bytes.

    Tries rasterio first for GeoTIFF/multispectral, then falls back to PIL.
    """
    if len(data) == 0:
        raise ImageLoadError("Empty file uploaded.")
    if len(data) > MAX_FILE_SIZE:
        raise ImageLoadError(
            f"File too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed: {MAX_FILE_SIZE // 1024 // 1024} MB."
        )

    # Try rasterio for geospatial formats
    raster: RasterImage | None = None
    try:
        raster = _load_with_rasterio(data, filename)
    except Exception as e:
        logger.debug(f"Rasterio load failed for {filename}: {e}")

    if raster is not None:
        return raster

    # Fall back to PIL for standard images
    return _load_with_pil(data, filename)


def _load_with_rasterio(data: bytes, filename: str) -> RasterImage | None:
    """Attempt to load with rasterio. Returns None if not a valid raster."""
    try:
        import rasterio
        from rasterio.errors import RasterioIOError
    except ImportError:
        return None

    try:
        with rasterio.open(io.BytesIO(data)) as src:
            n_bands = src.count
            if n_bands == 0:
                raise ImageLoadError("Raster has zero bands.")

            # Read all bands: shape (C, H, W) -> transpose to (H, W, C)
            arr = src.read()  # (C, H, W)
            if arr.shape[1] > MAX_DIMENSION or arr.shape[2] > MAX_DIMENSION:
                raise ImageLoadError(
                    f"Image dimensions ({arr.shape[2]}x{arr.shape[1]}) exceed "
                    f"maximum {MAX_DIMENSION}px per side."
                )

            arr_t = np.transpose(arr, (1, 2, 0)).astype(np.float32)  # (H, W, C)

            # Band names: use rasterio descriptions if meaningful,
            # otherwise use standard names for common formats
            driver = (src.driver or "").upper()
            has_meaningful_descriptions = (
                src.descriptions and any(d for d in src.descriptions if d)
            )

            if has_meaningful_descriptions:
                band_names = [
                    d if d else f"band_{i+1}" for i, d in enumerate(src.descriptions)
                ]
            elif driver in ("PNG", "JPEG", "JPG", "GIF", "BMP") or n_bands <= 4:
                # Standard image formats — use conventional names
                band_names = _standard_band_names(n_bands)
            else:
                band_names = _band_names_for_multispectral(n_bands)

            metadata = dict(src.meta) if src.meta else {}
            # Convert non-serializable values to strings
            for k, v in metadata.items():
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    metadata[k] = str(v)

            sensor = _detect_sensor_type(n_bands, band_names, metadata)
            nodata_val = src.nodata

            # Drop alpha channel for standard images
            if n_bands == 4 and "alpha" in band_names:
                alpha_idx = band_names.index("alpha")
                arr_t = np.delete(arr_t, alpha_idx, axis=2)
                band_names = [b for b in band_names if b != "alpha"]
                n_bands = len(band_names)

            return RasterImage(
                data=arr_t,
                bands=band_names,
                crs=str(src.crs) if src.crs else None,
                transform=src.transform,
                bounds=tuple(src.bounds) if src.bounds else None,
                resolution=(src.res[0], src.res[1]) if src.res else None,
                nodata=nodata_val,
                dtype=str(arr.dtype),
                sensor_type=sensor,
                metadata=metadata,
            )
    except ImageLoadError:
        raise
    except Exception as e:
        logger.debug(f"Not a valid raster for rasterio: {e}")
        return None


def _load_with_pil(data: bytes, filename: str) -> RasterImage:
    """Load image with PIL (RGB/PNG/JPEG fallback)."""
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

    # Convert to numpy
    mode = img.mode
    if mode == "RGBA":
        arr = np.array(img)[:, :, :3]  # Drop alpha
        band_names = ["red", "green", "blue"]
    elif mode == "RGB":
        arr = np.array(img)
        band_names = ["red", "green", "blue"]
    elif mode == "L":
        arr = np.array(img)
        band_names = ["gray"]
    elif mode == "P":
        img = img.convert("RGB")
        arr = np.array(img)
        band_names = ["red", "green", "blue"]
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
    Load from a PIL Image object (used when Streamlit provides one).
    """
    buf = io.BytesIO()
    fmt = "PNG"
    img.save(buf, format=fmt)
    buf.seek(0)
    return load_from_bytes(buf.read(), filename)
