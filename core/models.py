"""
Core data models for the SatQuery AI system.

Defines the unified RasterImage abstraction and AnalysisResult
that flow through the entire pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """Detected sensor/modality type of the input raster."""
    RGB = "rgb"              # Standard RGB image (PNG, JPEG, 3-band)
    MULTISPECTRAL = "multispectral"  # GeoTIFF with >3 bands or known band names
    SAR = "sar"              # Synthetic Aperture Radar (single-band intensity)
    UNKNOWN = "unknown"


class Intent(str, Enum):
    """Query intent categories."""
    WATER_DETECTION = "water_detection"
    VEGETATION_DETECTION = "vegetation_detection"
    BUILTUP_DETECTION = "builtup_detection"
    CHANGE_DETECTION = "change_detection"
    IMAGE_DESCRIPTION = "image_description"
    UNSUPPORTED = "unsupported"


@dataclass
class RasterImage:
    """
    Unified image data model.

    Abstracts over RGB (PIL) and multispectral (rasterio) inputs
    so downstream tools don't need to know the source format.
    """
    data: np.ndarray          # Shape: (H, W, C) float32 or int
    bands: list[str]          # Band names, e.g. ["red", "green", "blue"] or ["B1", ...]
    crs: str | None = None    # Coordinate reference system (WKT or PROJ4)
    transform: Any = None     # Affine transform from rasterio
    bounds: tuple | None = None  # (left, bottom, right, top)
    resolution: tuple | None = None  # (x_res, y_res)
    nodata: float | None = None
    dtype: str = "float32"
    sensor_type: SensorType = SensorType.RGB
    metadata: dict = field(default_factory=dict)
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if self.data is not None:
            self.height, self.width = self.data.shape[:2]

    @property
    def num_bands(self) -> int:
        if self.data.ndim == 2:
            return 1
        return self.data.shape[2]

    def has_band(self, name: str) -> bool:
        """Check if a named band is available (case-insensitive)."""
        return any(b.lower() == name.lower() for b in self.bands)

    def get_band_index(self, name: str) -> int | None:
        """Return index of named band, or None."""
        for i, b in enumerate(self.bands):
            if b.lower() == name.lower():
                return i
        return None

    def get_band(self, name: str) -> np.ndarray | None:
        """Return 2D array for the named band, or None if not found."""
        idx = self.get_band_index(name)
        if idx is None:
            return None
        if self.data.ndim == 2:
            return self.data
        return self.data[:, :, idx]

    def to_rgb(self) -> np.ndarray:
        """
        Extract/convert to an RGB uint8 array for display.

        If data has >= 3 bands, use first three.
        If single-band, replicate to 3 channels.
        """
        if self.data.ndim == 2:
            gray = self.data
        elif self.num_bands >= 3:
            gray = self.data[:, :, :3]
        else:
            gray = np.stack([self.data[:, :, 0]] * 3, axis=-1)

        # Normalize to 0-255 uint8
        if gray.dtype == np.uint8:
            return gray
        gmin, gmax = float(gray.min()), float(gray.max())
        if gmax - gmin < 1e-8:
            return np.zeros((*gray.shape[:2], 3), dtype=np.uint8)
        normalized = ((gray - gmin) / (gmax - gmin) * 255).astype(np.uint8)
        return normalized


@dataclass
class AnalysisResult:
    """
    Structured result returned by every analysis tool.

    Fields that cannot be meaningfully calculated are set to None.
    """
    answer: str
    evidence: Image.Image | None = None
    mask: np.ndarray | None = None
    index_name: str | None = None
    confidence: float | None = None  # None = not calculable
    tool_used: str = ""
    metadata: dict = field(default_factory=dict)


# Standard band-name aliases for common satellite sensors
SENTINEL2_BAND_NAMES = {
    0: "coastal",   # B1
    1: "blue",      # B2
    2: "green",     # B3
    3: "red",       # B4
    4: "rededge1",  # B5
    5: "rededge2",  # B6
    6: "rededge3",  # B7
    7: "nir",       # B8
    8: "nir2",      # B8A
    9: "water_vapour",  # B9
    10: "swir1",    # B11
    11: "swir2",    # B12
}
