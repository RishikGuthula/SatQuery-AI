"""
Spectral index calculations.

Implements real NDVI, NDWI, NDBI when the required bands are present.
For RGB-only images, provides clearly-labeled visual proxy indices
that are NOT equivalent to the true spectral indices.

Formulas:
  NDVI = (NIR - Red) / (NIR + Red)      — Rouse et al. 1974
  NDWI = (Green - NIR) / (Green + NIR)   — McFeeters 1996
  NDBI = (SWIR - NIR) / (SWIR + NIR)     — Zha et al. 2003
"""

from __future__ import annotations

import logging

import numpy as np

from core.models import RasterImage

logger = logging.getLogger(__name__)


def _normalized_index(
    band_a: np.ndarray,
    band_b: np.ndarray,
    nodata: float | None = None,
) -> np.ndarray:
    """
    Compute a normalized difference index: (A - B) / (A + B).

    Uses a small epsilon to avoid division by zero.
    Returns values in [-1, 1] range.
    """
    a = band_a.astype(np.float64)
    b = band_b.astype(np.float64)

    # Handle nodata
    if nodata is not None:
        valid = (a != nodata) & (b != nodata)
    else:
        valid = np.ones(a.shape, dtype=bool)

    denom = a + b
    # Avoid division by zero
    safe_denom = np.where(np.abs(denom) < 1e-10, np.nan, denom)
    result = np.where(valid, (a - b) / safe_denom, np.nan)

    return result.astype(np.float32)


def can_compute_ndvi(image: RasterImage) -> bool:
    """Check if image has the bands needed for true NDVI."""
    return image.has_band("nir") and image.has_band("red")


def can_compute_ndwi(image: RasterImage) -> bool:
    """Check if image has the bands needed for true NDWI (McFeeters)."""
    return image.has_band("green") and image.has_band("nir")


def can_compute_ndbi(image: RasterImage) -> bool:
    """Check if image has the bands needed for true NDBI."""
    return image.has_band("swir1") and image.has_band("nir")


def calculate_ndvi(image: RasterImage) -> np.ndarray:
    """
    Calculate NDVI: (NIR - Red) / (NIR + Red).

    Requires multispectral image with 'nir' and 'red' bands.

    Returns:
        2D float32 array of NDVI values in [-1, 1].
        NaN where data is invalid.
    """
    nir = image.get_band("nir")
    red = image.get_band("red")

    if nir is None or red is None:
        raise ValueError(
            "Cannot compute NDVI: image lacks required 'nir' and/or 'red' bands. "
            f"Available bands: {image.bands}"
        )

    logger.info(f"Computing NDVI from bands: nir, red. Shape: {nir.shape}")
    return _normalized_index(nir, red, image.nodata)


def calculate_ndwi(image: RasterImage) -> np.ndarray:
    """
    Calculate NDWI (McFeeters 1996): (Green - NIR) / (Green + NIR).

    Requires multispectral image with 'green' and 'nir' bands.

    Returns:
        2D float32 array of NDWI values in [-1, 1].
    """
    green = image.get_band("green")
    nir = image.get_band("nir")

    if green is None or nir is None:
        raise ValueError(
            "Cannot compute NDWI (McFeeters): image lacks required 'green' "
            f"and/or 'nir' bands. Available bands: {image.bands}"
        )

    logger.info(f"Computing NDWI from bands: green, nir. Shape: {green.shape}")
    return _normalized_index(green, nir, image.nodata)


def calculate_ndbi(image: RasterImage) -> np.ndarray:
    """
    Calculate NDBI (Zha et al. 2003): (SWIR - NIR) / (SWIR + NIR).

    Requires multispectral image with 'swir1' and 'nir' bands.

    Returns:
        2D float32 array of NDBI values in [-1, 1].
    """
    swir = image.get_band("swir1")
    nir = image.get_band("nir")

    if swir is None or nir is None:
        # Try swir2 as fallback
        swir = image.get_band("swir2")
        if swir is None or nir is None:
            raise ValueError(
                "Cannot compute NDBI: image lacks required 'swir1'/'swir2' "
                f"and/or 'nir' bands. Available bands: {image.bands}"
            )

    logger.info(f"Computing NDBI from bands: swir, nir. Shape: {swir.shape}")
    return _normalized_index(swir, nir, image.nodata)


# ---------------------------------------------------------------------------
# RGB Visual Proxies (NOT true spectral indices)
# ---------------------------------------------------------------------------

def rgb_greenness_index(image: RasterImage) -> np.ndarray:
    """
    Visual greenness proxy for RGB images: (Green - Red) / (Green + Red).

    This is NOT NDVI. It's a simple color-based heuristic that highlights
    green areas in RGB imagery. True NDVI requires a near-infrared band.

    Returns:
        2D float32 array in [-1, 1].
    """
    if not (image.has_band("red") and image.has_band("green")):
        raise ValueError(
            f"RGB greenness requires 'red' and 'green' bands. "
            f"Available: {image.bands}"
        )

    red = image.get_band("red")
    green = image.get_band("green")

    logger.info("Computing RGB greenness proxy (NOT true NDVI).")
    return _normalized_index(green, red)


def rgb_water_proxy(image: RasterImage) -> np.ndarray:
    """
    Visual water proxy for RGB images.

    Uses a simple ratio: high green+blue, low red → water-like pixels.
    This is NOT NDWI. It's a color heuristic for RGB imagery.

    Index: (Green + Blue - 2*Red) / (Green + Blue + 2*Red)

    Returns:
        2D float32 array.
    """
    if not (image.has_band("red") and image.has_band("green")
            and image.has_band("blue")):
        raise ValueError(
            f"RGB water proxy needs 'red', 'green', 'blue' bands. "
            f"Available: {image.bands}"
        )

    red = image.get_band("red").astype(np.float64)
    green = image.get_band("green").astype(np.float64)
    blue = image.get_band("blue").astype(np.float64)

    num = green + blue - 2.0 * red
    denom = green + blue + 2.0 * red
    safe_denom = np.where(np.abs(denom) < 1e-10, np.nan, denom)
    result = (num / safe_denom).astype(np.float32)

    logger.info("Computing RGB water proxy (NOT true NDWI).")
    return result


def rgb_urban_proxy(image: RasterImage) -> np.ndarray:
    """
    Visual urban/built-up proxy for RGB images.

    Uses: (Red - Blue) / (Red + Blue) as a simple built-up indicator.
    This is NOT NDBI. It's a color heuristic.

    Returns:
        2D float32 array.
    """
    if not (image.has_band("red") and image.has_band("blue")):
        raise ValueError(
            f"RGB urban proxy needs 'red' and 'blue' bands. "
            f"Available: {image.bands}"
        )

    red = image.get_band("red")
    blue = image.get_band("blue")

    logger.info("Computing RGB urban proxy (NOT true NDBI).")
    return _normalized_index(red, blue)


def create_mask(
    index: np.ndarray,
    threshold: float = 0.0,
    above: bool = True,
) -> np.ndarray:
    """
    Create a binary mask from an index array.

    Args:
        index: 2D float array of index values.
        threshold: Threshold value.
        above: If True, mask = where index > threshold.
               If False, mask = where index < threshold.

    Returns:
        Binary uint8 mask (0 or 255).
    """
    valid = ~np.isnan(index)
    if above:
        mask = np.where(valid & (index > threshold), 255, 0).astype(np.uint8)
    else:
        mask = np.where(valid & (index < threshold), 255, 0).astype(np.uint8)
    return mask
