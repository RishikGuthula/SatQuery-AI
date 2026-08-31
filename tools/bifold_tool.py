"""
BIFOLD Tool Interface for Multi-Modal Optical + SAR Satellite Analysis.

Wraps the BIFOLD RDNet Base model for 12-band Sentinel-1 (VV, VH) + Sentinel-2
(B02-B08, B8A, B11, B12) land-cover classification and structured analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.models import AnalysisResult, RasterImage
from models.bifold import BIFOLDAdapter, BIFOLDInputError

logger = logging.getLogger(__name__)


class BIFOLDToolError(Exception):
    """Raised when BIFOLD optical+SAR analysis cannot be performed."""
    pass


# Global singleton adapter instance for reuse across requests
_bifold_adapter: BIFOLDAdapter | None = None


def get_bifold_adapter() -> BIFOLDAdapter:
    """Retrieve or initialize the global BIFOLDAdapter singleton."""
    global _bifold_adapter
    if _bifold_adapter is None:
        _bifold_adapter = BIFOLDAdapter()
    return _bifold_adapter


def analyze_optical_sar_bifold(
    raster: RasterImage,
    query: str = "Classify land cover using optical and SAR data",
    threshold: float = 0.3,
) -> AnalysisResult:
    """
    Run multi-modal optical + SAR land-cover classification using BIFOLD RDNet Base.

    Args:
        raster: Multi-band RasterImage containing all 12 required Sentinel-1 & Sentinel-2 bands.
        query: User prompt or query string.
        threshold: Confidence threshold for declaring a land-cover class detected (default: 0.3).

    Returns:
        AnalysisResult containing structured predictions, summary answer, and metadata.
    """
    if raster is None or raster.data is None or raster.data.size == 0:
        raise BIFOLDToolError("Input raster image is empty or contains no valid data.")

    adapter = get_bifold_adapter()

    try:
        results = adapter.predict(raster, threshold=threshold)
    except BIFOLDInputError as e:
        raise BIFOLDToolError(str(e))
    except Exception as e:
        logger.exception("BIFOLD inference failed.")
        raise BIFOLDToolError(f"BIFOLD optical+SAR analysis failed: {e}")

    predictions = results.get("predictions", [])
    top_pred = results.get("top_prediction")

    # Format structured textual answer
    lines: List[str] = []
    lines.append("### BIFOLD RDNet Optical + SAR Land-Cover Analysis")
    lines.append(
        f"Analyzed **12-channel multi-modal satellite data** (Sentinel-1 SAR `VV/VH` + "
        f"Sentinel-2 Multispectral `B02-B08, B8A, B11, B12`) using pretrained **BIFOLD RDNet Base** on device `{results.get('device')}`.\n"
    )

    if top_pred:
        lines.append(f"**Primary Land Cover:** {top_pred['label']} ({top_pred['percentage']}%)")

    detected = [p for p in predictions if p["is_detected"]]
    if detected:
        lines.append("\n**Detected Land-Cover Categories:**")
        for p in detected:
            lines.append(f"- **{p['label']}**: `{p['percentage']}%` confidence")
    else:
        lines.append("\n**Top Predictions:**")
        for p in predictions[:3]:
            lines.append(f"- **{p['label']}**: `{p['percentage']}%` confidence")

    answer = "\n".join(lines)

    return AnalysisResult(
        answer=answer,
        tool_used="bifold_rdnet",
        metadata={
            "model": "BIFOLD-BigEarthNetv2-0-RDNet",
            "predictions": predictions,
            "top_prediction": top_pred,
            "detected_classes": results.get("detected_classes", []),
            "modality": results.get("modality", {}),
            "device": results.get("device"),
            "checkpoint_loaded": results.get("checkpoint_loaded", True),
        },
    )
