"""
BIFOLD RDNet Architecture & Model Adapter for Multi-Modal Optical + SAR Remote Sensing Analysis.

Implements the official BIFOLD-BigEarthNetv2-0 / RDNet Base architecture for 12-channel
Sentinel-1 (VV, VH) + Sentinel-2 (B02-B08, B8A, B11, B12) Earth observation analysis,
with strict checkpoint loading from safetensors, per-band normalization, and Apple Silicon MPS acceleration.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import timm
import torch
import torch.nn.functional as F

from core.models import RasterImage

logger = logging.getLogger(__name__)

# Input dimensions and band requirements for BIFOLD RDNet Base
INPUT_SIZE = (120, 120)

# Expected 12 input bands in exact order
REQUIRED_BANDS: List[str] = [
    "vv", "vh", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b11", "b12"
]

# Official BigEarthNet v2.0 (reBEN) 19-class land-cover taxonomy
BIGEARTHNET_19_CLASSES: List[str] = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]

# Official BigEarthNet v2.0 per-channel statistics (120_bilinear interpolation)
BAND_MEANS: Dict[str, float] = {
    "vv": -12.643863677978516,
    "vh": -19.352558135986328,
    "b02": 438.3720703125,
    "b03": 614.0556640625,
    "b04": 588.4096069335938,
    "b05": 942.7476806640625,
    "b06": 1769.8486328125,
    "b07": 2049.475830078125,
    "b08": 2193.2919921875,
    "b8a": 2235.48681640625,
    "b11": 1568.2115478515625,
    "b12": 997.715087890625,
}

BAND_STDS: Dict[str, float] = {
    "vv": 5.133493900299072,
    "vh": 5.590505599975586,
    "b02": 607.02685546875,
    "b03": 603.2968139648438,
    "b04": 684.56884765625,
    "b05": 727.5784301757812,
    "b06": 1087.4288330078125,
    "b07": 1261.4302978515625,
    "b08": 1369.3717041015625,
    "b8a": 1342.490478515625,
    "b11": 1063.9197998046875,
    "b12": 806.8846435546875,
}


class BIFOLDInputError(ValueError):
    """Raised when an input image lacks the required 12-channel Sentinel-1 + Sentinel-2 bands."""
    pass


def get_device() -> torch.device:
    """
    Determine optimal compute device for BIFOLD execution.
    Prefers Apple Silicon MPS if available, falls back to CUDA or CPU.
    """
    if torch.backends.mps.is_available():
        try:
            test_tensor = torch.zeros(1, device="mps")
            _ = test_tensor + 1.0
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS detected but test operation failed ({e}). Falling back to CPU.")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def normalize_band_name(name: str) -> str:
    """Standardize band naming for case/prefix-insensitive lookup."""
    clean = name.strip().lower()
    mapping = {
        "b2": "b02", "b3": "b03", "b4": "b04", "b5": "b05", "b6": "b06",
        "b7": "b07", "b8": "b08", "b8_a": "b8a", "b08a": "b8a", "band8a": "b8a",
        "band2": "b02", "band3": "b03", "band4": "b04", "band5": "b05",
        "band6": "b06", "band7": "b07", "band8": "b08", "band11": "b11",
        "band12": "b12", "sar_vv": "vv", "sar_vh": "vh",
    }
    return mapping.get(clean, clean)


class BIFOLDAdapter:
    """
    High-level adapter wrapping BIFOLD RDNet Base model initialization,
    strict safetensors checkpoint loading, multi-modal preprocessing, and inference.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: torch.device | None = None,
        require_checkpoint: bool = True,
    ):
        self.device = device or get_device()
        self.model = timm.create_model("rdnet_base", in_chans=12, num_classes=19, pretrained=False)

        # Checkpoint path resolution
        if checkpoint_path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), "checkpoints", "bifold_rdnet_base_all", "model.safetensors"
            )
            checkpoint_path = os.environ.get("BIFOLD_CHECKPOINT") or default_path

        self.checkpoint_path = checkpoint_path
        self.loaded_checkpoint = False
        self._load_weights(require_checkpoint=require_checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, require_checkpoint: bool = True) -> None:
        """Load pretrained weights strictly from safetensors. Fails loudly if invalid."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                from safetensors.torch import load_file
                sd = load_file(self.checkpoint_path)
                # Strip prefix if serialized as ConfigILM vision encoder
                prefix = "model.vision_encoder."
                cleaned_sd = {}
                for k, v in sd.items():
                    key = k[len(prefix):] if k.startswith(prefix) else k
                    cleaned_sd[key] = v

                missing, unexpected = self.model.load_state_dict(cleaned_sd, strict=True)
                if missing or unexpected:
                    raise RuntimeError(f"State dict mismatch. Missing: {missing}, Unexpected: {unexpected}")

                self.loaded_checkpoint = True
                logger.info(f"Loaded pretrained BIFOLD RDNet checkpoint strictly from: {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Strict BIFOLD checkpoint loading failed for {self.checkpoint_path}: {e}")
                if require_checkpoint:
                    raise RuntimeError(f"Pretrained BIFOLD RDNet checkpoint loading failed: {e}")
        else:
            if require_checkpoint:
                raise RuntimeError(
                    f"Pretrained BIFOLD RDNet checkpoint is required but not found at: {self.checkpoint_path}"
                )
            logger.warning(f"No BIFOLD checkpoint found at {self.checkpoint_path}. Checkpoint loaded is False.")

    def validate_input(self, raster: RasterImage) -> Dict[str, int]:
        """
        Validate that the input RasterImage contains all 12 required Sentinel-1 and Sentinel-2 bands.
        Returns a mapping of normalized band name to channel index in raster.
        """
        if not raster.bands:
            raise BIFOLDInputError(
                "BIFOLD RDNet requires Sentinel-1 VV/VH SAR bands and Sentinel-2 multispectral "
                "bands B02-B08, B8A, B11 and B12. The uploaded image does not contain the required 12-band input."
            )

        norm_band_map: Dict[str, int] = {}
        for idx, b in enumerate(raster.bands):
            norm_b = normalize_band_name(b)
            norm_band_map[norm_b] = idx

        missing = [b.upper() for b in REQUIRED_BANDS if b not in norm_band_map]
        if missing:
            raise BIFOLDInputError(
                "BIFOLD RDNet requires Sentinel-1 VV/VH SAR bands and Sentinel-2 multispectral "
                "bands B02-B08, B8A, B11 and B12. The uploaded image does not contain the required 12-band input."
            )

        return norm_band_map

    def preprocess_raster(self, raster: RasterImage) -> torch.Tensor:
        """
        Convert multi-modal RasterImage into a strictly normalized 12-channel PyTorch tensor [1, 12, 120, 120].
        """
        band_map = self.validate_input(raster)

        # Extract raster array: shape (H, W, C) or (C, H, W)
        data = raster.data
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        if data.shape[0] == len(raster.bands) and data.shape[2] != len(raster.bands):
            # Transpose (C, H, W) -> (H, W, C)
            data = np.transpose(data, (1, 2, 0))

        h, w, _ = data.shape
        channels_120: List[np.ndarray] = []

        for band_name in REQUIRED_BANDS:
            chan_idx = band_map[band_name]
            band_arr = data[:, :, chan_idx].astype(np.float32)

            # Resize to standard BigEarthNet patch size 120x120
            if (w, h) != INPUT_SIZE:
                pil_img = Image.fromarray(band_arr)
                resized = pil_img.resize(INPUT_SIZE, Image.BILINEAR)
                band_arr = np.array(resized, dtype=np.float32)

            # Standardize using official BigEarthNet v2.0 per-band statistics
            mean_val = BAND_MEANS[band_name]
            std_val = BAND_STDS[band_name]
            norm_arr = (band_arr - mean_val) / std_val
            channels_120.append(norm_arr)

        # Stack into (12, 120, 120)
        stacked = np.stack(channels_120, axis=0)  # Shape: (12, 120, 120)
        tensor = torch.from_numpy(stacked).float().unsqueeze(0)  # Shape: (1, 12, 120, 120)
        return tensor

    @torch.no_grad()
    def predict(self, raster: RasterImage, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run multi-modal optical+SAR land-cover inference on the 12-channel input raster.
        """
        if not self.loaded_checkpoint:
            raise RuntimeError("Pretrained BIFOLD RDNet checkpoint is required for inference.")

        tensor = self.preprocess_raster(raster).to(self.device)
        logits = self.model(tensor)  # Shape: (1, 19)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # Array of 19 probabilities

        # Structured prediction output
        predictions = []
        for label, prob in zip(BIGEARTHNET_19_CLASSES, probs):
            prob_float = float(prob)
            predictions.append({
                "label": label,
                "confidence": round(prob_float, 4),
                "percentage": round(prob_float * 100.0, 2),
                "is_detected": bool(prob_float >= threshold),
            })

        # Sort descending by confidence
        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        detected = [p for p in predictions if p["is_detected"]]
        top_prediction = predictions[0] if predictions else None

        return {
            "predictions": predictions,
            "detected_classes": [p["label"] for p in detected],
            "top_prediction": top_prediction,
            "modality": {
                "sentinel1": ["VV", "VH"],
                "sentinel2": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
            },
            "device": str(self.device),
            "checkpoint_loaded": self.loaded_checkpoint,
        }
