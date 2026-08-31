"""
ChangeFormer Architecture & Model Adapter for Bi-Temporal Satellite Change Detection.

Implements the ChangeFormer Siamese Transformer architecture for remote sensing
change detection, with device abstraction (MPS / CPU / CUDA) and clean inference adapter.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import RasterImage

logger = logging.getLogger(__name__)

# Default model input dimensions
INPUT_SIZE = (256, 256)

# ImageNet normalization constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device() -> torch.device:
    """
    Determine the optimal available compute device.
    Prefers Apple Silicon MPS if available and operational, then CUDA, then CPU.
    """
    if torch.backends.mps.is_available():
        try:
            # Verify MPS functionality with a tiny tensor operation
            test_tensor = torch.zeros(1, device="mps")
            _ = test_tensor + 1.0
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS detected but test operation failed ({e}). Falling back to CPU.")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class ConvBlock(nn.Module):
    """Convolutional feature extraction block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MLPDecoder(nn.Module):
    """Multi-scale feature difference fusion decoder."""

    def __init__(self, channels: list[int], embed_dim: int = 128):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for c in channels
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(channels), embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, diff_features: list[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        upsampled = []
        for i, feature in enumerate(diff_features):
            proj = self.mlps[i](feature)
            if proj.shape[2:] != target_size:
                proj = F.interpolate(proj, size=target_size, mode="bilinear", align_corners=False)
            upsampled.append(proj)

        fused = torch.cat(upsampled, dim=1)
        return self.fusion(fused)


class ChangeFormer(nn.Module):
    """
    Siamese ChangeFormer Model Architecture for Bi-Temporal Change Detection.

    Extracts multi-scale features from bi-temporal input pairs (T1, T2),
    computes multi-scale feature differences, fuses them via MLP decoder,
    and outputs 2-class change logits (0: no-change, 1: change).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # Encoder stages (shared weights for T1 and T2)
        self.stage1 = ConvBlock(in_channels, 32, stride=2)   # H/2, W/2
        self.stage2 = ConvBlock(32, 64, stride=2)           # H/4, W/4
        self.stage3 = ConvBlock(64, 128, stride=2)          # H/8, W/8
        self.stage4 = ConvBlock(128, 256, stride=2)         # H/16, W/16

        # Decoder & Classifier
        self.decoder = MLPDecoder(channels=[32, 64, 128, 256], embed_dim=128)
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        feats_t1 = self._extract_features(t1)
        feats_t2 = self._extract_features(t2)

        # Multi-scale difference feature maps |F1 - F2|
        diff_feats = [torch.abs(f1 - f2) for f1, f2 in zip(feats_t1, feats_t2)]

        # Target spatial size is stage1 feature resolution
        target_size = diff_feats[0].shape[2:]
        fused = self.decoder(diff_feats, target_size)

        logits = self.classifier(fused)
        # Upsample to original input resolution
        logits = F.interpolate(logits, size=t1.shape[2:], mode="bilinear", align_corners=False)
        return logits


class ChangeFormerAdapter:
    """
    High-level adapter wrapping ChangeFormer model initialization,
    checkpoint loading, pre/post-processing, and inference.
    """

    def __init__(self, checkpoint_path: str | None = None, device: torch.device | None = None):
        self.device = device or get_device()
        self.model = ChangeFormer(in_channels=3, num_classes=2)

        # Default checkpoint path search
        if checkpoint_path is None:
            checkpoint_path = os.environ.get(
                "CHANGEFORMER_CHECKPOINT",
                os.path.join(os.path.dirname(__file__), "checkpoints", "changeformer_levir_cd.pth")
            )

        self.checkpoint_path = checkpoint_path
        self.loaded_checkpoint = False
        self._load_weights()

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self) -> None:
        """Load pretrained checkpoint if available, otherwise initialize weights."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                self.loaded_checkpoint = True
                logger.info(f"Loaded ChangeFormer weights from: {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load ChangeFormer checkpoint from {self.checkpoint_path}: {e}")
        else:
            logger.info("No ChangeFormer checkpoint found at target path. Using initialized model.")

    def preprocess_image(self, raster: RasterImage) -> torch.Tensor:
        """Convert RasterImage to normalized PyTorch tensor [1, 3, H, W]."""
        rgb_arr = raster.to_rgb()  # (H, W, 3) uint8
        pil_img = Image.fromarray(rgb_arr).resize(INPUT_SIZE, Image.BILINEAR)
        img_np = np.array(pil_img).astype(np.float32) / 255.0

        # Normalize with ImageNet mean and std
        img_np = (img_np - MEAN) / STD
        # Transpose to (C, H, W)
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0)  # Add batch dimension -> (1, 3, H, W)

    @torch.no_grad()
    def predict(self, raster_t1: RasterImage, raster_t2: RasterImage) -> Dict[str, Any]:
        """
        Run bi-temporal change prediction on T1 and T2 images.

        Returns structured dictionary with:
            - mask: 2D uint8 numpy array (0 or 255) matched to raster_t1 spatial dimensions
            - change_probability: 2D float32 numpy array [0, 1]
            - changed_pixels: integer count of changed pixels
            - total_pixels: total spatial pixel count
            - change_percentage: percentage of changed area (0 - 100)
            - confidence: mean prediction confidence over detected change region
            - device: string name of execution device
            - checkpoint_loaded: boolean indicating if weights were loaded
        """
        orig_h, orig_w = raster_t1.height, raster_t1.width

        t1_tensor = self.preprocess_image(raster_t1).to(self.device)
        t2_tensor = self.preprocess_image(raster_t2).to(self.device)

        logits = self.model(t1_tensor, t2_tensor)
        probs = F.softmax(logits, dim=1)  # Shape: (1, 2, H, W)

        change_prob = probs[0, 1].cpu().numpy()  # (256, 256) float32 in [0, 1]

        # Resize probability map back to original spatial dimensions
        prob_pil = Image.fromarray(change_prob).resize((orig_w, orig_h), Image.BILINEAR)
        prob_orig = np.array(prob_pil, dtype=np.float32)

        # Threshold at 0.5 to produce binary mask
        binary_mask = (prob_orig >= 0.5).astype(np.uint8) * 255

        changed_pixels = int(np.sum(binary_mask > 0))
        total_pixels = binary_mask.size
        change_pct = (changed_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

        if changed_pixels > 0:
            avg_confidence = float(np.mean(prob_orig[binary_mask > 0]))
        else:
            avg_confidence = float(np.mean(1.0 - prob_orig))

        return {
            "mask": binary_mask,
            "change_probability": prob_orig,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(change_pct, 2),
            "confidence": round(avg_confidence, 4),
            "device": str(self.device),
            "checkpoint_loaded": self.loaded_checkpoint,
        }
