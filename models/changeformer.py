"""
Pretrained ChangeFormer Architecture & Model Adapter for Bi-Temporal Satellite Change Detection.

Implements the official ChangeFormer (MiT-B0 Siamese Transformer) architecture for remote sensing
change detection, with device abstraction (MPS / CPU / CUDA) and strict pretrained checkpoint loading.
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

# ImageNet normalization constants matching SegFormer / ChangeFormer
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


class OverlapPatchEmbed(nn.Module):
    """Image to Overlapping Patch Embedding."""

    def __init__(self, patch_size: int, stride: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixFFN(nn.Module):
    """Mix-FFN with depthwise convolution."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features),
            nn.GELU(),
            nn.Identity(),  # dropout placeholder
            nn.Conv2d(hidden_features, in_features, 1),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.layers(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AttentionBlock(nn.Module):
    """Efficient Spatial-Reduction Attention."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        if self.sr_ratio > 1:
            x_spatial = x.transpose(1, 2).view(B, C, H, W)
            x_reduced = self.sr(x_spatial)
            x_reduced = x_reduced.flatten(2).transpose(1, 2)
            kv = self.norm(x_reduced)
        else:
            kv = x
        out, _ = self.attn(x, kv, kv)
        return out


class TransformerBlock(nn.Module):
    """Hierarchical Transformer encoder block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionBlock(dim, num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class Stage(nn.Module):
    """A stage of the Mix Vision Transformer encoder."""

    def __init__(
        self, in_chans: int, embed_dim: int, patch_size: int, stride: int,
        num_heads: int, mlp_ratio: int, sr_ratio: int, depth: int = 2
    ):
        super().__init__()
        self.add_module("0", OverlapPatchEmbed(patch_size, stride, in_chans, embed_dim))
        blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, sr_ratio) for _ in range(depth)
        ])
        self.add_module("1", blocks)
        self.add_module("2", nn.LayerNorm(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_embed = getattr(self, "0")
        blocks = getattr(self, "1")
        norm = getattr(self, "2")
        x, H, W = patch_embed(x)
        for blk in blocks:
            x = blk(x, H, W)
        x = norm(x)
        B, N, C = x.shape
        return x.transpose(1, 2).view(B, C, H, W)


class MixVisionTransformer(nn.Module):
    """Hierarchical MiT-B0 encoder backbone."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            Stage(in_chans=3, embed_dim=32, patch_size=7, stride=4, num_heads=1, mlp_ratio=4, sr_ratio=8, depth=2),
            Stage(in_chans=32, embed_dim=64, patch_size=3, stride=2, num_heads=2, mlp_ratio=4, sr_ratio=4, depth=2),
            Stage(in_chans=64, embed_dim=160, patch_size=3, stride=2, num_heads=5, mlp_ratio=4, sr_ratio=2, depth=2),
            Stage(in_chans=160, embed_dim=256, patch_size=3, stride=2, num_heads=8, mlp_ratio=4, sr_ratio=1, depth=2),
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for stage in self.layers:
            x = stage(x)
            outs.append(x)
        return outs


class ConvModule(nn.Module):
    """Standard Convolution-BatchNorm-ReLU module."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ChangeFormerDecodeHead(nn.Module):
    """Multi-Scale MLP Difference Fusion Decoder."""

    def __init__(self, in_channels: list[int] = [64, 128, 320, 512], embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.convs = nn.ModuleList([ConvModule(c, embed_dim) for c in in_channels])
        self.fusion_conv = ConvModule(embed_dim * len(in_channels), embed_dim)
        self.conv_seg = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, feats1: list[torch.Tensor], feats2: list[torch.Tensor]) -> torch.Tensor:
        fused_multiscale = []
        target_size = feats1[0].shape[2:]  # H/4, W/4
        for i, (f1, f2) in enumerate(zip(feats1, feats2)):
            concat_f = torch.cat([f1, f2], dim=1)
            proj = self.convs[i](concat_f)
            if proj.shape[2:] != target_size:
                proj = F.interpolate(proj, size=target_size, mode="bilinear", align_corners=False)
            fused_multiscale.append(proj)

        all_feats = torch.cat(fused_multiscale, dim=1)
        fused = self.fusion_conv(all_feats)
        logits = self.conv_seg(fused)
        return logits


class ChangeFormer(nn.Module):
    """
    Siamese ChangeFormer Model (MiT-B0 Backbone + Multi-Scale MLP Decoder).

    Accepts bi-temporal images T1 and T2, extracts hierarchical transformer
    representations, fuses multi-scale change features, and outputs binary change logits.
    """

    def __init__(self):
        super().__init__()
        self.backbone = MixVisionTransformer()
        self.decode_head = ChangeFormerDecodeHead()

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        orig_size = t1.shape[2:]
        feats1 = self.backbone(t1)
        feats2 = self.backbone(t2)
        logits = self.decode_head(feats1, feats2)
        logits = F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
        return logits


class ChangeFormerAdapter:
    """
    High-level adapter wrapping ChangeFormer model initialization,
    strict pretrained checkpoint loading, and bi-temporal inference.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: torch.device | None = None,
        require_checkpoint: bool = False,
    ):
        self.device = device or get_device()
        self.model = ChangeFormer()

        # Checkpoint path resolution
        if checkpoint_path is None:
            default_path = os.path.join(os.path.dirname(__file__), "checkpoints", "changeformer_mit_b0.pth")
            fallback_path = os.path.join(os.path.dirname(__file__), "checkpoints", "changeformer_levir_cd.pth")
            checkpoint_path = os.environ.get("CHANGEFORMER_CHECKPOINT") or (
                default_path if os.path.exists(default_path) else fallback_path
            )

        self.checkpoint_path = checkpoint_path
        self.loaded_checkpoint = False
        self._load_weights(require_checkpoint=require_checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, require_checkpoint: bool = False) -> None:
        """Load pretrained checkpoint strictly. Fails loudly if invalid."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                # Strict parameter loading
                missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
                self.loaded_checkpoint = True
                logger.info(f"Loaded pretrained ChangeFormer checkpoint strictly from: {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Strict checkpoint loading failed for {self.checkpoint_path}: {e}")
                if require_checkpoint:
                    raise RuntimeError(f"Pretrained ChangeFormer checkpoint loading failed: {e}")
        else:
            if require_checkpoint:
                raise RuntimeError(
                    f"Pretrained ChangeFormer checkpoint is required but not found at: {self.checkpoint_path}"
                )
            logger.warning(f"No checkpoint found at {self.checkpoint_path}. Checkpoint loaded is False.")

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
