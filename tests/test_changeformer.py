"""
Comprehensive Test Suite for ChangeFormer Integration.

Verifies model architecture, device selection (MPS / CPU), inference adapter,
bi-temporal image validation, error handling, and routing logic.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from core.models import Intent, RasterImage, SensorType
from core.planner import plan_query
from models.changeformer import ChangeFormer, ChangeFormerAdapter, get_device
from tools.changeformer_tool import (
    ChangeFormerError,
    detect_changes_changeformer,
    validate_bitemporal_pair,
)
from tools.registry import get_registry


def make_test_raster(
    height: int = 128,
    width: int = 128,
    color: tuple[int, int, int] = (100, 100, 100),
    sensor_type: SensorType = SensorType.RGB,
) -> RasterImage:
    """Helper to generate a mock synthetic RasterImage for fast testing."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = color
    return RasterImage(
        data=arr.astype(np.float32),
        bands=["red", "green", "blue"],
        sensor_type=sensor_type,
        width=width,
        height=height,
    )


# ── Test 1: ChangeFormer model imports successfully ───────────────────
def test_changeformer_imports():
    assert ChangeFormer is not None
    assert ChangeFormerAdapter is not None
    assert get_device is not None


# ── Test 2 & 3: Device selection & MPS check ─────────────────────────
def test_device_selection():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("mps", "cuda", "cpu")
    if torch.backends.mps.is_available():
        assert device.type == "mps"


# ── Test 4 & 5 & 6 & 7: Model & Adapter inference output ────────────
def test_changeformer_adapter_predict():
    img1 = make_test_raster(height=128, width=128, color=(50, 50, 50))
    img2 = make_test_raster(height=128, width=128, color=(200, 200, 200))

    adapter = ChangeFormerAdapter(checkpoint_path=None)  # Uses clean initialized weights
    result = adapter.predict(img1, img2)

    assert "mask" in result
    assert "change_percentage" in result
    assert "changed_pixels" in result
    assert "total_pixels" in result
    assert "confidence" in result
    assert "device" in result

    mask = result["mask"]
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (128, 128)
    assert set(np.unique(mask)).issubset({0, 255})
    assert result["total_pixels"] == 128 * 128
    assert 0.0 <= result["change_percentage"] <= 100.0


# ── Test 8: Missing T2 image ──────────────────────────────────────────
def test_missing_t2_validation():
    img1 = make_test_raster()
    with pytest.raises(ChangeFormerError, match="Both primary .* and secondary .* images are required"):
        validate_bitemporal_pair(img1, None)


# ── Test 9: Invalid / empty image ────────────────────────────────────
def test_invalid_empty_image_validation():
    img1 = make_test_raster()
    empty_img = RasterImage(
        data=np.zeros((0, 0, 3), dtype=np.float32),
        bands=[],
        sensor_type=SensorType.RGB,
        width=0,
        height=0,
    )
    with pytest.raises(ChangeFormerError, match="contains no readable image data"):
        validate_bitemporal_pair(img1, empty_img)


# ── Test 10: Automatic spatial dimension alignment (T1 437x275, T2 432x270) ──
def test_exact_dimension_mismatch_437x275_and_432x270():
    img1 = make_test_raster(height=275, width=437, color=(60, 60, 60))
    img2 = make_test_raster(height=270, width=432, color=(180, 180, 180))

    res = detect_changes_changeformer(img1, img2, query="Detect changes between T1 and T2")

    assert res.status == "success"
    assert res.mask is not None
    assert res.mask.shape == (275, 437)
    assert res.metadata["alignment_applied"] is True
    assert res.metadata["aligned_from"] == "432x270"
    assert res.metadata["aligned_to"] == "437x275"
    assert "Secondary image automatically aligned from 432x270 to 437x275" in res.answer
    assert any("432x270" in ev and "437x275" in ev for ev in res.evidence_sources)


def test_matching_dimensions_no_alignment():
    img1 = make_test_raster(height=128, width=128, color=(70, 70, 70))
    img2 = make_test_raster(height=128, width=128, color=(170, 170, 170))

    res = detect_changes_changeformer(img1, img2)

    assert res.status == "success"
    assert res.mask.shape == (128, 128)
    assert res.metadata["alignment_applied"] is False
    assert "Spatial alignment" not in res.answer


def test_large_dimension_mismatch_alignment():
    img1 = make_test_raster(height=256, width=512, color=(50, 80, 50))
    img2 = make_test_raster(height=512, width=256, color=(150, 80, 50))

    res = detect_changes_changeformer(img1, img2)

    assert res.status == "success"
    assert res.mask.shape == (256, 512)
    assert res.metadata["alignment_applied"] is True
    assert res.metadata["aligned_from"] == "256x512"
    assert res.metadata["aligned_to"] == "512x256"


def test_multispectral_dimension_alignment():
    # 12-channel multispectral raster
    arr1 = np.ones((100, 120, 12), dtype=np.float32) * 50.0
    arr2 = np.ones((80, 90, 12), dtype=np.float32) * 150.0

    bands = [f"B{i}" for i in range(12)]
    r1 = RasterImage(data=arr1, bands=bands, sensor_type=SensorType.MULTISPECTRAL, width=120, height=100)
    r2 = RasterImage(data=arr2, bands=bands, sensor_type=SensorType.MULTISPECTRAL, width=90, height=80)

    from tools.changeformer_tool import align_and_validate_bitemporal_pair
    _, aligned_r2, info = align_and_validate_bitemporal_pair(r1, r2)

    assert info["aligned"] is True
    assert aligned_r2.width == 120
    assert aligned_r2.height == 100
    assert aligned_r2.data.shape == (100, 120, 12)
    assert aligned_r2.bands == bands


def test_modality_mismatch_raises_error():
    img_opt = make_test_raster(height=64, width=64, sensor_type=SensorType.RGB)
    img_sar = make_test_raster(height=64, width=64, sensor_type=SensorType.SAR)

    with pytest.raises(ChangeFormerError, match="Cannot compare SAR imagery with optical imagery"):
        validate_bitemporal_pair(img_sar, img_opt)

    with pytest.raises(ChangeFormerError, match="Cannot compare optical imagery with SAR imagery"):
        validate_bitemporal_pair(img_opt, img_sar)


# ── Test 11: Change detection tool wrapper ───────────────────────────
def test_detect_changes_changeformer_tool():
    img1 = make_test_raster(height=64, width=64, color=(50, 100, 50))
    img2 = make_test_raster(height=64, width=64, color=(150, 100, 50))

    res = detect_changes_changeformer(img1, img2, query="Compare these images")

    assert "changeformer" in res.tool_used.lower()
    assert "ChangeFormer detected changes across approximately" in res.answer
    assert res.evidence is not None
    assert isinstance(res.evidence, Image.Image)
    assert res.mask is not None
    assert res.metadata["model"] == "ChangeFormer-BiTemporal"


# ── Test 12 & 13: Capability Registry integration ────────────────────
def test_registry_has_changeformer():
    reg = get_registry()
    cap = reg.get("changeformer")
    assert cap is not None
    assert cap.is_available()
    assert "dual_image" in cap.supported_inputs

    cap_cd = reg.get("change_detection")
    assert cap_cd is not None
    assert cap_cd.is_available()


# ── Test 14 & 15: Router logic (Temporal -> ChangeFormer, Single -> GeoChat) ──
def test_routing_temporal_queries():
    queries = [
        "What changed between these two images?",
        "Compare these satellite images.",
        "Find changes between the before and after images.",
        "Detect changes over time.",
        "Show areas that changed.",
        "Did the built-up area change?",
    ]
    for q in queries:
        plan = plan_query(q, has_second_image=True)
        assert plan.intent == Intent.CHANGE_DETECTION, f"Query failed routing: {q}"


def test_routing_single_image_queries():
    single_queries = [
        "Describe this satellite image.",
        "What do you see?",
        "Identify objects in this image.",
    ]
    for q in single_queries:
        plan = plan_query(q, has_second_image=False)
        assert plan.intent == Intent.IMAGE_DESCRIPTION, f"Single query failed routing: {q}"


# ── Test 16: Real Satellite Bi-Temporal Inference with Pretrained Checkpoint ──
def test_real_bitemporal_inference_with_checkpoint():
    import os
    t1_path = "sample_data/levir_cd/real_t1.png"
    t2_path = "sample_data/levir_cd/real_t2.png"
    ckpt_path = "models/checkpoints/changeformer_mit_b0.pth"

    if not (os.path.exists(t1_path) and os.path.exists(t2_path) and os.path.exists(ckpt_path)):
        pytest.skip("Real satellite test images or checkpoint not found on disk")

    im1 = Image.open(t1_path)
    im2 = Image.open(t2_path)
    raster1 = RasterImage(
        data=np.array(im1, dtype=np.float32),
        bands=["red", "green", "blue"],
        sensor_type=SensorType.RGB,
        width=im1.width,
        height=im1.height,
    )
    raster2 = RasterImage(
        data=np.array(im2, dtype=np.float32),
        bands=["red", "green", "blue"],
        sensor_type=SensorType.RGB,
        width=im2.width,
        height=im2.height,
    )

    adapter = ChangeFormerAdapter(checkpoint_path=ckpt_path, require_checkpoint=True)
    assert adapter.loaded_checkpoint is True, "Pretrained checkpoint must be loaded"

    res = adapter.predict(raster1, raster2)
    assert res["checkpoint_loaded"] is True
    assert res["total_pixels"] == raster1.width * raster1.height
    assert isinstance(res["mask"], np.ndarray)
    assert res["mask"].shape == (raster1.height, raster1.width)
    assert 0.0 <= res["change_percentage"] <= 100.0
    assert 0.0 <= res["confidence"] <= 1.0
