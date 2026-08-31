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


# ── Test 10: Spatial dimension mismatch ──────────────────────────────
def test_dimension_mismatch_validation():
    img1 = make_test_raster(height=128, width=128)
    img2 = make_test_raster(height=256, width=256)

    with pytest.raises(ChangeFormerError, match="Dimension mismatch for ChangeFormer analysis"):
        validate_bitemporal_pair(img1, img2)


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
