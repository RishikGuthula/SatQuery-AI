"""
Comprehensive Test Suite for BIFOLD RDNet Multi-Modal Optical + SAR Integration.

Verifies model architecture, safetensors checkpoint loading, device placement (MPS / CPU),
12-channel multi-modal preprocessing, band validation, error handling, structured evidence,
routing, and real Sentinel-1 + Sentinel-2 inference.
"""

from __future__ import annotations

import os
import numpy as np
import pytest
import torch

from core.models import Intent, RasterImage, SensorType
from core.planner import plan_query
from models.bifold import (
    BAND_MEANS,
    BAND_STDS,
    BIGEARTHNET_19_CLASSES,
    BIFOLDAdapter,
    BIFOLDInputError,
    REQUIRED_BANDS,
    get_device,
    normalize_band_name,
)
from tools.bifold_tool import (
    BIFOLDToolError,
    analyze_optical_sar_bifold,
    get_bifold_adapter,
)
from tools.registry import get_registry


def make_12band_raster(
    height: int = 120,
    width: int = 120,
    bands: list[str] | None = None,
) -> RasterImage:
    """Helper to generate a mock 12-channel Sentinel-1 + Sentinel-2 RasterImage."""
    use_bands = bands or REQUIRED_BANDS
    channels = []
    for b in use_bands:
        mean = BAND_MEANS.get(b.lower(), 100.0)
        arr = np.full((height, width), mean, dtype=np.float32)
        channels.append(arr)
    stacked = np.stack(channels, axis=2)  # Shape: (height, width, num_bands)
    return RasterImage(
        data=stacked,
        bands=use_bands,
        sensor_type=SensorType.MULTISPECTRAL,
        width=width,
        height=height,
    )


# ── Test 1: Model & Adapter Imports ──────────────────────────────────
def test_bifold_imports():
    assert BIFOLDAdapter is not None
    assert BIFOLDInputError is not None
    assert BIGEARTHNET_19_CLASSES is not None
    assert len(BIGEARTHNET_19_CLASSES) == 19
    assert get_device is not None


# ── Test 2 & 3: Checkpoint Discovery & Strict Loading ─────────────────
def test_bifold_checkpoint_strict_loading():
    adapter = BIFOLDAdapter(require_checkpoint=True)
    assert adapter.loaded_checkpoint is True, "Pretrained BIFOLD checkpoint must load successfully"
    assert adapter.model is not None


# ── Test 4: Strict Compatibility (No Missing/Unexpected Keys) ─────────
def test_bifold_strict_compatibility():
    ckpt_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "checkpoints", "bifold_rdnet_base_all", "model.safetensors"
    )
    if not os.path.exists(ckpt_path):
        pytest.skip("BIFOLD checkpoint not on disk")

    from safetensors.torch import load_file
    sd = load_file(ckpt_path)
    prefix = "model.vision_encoder."
    cleaned_sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

    adapter = BIFOLDAdapter()
    missing, unexpected = adapter.model.load_state_dict(cleaned_sd, strict=True)
    assert missing == [], f"Missing keys during strict loading: {missing}"
    assert unexpected == [], f"Unexpected keys during strict loading: {unexpected}"


# ── Test 5 & 6: Device Selection & MPS/CPU Fallback ───────────────────
def test_bifold_device_selection():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("mps", "cuda", "cpu")
    if torch.backends.mps.is_available():
        assert device.type == "mps"

    # Verify adapter can be instantiated explicitly on CPU
    cpu_adapter = BIFOLDAdapter(device=torch.device("cpu"), require_checkpoint=True)
    assert cpu_adapter.device.type == "cpu"


# ── Test 7 & 8 & 9: 12-Channel Tensor Construction, Band Order & Normalization ──
def test_12channel_preprocessing_and_normalization():
    adapter = BIFOLDAdapter(require_checkpoint=True)
    raster = make_12band_raster(height=120, width=120)

    tensor = adapter.preprocess_raster(raster)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 12, 120, 120)
    assert tensor.dtype == torch.float32

    # Since pixel values equal mean, normalized values should be approximately 0.0
    assert torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-4)


# ── Test 10: Missing VV Band Raises BIFOLDInputError ──────────────────
def test_missing_vv_raises_error():
    bands = [b for b in REQUIRED_BANDS if b != "vv"]
    raster = make_12band_raster(bands=bands)
    adapter = BIFOLDAdapter(require_checkpoint=True)

    with pytest.raises(BIFOLDInputError, match="BIFOLD RDNet requires Sentinel-1 VV/VH SAR bands"):
        adapter.predict(raster)


# ── Test 11: Missing VH Band Raises BIFOLDInputError ──────────────────
def test_missing_vh_raises_error():
    bands = [b for b in REQUIRED_BANDS if b != "vh"]
    raster = make_12band_raster(bands=bands)
    adapter = BIFOLDAdapter(require_checkpoint=True)

    with pytest.raises(BIFOLDInputError, match="BIFOLD RDNet requires Sentinel-1 VV/VH SAR bands"):
        adapter.predict(raster)


# ── Test 12: Missing Sentinel-2 Multispectral Band Raises Error ───────
def test_missing_s2_band_raises_error():
    bands = [b for b in REQUIRED_BANDS if b != "b12"]
    raster = make_12band_raster(bands=bands)
    adapter = BIFOLDAdapter(require_checkpoint=True)

    with pytest.raises(BIFOLDInputError, match="BIFOLD RDNet requires Sentinel-1 VV/VH SAR bands"):
        adapter.predict(raster)


# ── Test 13: RGB Image Does NOT Run BIFOLD (Fails Explicitly) ─────────
def test_rgb_image_fails_explicitly():
    rgb_raster = RasterImage(
        data=np.zeros((120, 120, 3), dtype=np.float32),
        bands=["red", "green", "blue"],
        sensor_type=SensorType.RGB,
        width=120,
        height=120,
    )
    adapter = BIFOLDAdapter(require_checkpoint=True)

    with pytest.raises(BIFOLDInputError, match="The uploaded image does not contain the required 12-band input"):
        adapter.predict(rgb_raster)


# ── Test 14: Spatial Dimension Auto-Resizing to 120x120 ────────────────
def test_spatial_resizing():
    raster = make_12band_raster(height=256, width=256)
    adapter = BIFOLDAdapter(require_checkpoint=True)

    tensor = adapter.preprocess_raster(raster)
    assert tensor.shape == (1, 12, 120, 120)


# ── Test 15: Output Probability Range & Format ────────────────────────
def test_output_probability_range():
    raster = make_12band_raster(height=120, width=120)
    adapter = BIFOLDAdapter(require_checkpoint=True)

    res = adapter.predict(raster)
    assert "predictions" in res
    assert len(res["predictions"]) == 19
    assert res["checkpoint_loaded"] is True

    for p in res["predictions"]:
        assert "label" in p
        assert p["label"] in BIGEARTHNET_19_CLASSES
        assert 0.0 <= p["confidence"] <= 1.0
        assert 0.0 <= p["percentage"] <= 100.0
        assert isinstance(p["is_detected"], bool)


# ── Test 16: High-Level BIFOLD Tool Wrapper ───────────────────────────
def test_bifold_tool_execution():
    raster = make_12band_raster(height=120, width=120)
    res = analyze_optical_sar_bifold(raster, query="Classify land cover")

    assert res.tool_used == "bifold_rdnet"
    assert "BIFOLD RDNet Optical + SAR Land-Cover Analysis" in res.answer
    assert res.metadata["model"] == "BIFOLD-BigEarthNetv2-0-RDNet"
    assert res.metadata["checkpoint_loaded"] is True
    assert len(res.metadata["predictions"]) == 19


# ── Test 17: Capability Registry has BIFOLD ───────────────────────────
def test_registry_has_bifold():
    reg = get_registry()
    cap = reg.get("bifold")
    assert cap is not None
    assert cap.is_available()
    assert len(cap.required_bands) == 12

    cap_rdnet = reg.get("bifold_rdnet")
    assert cap_rdnet is not None
    assert cap_rdnet.is_available()


# ── Test 18: Routing - Optical+SAR Queries Route to BIFOLD ───────────
def test_routing_optical_sar_queries():
    queries = [
        "Classify the land cover using Sentinel-1 and Sentinel-2.",
        "Analyze this optical and SAR imagery.",
        "Run BIFOLD RDNet classification.",
        "What land-cover classes are present in this optical and SAR dataset?",
        "Perform SAR and optical land-cover classification.",
    ]
    for q in queries:
        plan = plan_query(q, has_second_image=False)
        assert plan.intent == Intent.OPTICAL_SAR_ANALYSIS, f"Failed routing for query: {q}"


# ── Test 19: Routing - Standard RGB Image Queries Do NOT Route to BIFOLD ──
def test_routing_rgb_queries_not_routed_to_bifold():
    rgb_queries = [
        "Describe this satellite image.",
        "What do you see in this photo?",
        "Identify objects in this aerial image.",
    ]
    for q in rgb_queries:
        plan = plan_query(q, has_second_image=False)
        assert plan.intent != Intent.OPTICAL_SAR_ANALYSIS
        assert plan.intent == Intent.IMAGE_DESCRIPTION


# ── Test 20: Real Data Inference with Pretrained Checkpoint ───────────
def test_real_s1_s2_inference():
    sample_path = "sample_data/bifold/real_s1_s2_sample.npz"
    if not os.path.exists(sample_path):
        pytest.skip("Real S1/S2 sample data not on disk")

    loaded = np.load(sample_path)
    stacked_arr = np.stack([loaded[b] for b in REQUIRED_BANDS], axis=2)

    raster = RasterImage(
        data=stacked_arr,
        bands=REQUIRED_BANDS,
        sensor_type=SensorType.MULTISPECTRAL,
        width=120,
        height=120,
    )

    adapter = BIFOLDAdapter(require_checkpoint=True)
    assert adapter.loaded_checkpoint is True

    res = adapter.predict(raster)
    assert res["checkpoint_loaded"] is True
    assert res["top_prediction"] is not None
    assert not np.isnan(res["top_prediction"]["confidence"])
    assert not np.isinf(res["top_prediction"]["confidence"])
    assert res["top_prediction"]["confidence"] > 0.0
    assert len(res["predictions"]) == 19
