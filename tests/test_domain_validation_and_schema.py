"""
Unit and regression tests for Domain & Intent Validation, Structured Output Schema,
and Grounded Synthesis Engine.

Verifies:
1. Valid satellite-image queries
2. Out-of-context queries (e.g. politics, coding, weather forecast, jokes)
3. Query requiring image evidence
4. Query requiring authoritative tools (NDVI, NDWI, NDBI, ChangeFormer, BIFOLD)
5. Insufficient evidence handling (missing second image, missing bands)
6. Structured synthesis JSON schema validation
7. Malformed / non-JSON LLM output resilience
8. Streamlit UI helper and display rendering integrity
"""

import json
import pytest
import numpy as np
from PIL import Image

from agent.controller import process_query
from core.models import AnalysisResult, RasterImage, SensorType, Intent
from core.planner import plan_query, is_out_of_scope_query
from llm.client import LLMClient
from llm.planner import plan_with_llm, ExecutionPlan, TaskItem
from llm.synthesis import synthesize_results, StructuredSynthesis
from tests.test_llm_planner import MockLLMProvider
from app import get_display_image


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def test_rgb_image() -> RasterImage:
    """Standard 3-channel RGB image."""
    data = np.zeros((40, 40, 3), dtype=np.float32)
    data[:, :, 0] = 50.0   # Red
    data[:, :, 1] = 170.0  # Green
    data[:, :, 2] = 210.0  # Blue
    return RasterImage(
        data=data,
        bands=["red", "green", "blue"],
        sensor_type=SensorType.RGB,
        dtype="float32",
    )


@pytest.fixture
def test_rgb_bytes(test_rgb_image) -> bytes:
    """PNG bytes for test RGB image."""
    import io
    img = test_rgb_image.to_pil()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def test_multispectral_image() -> RasterImage:
    """Multispectral image with Red, Green, Blue, NIR, SWIR."""
    data = np.zeros((40, 40, 5), dtype=np.float32)
    data[:, :, 0] = 800.0   # Blue
    data[:, :, 1] = 1200.0  # Green
    data[:, :, 2] = 900.0   # Red
    data[:, :, 3] = 4200.0  # NIR
    data[:, :, 4] = 600.0   # SWIR
    return RasterImage(
        data=data,
        bands=["blue", "green", "red", "nir", "swir1"],
        sensor_type=SensorType.MULTISPECTRAL,
        dtype="float32",
    )


# =========================================================================
# 1. Valid Satellite Image Queries
# =========================================================================

class TestValidSatelliteQueries:
    """Tests for in-scope satellite and remote sensing questions."""

    def test_water_detection_query(self, test_rgb_bytes):
        res = process_query("Find water bodies in this satellite image", test_rgb_bytes)
        assert res.status in ("success", "insufficient_evidence")
        assert "water" in res.tool_used.lower()
        assert res.summary
        assert len(res.observations) > 0

    def test_vegetation_query(self, test_rgb_bytes):
        res = process_query("Detect vegetation and green canopy", test_rgb_bytes)
        assert res.status == "success"
        assert "vegetation" in res.tool_used.lower() or "ndvi" in res.tool_used.lower()

    def test_builtup_query(self, test_rgb_bytes):
        res = process_query("Identify built-up urban areas", test_rgb_bytes)
        assert res.status == "success"
        assert "built" in res.tool_used.lower() or "urban" in res.answer.lower()


# =========================================================================
# 2. Out-of-Context Queries
# =========================================================================

class TestOutOfContextQueries:
    """Tests ensuring out-of-domain queries are not answered with hallucinated satellite metrics."""

    @pytest.mark.parametrize(
        "query",
        [
            "Who is the president of India?",
            "Write me a Python program",
            "What is the weather today?",
            "Tell me a joke",
            "How to bake a chocolate cake?",
            "Translate this text to Spanish",
            "What is the capital of France?",
        ],
    )
    def test_deterministic_router_out_of_scope(self, query):
        plan = plan_query(query)
        assert plan.intent == Intent.OUT_OF_SCOPE, f"Query '{query}' was not recognized as OUT_OF_SCOPE"

    @pytest.mark.parametrize(
        "query",
        [
            "Who is the president of India?",
            "Write me a Python program",
            "What is the weather today?",
            "Tell me a joke",
        ],
    )
    def test_controller_out_of_scope_handling(self, query, test_rgb_bytes):
        res = process_query(query, test_rgb_bytes)
        assert res.status == "out_of_scope"
        assert res.intent == "out_of_scope"
        assert res.tool_used == "domain_validator"
        assert "SatQuery AI is designed for satellite" in res.answer
        assert res.confidence_level == "Not Applicable"
        assert len(res.observations) == 0
        assert "coverage_percent" not in res.metadata

    def test_llm_planner_out_of_scope_classification(self, test_rgb_image):
        llm_json = json.dumps({
            "status": "out_of_scope",
            "intent": "out_of_scope",
            "reasoning": "User asked for general knowledge trivia",
            "tasks": [],
            "synthesis_required": False
        })
        provider = MockLLMProvider(response_payload=llm_json, available=True)
        client = LLMClient(provider=provider)

        plan = plan_with_llm("Who is the president of India?", image1=test_rgb_image, llm_client=client)
        assert plan.status == "out_of_scope"
        assert plan.intent == "out_of_scope"
        assert len(plan.tasks) == 0


# =========================================================================
# 3. Query Requiring Image Evidence
# =========================================================================

class TestImageEvidenceRequirement:
    """Verifies that valid geospatial detection tasks generate visual evidence maps."""

    def test_evidence_map_generated_for_water_detection(self, test_rgb_bytes):
        res = process_query("Find water bodies", test_rgb_bytes)
        assert res.evidence is not None
        assert isinstance(res.evidence, Image.Image)
        assert res.mask is not None
        assert res.mask.shape[:2] == (40, 40)

    def test_evidence_sources_listed(self, test_rgb_bytes):
        res = process_query("Find water bodies", test_rgb_bytes)
        assert len(res.evidence_sources) > 0
        assert any("satellite image" in ev.lower() or "proxy" in ev.lower() for ev in res.evidence_sources)


# =========================================================================
# 4. Query Requiring Authoritative Tools
# =========================================================================

class TestAuthoritativeTools:
    """Verifies spectral tools produce exact metrics without fabrication."""

    def test_authoritative_ndwi_spectral_metrics(self, test_multispectral_image):
        from tools.water_detection import detect_water
        res = detect_water(test_multispectral_image, query="Calculate true NDWI")
        assert res.metadata["method"] == "true_ndwi"
        assert "coverage_percent" in res.metadata
        assert res.metadata["threshold"] == 0.0
        assert "NDWI (McFeeters 1996)" in res.index_name

    def test_authoritative_ndvi_spectral_metrics(self, test_multispectral_image):
        from tools.vegetation_detection import detect_vegetation
        res = detect_vegetation(test_multispectral_image, query="Calculate true NDVI")
        assert res.metadata["method"] == "true_ndvi"
        assert "coverage_percent" in res.metadata
        assert "NDVI (Rouse" in res.index_name


# =========================================================================
# 5. Insufficient Evidence Handling
# =========================================================================

class TestInsufficientEvidence:
    """Verifies honest reporting when required evidence or bands are missing."""

    def test_change_detection_missing_second_image(self, test_rgb_bytes):
        res = process_query("Detect changes between before and after", test_rgb_bytes, image2_bytes=None)
        assert res.status == "insufficient_evidence"
        assert "two images" in res.answer.lower() or "second image" in res.answer.lower()
        assert res.evidence is None

    def test_explicit_ndvi_on_rgb_only_image(self, test_rgb_image):
        from tools.vegetation_detection import detect_vegetation
        res = detect_vegetation(test_rgb_image, query="Calculate true NDVI")
        assert res.metadata["method"] == "missing_bands"
        assert "cannot be calculated" in res.answer or "missing" in res.answer.lower()
        assert res.evidence is None


# =========================================================================
# 6. Structured Synthesis Output
# =========================================================================

class TestStructuredSynthesisOutput:
    """Verifies schema adherence and structure of synthesis outputs."""

    def test_structured_synthesis_fields(self, test_rgb_image):
        plan = ExecutionPlan(
            status="ready",
            intent="water_detection",
            reasoning="User requested water detection",
            tasks=[TaskItem(capability="water_detection", reason="detect water")],
            synthesis_required=True,
        )
        res = AnalysisResult(
            answer="Water detected covering 18.5% of the scene.",
            tool_used="water_detection",
            metadata={"coverage_percent": 18.5, "method": "rgb_water_proxy"},
        )

        synth_json = json.dumps({
            "status": "success",
            "intent": "water_detection",
            "summary": "Water detected covering 18.5% of the scene.",
            "observations": [
                "Water bodies occupy approximately 18.5% of the image.",
                "Analysis performed using RGB color proxy."
            ],
            "evidence": [
                "Uploaded satellite image (40x40 px, 3 bands: red, green, blue)",
                "RGB water proxy heuristic"
            ],
            "confidence": "Moderate — visual proxy without NIR band",
            "sources": ["water_detection (RGB proxy)"]
        })
        provider = MockLLMProvider(response_payload=synth_json, available=True)
        client = LLMClient(provider=provider)

        synth = synthesize_results("Find water", plan, [res], test_rgb_image, llm_client=client)

        assert isinstance(synth, StructuredSynthesis)
        assert synth.status == "success"
        assert synth.intent == "water_detection"
        assert "18.5%" in synth.summary
        assert len(synth.observations) == 2
        assert len(synth.evidence) == 2
        assert "Moderate" in synth.confidence
        assert "water_detection (RGB proxy)" in synth.sources


# =========================================================================
# 7. Malformed / Non-JSON LLM Output Handling
# =========================================================================

class TestMalformedLLMOutput:
    """Verifies graceful fallback when LLM output is malformed or invalid."""

    def test_synthesis_recovers_from_corrupted_json(self, test_rgb_image):
        plan = ExecutionPlan(
            status="ready",
            intent="vegetation_detection",
            reasoning="Detect vegetation",
            tasks=[TaskItem(capability="vegetation_detection", reason="detect vegetation")],
            synthesis_required=True,
        )
        res = AnalysisResult(
            answer="Vegetation canopy covers 34.2% of the scene.",
            tool_used="vegetation_detection",
            metadata={"coverage_percent": 34.2, "method": "rgb_greenness"},
        )

        corrupted_payload = "NOT VALID JSON { status: 'bad', broken"
        provider = MockLLMProvider(response_payload=corrupted_payload, available=True)
        client = LLMClient(provider=provider)

        synth = synthesize_results("Find vegetation", plan, [res], test_rgb_image, llm_client=client)

        assert isinstance(synth, StructuredSynthesis)
        assert synth.status == "success"
        assert "34.2%" in synth.summary or "34.2%" in str(synth)
        assert len(synth.observations) > 0


# =========================================================================
# 8. Streamlit UI Rendering Helpers
# =========================================================================

class TestStreamlitUIRendering:
    """Verifies UI helpers prepare images and display cards properly."""

    def test_display_image_rgb(self, test_rgb_bytes):
        from tests.test_response_quality_and_routing import MockUploadedFile
        upload = MockUploadedFile("sample.png", test_rgb_bytes)
        disp = get_display_image(upload)
        assert disp is not None
        assert disp.mode == "RGB"
        assert disp.size == (40, 40)

    def test_controller_result_has_all_ui_attributes(self, test_rgb_bytes):
        res = process_query("Find water bodies", test_rgb_bytes)
        assert hasattr(res, "status")
        assert hasattr(res, "intent")
        assert hasattr(res, "summary")
        assert hasattr(res, "observations")
        assert hasattr(res, "evidence_sources")
        assert hasattr(res, "confidence_level")
        assert hasattr(res, "sources")
        assert hasattr(res, "structured_output")
