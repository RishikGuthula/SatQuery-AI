"""
Tests for GeoChat-7B FastAPI Server Endpoints using TestClient.
"""

import io
import sys
import pytest
from unittest.mock import MagicMock
from PIL import Image

try:
    from fastapi.testclient import TestClient
    from geochat_server.server import app, MODEL_STATE
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="fastapi / httpx not installed")
class TestGeoChatServer:
    """Test server endpoints, authentication, and error handling."""

    @pytest.fixture(autouse=True)
    def setup_mock_model(self):
        """Set up mock model state for testing."""
        orig_state = dict(MODEL_STATE)

        # Mock torch and transformers modules if not installed locally
        mock_torch = MagicMock()
        mock_torch.tensor.return_value = MagicMock(unsqueeze=MagicMock(return_value=MagicMock(to=MagicMock(return_value=MagicMock(shape=[1, 10])))))
        mock_torch.inference_mode.return_value.__enter__ = MagicMock()
        mock_torch.inference_mode.return_value.__exit__ = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.StoppingCriteria = object
        mock_transformers.StoppingCriteriaList = MagicMock()

        orig_torch = sys.modules.get("torch")
        orig_trans = sys.modules.get("transformers")

        sys.modules["torch"] = mock_torch
        sys.modules["transformers"] = mock_transformers

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()

        mock_processor.preprocess.return_value = {"pixel_values": MagicMock(to=MagicMock(return_value=MagicMock()))}
        mock_tokenizer.return_value = MagicMock(input_ids=MagicMock(to=MagicMock(return_value=MagicMock(shape=[1, 10]))))
        mock_tokenizer.batch_decode.return_value = ["Mock GeoChat response for remote sensing image."]
        mock_tokenizer.bos_token_id = 1

        MODEL_STATE["model"] = mock_model
        MODEL_STATE["tokenizer"] = mock_tokenizer
        MODEL_STATE["image_processor"] = mock_processor
        MODEL_STATE["ready"] = True
        MODEL_STATE["device"] = "cpu"
        MODEL_STATE["load_error"] = None

        yield

        MODEL_STATE.clear()
        MODEL_STATE.update(orig_state)
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch
        else:
            sys.modules.pop("torch", None)

        if orig_trans is not None:
            sys.modules["transformers"] = orig_trans
        else:
            sys.modules.pop("transformers", None)

    def test_health_endpoint(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "geochat-7b"
        assert data["ready"] is True

    def test_analyze_endpoint_success(self):
        client = TestClient(app)

        img = Image.new("RGB", (64, 64), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"image": ("test.png", buf.getvalue(), "image/png")}
        data = {"question": "Describe this scene."}

        resp = client.post("/v1/analyze", files=files, data=data)
        assert resp.status_code == 200
        result = resp.json()
        assert "answer" in result
        assert result["model"] == "geochat-7b"

    def test_analyze_empty_image_error(self):
        client = TestClient(app)
        files = {"image": ("empty.png", b"", "image/png")}
        data = {"question": "Describe this scene."}

        resp = client.post("/v1/analyze", files=files, data=data)
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_analyze_when_model_not_ready(self):
        MODEL_STATE["ready"] = False
        MODEL_STATE["load_error"] = "CUDA out of memory"

        client = TestClient(app)
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"image": ("test.png", buf.getvalue(), "image/png")}
        data = {"question": "Describe"}

        resp = client.post("/v1/analyze", files=files, data=data)
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()
