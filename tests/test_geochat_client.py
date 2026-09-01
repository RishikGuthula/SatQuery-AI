"""
Tests for GeoChat-7B Remote API Client.
"""

import json
from unittest.mock import patch, MagicMock
import pytest
import requests

from core.models import RasterImage, SensorType
from vlm.geochat import GeoChatVLM


class TestGeoChatClient:
    """Test GeoChat API client network calls, retries, auth, and fallbacks."""

    def test_health_check_success(self):
        client = GeoChatVLM(api_url="https://mock-geochat.example.com", api_key="test-key")

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "ok", "model": "geochat-7b"}
            mock_get.return_value = mock_resp

            assert client.is_available() is True
            mock_get.assert_called_once()
            # Verify headers
            headers = mock_get.call_args[1]["headers"]
            assert headers["X-API-Key"] == "test-key"
            assert headers["Authorization"] == "Bearer test-key"

    def test_health_check_offline(self):
        client = GeoChatVLM(api_url="https://mock-geochat.example.com")

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            assert client.is_available() is False

    def test_analyze_success(self, rgb_image):
        client = GeoChatVLM(api_url="https://mock-geochat.example.com", api_key="test-key")

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "answer": "This satellite image depicts an agricultural area near a lake.",
                "model": "geochat-7b",
                "metadata": {"processing_time_seconds": 1.2}
            }
            mock_post.return_value = mock_resp

            ans = client.analyze("Describe this scene", rgb_image)

            assert "agricultural area" in ans
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert "files" in call_kwargs
            assert call_kwargs["data"]["question"] == "Describe this scene"

    def test_analyze_auth_failure(self, rgb_image):
        client = GeoChatVLM(api_url="https://mock-geochat.example.com", api_key="wrong-key")

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.text = "Unauthorized"
            mock_post.return_value = mock_resp

            ans = client.analyze("Describe this scene", rgb_image)

            assert "authentication failed" in ans.lower()

    def test_analyze_timeout_retry_and_graceful_fallback(self, rgb_image):
        client = GeoChatVLM(
            api_url="https://mock-geochat.example.com",
            timeout=5,
            max_retries=2,
        )

        with patch("requests.post", side_effect=requests.exceptions.Timeout("Timed out")):
            with patch("time.sleep"):  # Speed up test
                ans = client.analyze("Describe this scene", rgb_image)

                assert "service unavailable" in ans.lower() or "timed out" in ans.lower()
