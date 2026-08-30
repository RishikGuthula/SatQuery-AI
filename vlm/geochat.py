"""
GeoChat-7B Remote API Client.

Communicates with the standalone GeoChat-7B FastAPI GPU inference service
over authenticated HTTPS. Keeps the main application completely decoupled
from heavy PyTorch/CUDA dependencies.
"""

from __future__ import annotations

import io
import json
import logging
import time
from typing import Any

import requests
from PIL import Image

from core.models import AnalysisResult, RasterImage
from vlm.base import VisionLanguageModel

logger = logging.getLogger(__name__)


class GeoChatVLM(VisionLanguageModel):
    """Client for remote GeoChat-7B GPU inference service."""

    def __init__(
        self,
        api_url: str,
        api_key: str = "",
        timeout: int = 120,
        max_retries: int = 2,
    ):
        self.api_url = api_url.rstrip("/") if api_url else ""
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._cached_available: bool | None = None
        self._last_health_check_time: float = 0.0

    def _get_headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_available(self) -> bool:
        """Check if GeoChat API endpoint is configured and reachable."""
        if not self.api_url:
            return False

        # Cache health status for 30 seconds to avoid spamming the endpoint
        now = time.time()
        if self._cached_available is not None and (now - self._last_health_check_time < 30):
            return self._cached_available

        try:
            health_url = f"{self.api_url}/health"
            resp = requests.get(health_url, headers=self._get_headers(), timeout=5)
            is_ok = resp.status_code == 200 and resp.json().get("status") == "ok"
            self._cached_available = is_ok
            self._last_health_check_time = now
            return is_ok
        except Exception as e:
            logger.debug(f"GeoChat health check failed: {e}")
            self._cached_available = False
            self._last_health_check_time = now
            return False

    def health_check(self) -> dict[str, Any]:
        """Query detailed health status from remote server."""
        if not self.api_url:
            return {"available": False, "reason": "GEOCHAT_API_URL not configured"}
        try:
            health_url = f"{self.api_url}/health"
            resp = requests.get(health_url, headers=self._get_headers(), timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                data["available"] = True
                return data
            return {"available": False, "status_code": resp.status_code, "reason": resp.text}
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def analyze(
        self,
        query: str,
        image: RasterImage,
        analysis_result: AnalysisResult | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Send image and query to remote GeoChat service for visual inference.
        """
        if not self.api_url:
            logger.warning("GeoChat called but GEOCHAT_API_URL is not configured.")
            return "⚠️ GeoChat service is not configured (GEOCHAT_API_URL is empty)."

        analyze_url = f"{self.api_url}/v1/analyze"

        # Convert image to PNG bytes
        pil_img = image.to_pil()
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        question_text = query if query.strip() else "Describe this satellite image in detail."

        payload_context = {}
        if analysis_result and analysis_result.metadata:
            payload_context["tool_metrics"] = analysis_result.metadata
        if context:
            payload_context.update(context)

        files = {
            "image": ("image.png", img_bytes, "image/png"),
        }
        data = {
            "question": question_text,
        }
        if payload_context:
            data["context"] = json.dumps(payload_context)

        # Retry loop with exponential backoff
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Calling GeoChat API at {analyze_url} (attempt {attempt}/{self.max_retries})..."
                )
                t_start = time.time()
                resp = requests.post(
                    analyze_url,
                    files=files,
                    data=data,
                    headers=self._get_headers(),
                    timeout=self.timeout,
                )
                elapsed = time.time() - t_start

                if resp.status_code == 200:
                    res_json = resp.json()
                    raw_answer = res_json.get("answer", "").strip()
                    if raw_answer:
                        from vlm.response_cleaner import clean_vlm_response
                        answer = clean_vlm_response(raw_answer)
                        logger.info(f"GeoChat inference succeeded in {elapsed:.2f}s.")
                        return answer
                    else:
                        logger.warning("GeoChat returned an empty answer.")
                        return "⚠️ GeoChat returned an empty response."

                elif resp.status_code in (401, 403):
                    logger.error(f"GeoChat authentication failed (HTTP {resp.status_code}).")
                    return f"❌ GeoChat authentication failed. Please check GEOCHAT_API_KEY."

                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    logger.warning(f"GeoChat request failed: {last_error}")

            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.timeout}s"
                logger.warning(f"GeoChat timeout on attempt {attempt}: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"GeoChat connection error on attempt {attempt}: {last_error}")

            if attempt < self.max_retries:
                time.sleep(1.5 * attempt)

        return (
            f"⚠️ GeoChat remote visual reasoning service unavailable: {last_error}\n\n"
            f"The system continued using authoritative local analysis."
        )
