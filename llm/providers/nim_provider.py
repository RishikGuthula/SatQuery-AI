"""
NVIDIA NIM LLM Provider.

Integrates with NVIDIA NIM (Inference Microservice) using OpenAI-compatible API
for Nemotron 3.5 Lightning (nvidia/nemotron-3.5-lightning-30b-a3b).
"""

from __future__ import annotations

import logging
import re
from typing import Any
import requests
from openai import OpenAI

from llm.base import LLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_NIM_TIMEOUT = 120


def _sanitize_error(error_msg: str, api_key: str | None = None) -> str:
    """Sanitize error message to prevent accidental credential leakage."""
    sanitized = str(error_msg)
    if api_key and len(api_key) > 4:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    sanitized = re.sub(r'Bearer\s+[A-Za-z0-9_\-\.]+', 'Bearer [REDACTED]', sanitized)
    sanitized = re.sub(r'api_key=[A-Za-z0-9_\-\.]+', 'api_key=[REDACTED]', sanitized)
    return sanitized


class NvidiaNIMProvider(LLMProvider):
    """NVIDIA NIM provider using OpenAI-compatible client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_NIM_BASE_URL,
        model: str = DEFAULT_NIM_MODEL,
        timeout: int = DEFAULT_NIM_TIMEOUT,
    ):
        super().__init__(
            api_key=api_key.strip() if api_key else "",
            model=model.strip() if model else DEFAULT_NIM_MODEL,
            timeout=timeout if timeout > 0 else DEFAULT_NIM_TIMEOUT,
        )
        self.base_url = (base_url or DEFAULT_NIM_BASE_URL).rstrip("/")
        self._client: OpenAI | None = None

        if self.is_available():
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=float(self.timeout),
            )

    def is_available(self) -> bool:
        """Check if provider has a configured API key."""
        return bool(self.api_key and self.api_key.strip())

    def health_check(self) -> dict[str, Any]:
        """
        Check health/readiness of NVIDIA NIM endpoint.
        Does not leak credentials in response or errors.
        """
        if not self.is_available():
            return {
                "status": "unconfigured",
                "healthy": False,
                "model": self.model,
                "base_url": self.base_url,
                "message": "NVIDIA_API_KEY is not configured.",
            }

        # Try GET /v1/health/ready or models listing
        try:
            ready_url = f"{self.base_url}/health/ready"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(ready_url, headers=headers, timeout=min(5, self.timeout))
            if resp.status_code == 200:
                return {
                    "status": "healthy",
                    "healthy": True,
                    "model": self.model,
                    "base_url": self.base_url,
                    "endpoint": "health/ready",
                }
        except Exception:
            pass

        # Fallback to model check
        try:
            models = self.list_models()
            is_model_present = self.model in models if models else True
            return {
                "status": "healthy" if is_model_present else "model_unavailable",
                "healthy": True,
                "model": self.model,
                "base_url": self.base_url,
                "available_models_count": len(models),
            }
        except Exception as e:
            clean_err = _sanitize_error(str(e), self.api_key)
            logger.warning(f"NIM health check failed: {clean_err}")
            return {
                "status": "unhealthy",
                "healthy": False,
                "model": self.model,
                "base_url": self.base_url,
                "error": clean_err,
            }

    def list_models(self) -> list[str]:
        """
        Retrieve available models via GET /v1/models.
        """
        if not self.is_available() or self._client is None:
            return []

        try:
            resp = self._client.models.list()
            return [m.id for m in resp.data]
        except Exception as e:
            clean_err = _sanitize_error(str(e), self.api_key)
            logger.warning(f"Failed to list NIM models: {clean_err}")
            raise RuntimeError(f"NIM Model discovery failed: {clean_err}") from None

    def verify_model_available(self, model_name: str | None = None) -> bool:
        """Verify that the target model is available in NIM."""
        target = model_name or self.model
        try:
            models = self.list_models()
            return target in models
        except Exception:
            return False

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: str | None = None,
        enable_thinking: bool = False,
    ) -> LLMResponse:
        """
        Generate completion using NVIDIA NIM Nemotron model.
        Guarantees that reasoning_content is never exposed.
        """
        if not self.is_available() or self._client is None:
            raise ValueError("NVIDIA NIM is not configured with an API key.")

        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        extra_body: dict[str, Any] = {}
        if enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
            extra_body["reasoning_budget"] = 16384

        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format == "json":
            req_kwargs["response_format"] = {"type": "json_object"}

        if extra_body:
            req_kwargs["extra_body"] = extra_body

        try:
            completion = self._client.chat.completions.create(**req_kwargs)
        except Exception as e:
            clean_err = _sanitize_error(str(e), self.api_key)
            logger.error(f"NVIDIA NIM generation error: {clean_err}")
            raise RuntimeError(f"NIM API error: {clean_err}") from None

        if not completion.choices:
            return LLMResponse(content="", model=self.model)

        choice = completion.choices[0]
        # Strictly extract ONLY message.content (never reasoning_content)
        content = choice.message.content or ""

        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
                "total_tokens": getattr(completion.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=getattr(completion, "model", self.model),
            usage=usage,
            raw_response={"id": getattr(completion, "id", "")},
        )
