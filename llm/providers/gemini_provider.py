"""
Google Gemini LLM provider implementation using REST API.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from llm.base import LLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini REST API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        timeout: int = 60,
    ):
        super().__init__(api_key=api_key, model=model or "gemini-3.6-flash", timeout=timeout)

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: str | None = None,
    ) -> LLMResponse:
        if not self.is_available():
            raise ValueError("Gemini API key is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        # Convert standard messages to Gemini contents format
        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max(max_tokens, 4096),
        }

        if response_format == "json":
            generation_config["responseMimeType"] = "application/json"

        payload: dict = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
            content = "".join(p.get("text", "") for p in parts)
            usage = data.get("usageMetadata", {})

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                raw_response=data,
            )
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
