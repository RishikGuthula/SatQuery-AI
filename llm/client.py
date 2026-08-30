"""
Unified LLM Client Factory.

Instantiates configured LLM providers based on environment variables
and provides safety fallback checks.
"""

from __future__ import annotations

import os
import logging

from llm.base import LLMProvider, LLMMessage, LLMResponse
from llm.providers.openai_provider import OpenAIProvider
from llm.providers.gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)


class LLMClient:
    """High-level client wrapper around configured LLMProvider."""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider

    @property
    def is_available(self) -> bool:
        return self.provider is not None and self.provider.is_available()

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: str | None = None,
    ) -> LLMResponse | None:
        """Generate completion if provider is available, else return None."""
        if not self.is_available or self.provider is None:
            logger.debug("LLMClient: Provider is unavailable or unconfigured.")
            return None
        try:
            return self.provider.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}. Falling back.")
            return None


def _load_env_if_present():
    """Load key-value pairs from .env file into os.environ if not already set."""
    for env_path in (".env", "../.env", os.path.join(os.path.dirname(__file__), "..", ".env")):
        if os.path.exists(env_path):
            try:
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip('"').strip("'")
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception:
                pass


def get_llm_client() -> LLMClient:
    """
    Construct an LLMClient based on environment variables:
    LLM_PROVIDER ("openai", "gemini")
    LLM_API_KEY (or GEMINI_API_KEY / GOOGLE_API_KEY)
    LLM_MODEL
    LLM_TIMEOUT
    """
    _load_env_if_present()

    provider_name = os.environ.get("LLM_PROVIDER", "").strip().lower()
    api_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    model = os.environ.get("LLM_MODEL", "").strip()
    timeout_str = os.environ.get("LLM_TIMEOUT", "60").strip()

    try:
        timeout = int(timeout_str)
    except ValueError:
        timeout = 60

    if not api_key:
        logger.debug("No LLM_API_KEY found. LLM client in offline/fallback mode.")
        return LLMClient(provider=None)

    if provider_name in ("openai", "chatgpt"):
        provider = OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini", timeout=timeout)
        return LLMClient(provider=provider)
    elif provider_name in ("gemini", "google"):
        provider = GeminiProvider(api_key=api_key, model=model or "gemini-3.6-flash", timeout=timeout)
        return LLMClient(provider=provider)
    elif api_key.startswith("sk-"):
        # Auto-detect OpenAI key
        logger.info("Auto-detected OpenAI API key format.")
        provider = OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini", timeout=timeout)
        return LLMClient(provider=provider)
    elif api_key.startswith("AIza") or "gemini" in provider_name or "google" in provider_name:
        # Auto-detect Gemini key
        logger.info("Auto-detected Gemini API key format.")
        provider = GeminiProvider(api_key=api_key, model=model or "gemini-3.6-flash", timeout=timeout)
        return LLMClient(provider=provider)

    # If key exists but provider wasn't specified, try Gemini first (free tier / standard)
    logger.info("Defaulting configured API key to Gemini provider.")
    provider = GeminiProvider(api_key=api_key, model=model or "gemini-3.6-flash", timeout=timeout)
    return LLMClient(provider=provider)
