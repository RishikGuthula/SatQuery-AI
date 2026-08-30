"""
Base definitions and interfaces for Online LLM providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """Chat message structure."""
    role: str                       # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract interface for LLM backends (OpenAI, Gemini, etc.)."""

    def __init__(self, api_key: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 1000,
        response_format: str | None = None,  # "json" or None
    ) -> LLMResponse:
        """Generate text completion from messages."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and credentials exist."""
        ...
