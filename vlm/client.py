"""
VLM Client factory and management.
"""

from __future__ import annotations

import os
import logging

from vlm.base import VisionLanguageModel, RuleBasedVLM
from vlm.geochat import GeoChatVLM

logger = logging.getLogger(__name__)


def get_vlm() -> VisionLanguageModel:
    """
    Instantiate the appropriate VLM client based on environment variables:
    - GEOCHAT_ENABLED: "true" / "1"
    - GEOCHAT_API_URL: HTTPS URL of the remote GPU inference service
    - GEOCHAT_API_KEY: Authentication token
    - GEOCHAT_TIMEOUT: Timeout in seconds (default: 120)
    """
    enabled_str = os.environ.get("GEOCHAT_ENABLED", "true").strip().lower()
    is_enabled = enabled_str in ("true", "1", "yes")

    api_url = os.environ.get("GEOCHAT_API_URL", "").strip()
    api_key = os.environ.get("GEOCHAT_API_KEY", "").strip()
    timeout_str = os.environ.get("GEOCHAT_TIMEOUT", "120").strip()

    try:
        timeout = int(timeout_str)
    except ValueError:
        timeout = 120

    if is_enabled and api_url:
        logger.info(f"GeoChat VLM configured with endpoint: {api_url}")
        return GeoChatVLM(api_url=api_url, api_key=api_key, timeout=timeout)

    logger.debug("GeoChat not configured. Using rule-based visual fallback.")
    return RuleBasedVLM()
