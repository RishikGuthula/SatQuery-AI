"""
Core data models for the SatQuery AI system.

Defines the unified RasterImage abstraction, execution tracing, session
context, and AnalysisResult models that flow through the entire pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """Detected sensor/modality type of the input raster."""
    RGB = "rgb"                          # Standard RGB image (PNG, JPEG, 3-band)
    MULTISPECTRAL = "multispectral"      # GeoTIFF with >3 bands or known satellite band names
    SAR = "sar"                          # Synthetic Aperture Radar (single-band intensity)
    UNKNOWN = "unknown"


class Intent(str, Enum):
    """Query intent categories."""
    WATER_DETECTION = "water_detection"
    VEGETATION_DETECTION = "vegetation_detection"
    BUILTUP_DETECTION = "builtup_detection"
    CHANGE_DETECTION = "change_detection"
    IMAGE_DESCRIPTION = "image_description"
    MULTI_FEATURE_ANALYSIS = "multi_feature_analysis"
    UNSUPPORTED = "unsupported"


@dataclass
class ExecutionStep:
    """A single executed step in the agent's plan for transparency."""
    step_number: int
    capability: str
    status: str                         # "success", "failed", "skipped", "fallback"
    description: str
    duration_seconds: float = 0.0
    output_summary: str = ""


@dataclass
class AgentTrace:
    """Full execution trace of the agent's reasoning and tool calls."""
    planner_type: str = "llm"           # "llm" or "deterministic"
    planned_intent: str = ""
    plan_reasoning: str = ""
    steps: list[ExecutionStep] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def add_step(
        self,
        capability: str,
        status: str,
        description: str,
        duration: float = 0.0,
        summary: str = "",
    ) -> None:
        step = ExecutionStep(
            step_number=len(self.steps) + 1,
            capability=capability,
            status=status,
            description=description,
            duration_seconds=round(duration, 3),
            output_summary=summary,
        )
        self.steps.append(step)


@dataclass
class ConversationTurn:
    """A single turn in conversational multi-turn context."""
    query: str
    answer: str
    tool_used: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class SessionContext:
    """Maintains lightweight session context for follow-up questions."""
    history: list[ConversationTurn] = field(default_factory=list)
    last_query: str = ""
    last_answer: str = ""
    last_tool_used: str = ""
    last_metrics: dict = field(default_factory=dict)

    def add_turn(self, query: str, answer: str, tool_used: str, metadata: dict | None = None) -> None:
        self.last_query = query
        self.last_answer = answer
        self.last_tool_used = tool_used
        self.last_metrics = metadata or {}
        self.history.append(
            ConversationTurn(
                query=query,
                answer=answer,
                tool_used=tool_used,
                metadata=metadata or {},
            )
        )
        # Limit history length
        if len(self.history) > 10:
            self.history = self.history[-10:]


@dataclass
class RasterImage:
    """
    Unified image data model.

    Abstracts over RGB (PIL) and multispectral (GeoTIFF) inputs
    so downstream tools don't need to know the source format.
    """
    data: np.ndarray          # Shape: (H, W, C) float32 or int
    bands: list[str]          # Band names, e.g. ["red", "green", "blue"] or ["B1", ...]
    crs: str | None = None    # Coordinate reference system (WKT or PROJ4)
    transform: Any = None     # Affine transform if available
    bounds: tuple | None = None  # (left, bottom, right, top)
    resolution: tuple | None = None  # (x_res, y_res)
    nodata: float | None = None
    dtype: str = "float32"
    sensor_type: SensorType = SensorType.RGB
    metadata: dict = field(default_factory=dict)
    width: int = 0
    height: int = 0

    def __post_init__(self):
        if self.data is not None:
            self.height, self.width = self.data.shape[:2]

    @property
    def num_bands(self) -> int:
        if self.data.ndim == 2:
            return 1
        return self.data.shape[2]

    BAND_ALIASES = {
        "red": ["red", "b04", "b4", "band 4", "band4", "r"],
        "green": ["green", "b03", "b3", "band 3", "band3", "g"],
        "blue": ["blue", "b02", "b2", "band 2", "band2", "b"],
        "nir": ["nir", "b08", "b8", "b8a", "band 8", "band8", "near_infrared"],
        "swir": ["swir", "swir1", "b11", "band 11", "band11"],
        "swir1": ["swir1", "swir", "b11", "band 11", "band11"],
        "swir2": ["swir2", "b12", "band 12", "band12"],
    }

    def has_band(self, name: str) -> bool:
        """Check if a named band is available (case-insensitive with sensor aliases)."""
        name_lower = name.lower().strip()
        aliases = self.BAND_ALIASES.get(name_lower, [name_lower])
        return any(b.lower().strip() in aliases for b in self.bands)

    def get_band_index(self, name: str) -> int | None:
        """Return index of named band, or None."""
        name_lower = name.lower().strip()
        aliases = self.BAND_ALIASES.get(name_lower, [name_lower])
        for i, b in enumerate(self.bands):
            if b.lower().strip() in aliases:
                return i
        return None

    def get_band(self, name: str) -> np.ndarray | None:
        """Return 2D array for the named band, or None if not found."""
        idx = self.get_band_index(name)
        if idx is None:
            return None
        if self.data.ndim == 2:
            return self.data
        return self.data[:, :, idx]

    def to_rgb(self) -> np.ndarray:
        """
        Extract/convert to an RGB uint8 array for display and visual reasoning.

        If data has named red, green, blue bands, extracts them in RGB order.
        If single-band (like mode I GeoTIFF), replicates to 3 channels with robust contrast stretching.
        """
        if self.has_band("red") and self.has_band("green") and self.has_band("blue"):
            r = self.get_band("red")
            g = self.get_band("green")
            b = self.get_band("blue")
            if r is not None and g is not None and b is not None:
                gray = np.stack([r, g, b], axis=-1)
            else:
                gray = self.data[:, :, :3] if self.data.ndim == 3 and self.data.shape[2] >= 3 else np.stack([self.data[:, :, 0]] * 3, axis=-1)
        elif self.data.ndim == 2:
            gray = np.stack([self.data] * 3, axis=-1)
        elif self.num_bands >= 3:
            gray = self.data[:, :, :3]
        else:
            gray = np.stack([self.data[:, :, 0]] * 3, axis=-1)

        # Normalize to 0-255 uint8 with percentile stretch for scientific dynamic range
        gray = gray.astype(np.float32)
        if np.all(np.isnan(gray)):
            return np.zeros((*gray.shape[:2], 3), dtype=np.uint8)

        p2 = float(np.nanpercentile(gray, 1))
        p98 = float(np.nanpercentile(gray, 99))
        if p98 - p2 < 1e-6:
            gmin, gmax = float(np.nanmin(gray)), float(np.nanmax(gray))
            if gmax - gmin < 1e-6:
                return np.zeros((*gray.shape[:2], 3), dtype=np.uint8)
            p2, p98 = gmin, gmax

        normalized = np.clip((gray - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        return normalized

    def to_pil(self) -> Image.Image:
        """Convert the raster's RGB representation to a PIL Image."""
        return Image.fromarray(self.to_rgb())


@dataclass
class AnalysisResult:
    """
    Structured result returned by the SatQuery agent.

    Fields that cannot be meaningfully calculated are set to None.
    """
    answer: str
    evidence: Image.Image | None = None
    mask: np.ndarray | None = None
    index_name: str | None = None
    confidence: float | None = None  # None = not calculable / honest
    tool_used: str = ""
    metadata: dict = field(default_factory=dict)
    trace: AgentTrace | None = None
    session_context: SessionContext | None = None


# Standard band-name aliases for common satellite sensors
SENTINEL2_BAND_NAMES = {
    0: "coastal",       # B1
    1: "blue",          # B2
    2: "green",         # B3
    3: "red",           # B4
    4: "rededge1",      # B5
    5: "rededge2",      # B6
    6: "rededge3",      # B7
    7: "nir",           # B8
    8: "nir2",          # B8A
    9: "water_vapour",  # B9
    10: "swir1",        # B11
    11: "swir2",        # B12
}
