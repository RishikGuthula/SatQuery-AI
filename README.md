# 🛰️ SatQuery AI

**Multimodal Remote-Sensing Assistant for Satellite Image Analysis**

SatQuery AI is an interactive tool that analyzes satellite imagery using spectral indices (NDVI, NDWI, NDBI), change detection, and visual heuristics. It provides clear, honest analysis with visual evidence maps.

---

## Problem

Remote-sensing analysis traditionally requires specialized GIS software and expertise. Satellite imagery interpretation is complex, and many tools either oversimplify results or fabricate confidence metrics. SatQuery AI addresses this by providing **scientifically honest** analysis tools with clear separation between true spectral calculations and RGB-based visual heuristics.

## Solution

SatQuery AI uses a modular architecture:

1. **Query Planner** — Routes user queries to the appropriate analysis tool
2. **Spectral Tools** — Compute real NDVI/NDWI/NDBI when multispectral bands are available
3. **Change Detection** — Compares two images to detect land-cover changes
4. **Evidence Engine** — Generates clear visual overlays showing what was detected
5. **VLM Interface** — Abstract interface for future Vision-Language Model integration

```
┌──────────────────┐
│   Streamlit UI   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Query Planner   │  ← Intent classification
└────────┬─────────┘
         │
    ┌────┼────────────┐
    ▼    ▼            ▼
  Water  Veg.    Change
  (NDWI) (NDVI)  Detection
    │    │            │
    └────┼────────────┘
         │
         ▼
┌──────────────────┐
│ Evidence Engine  │  ← Visual overlay
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Analysis Result  │
└──────────────────┘
```

## Features

### ✅ Implemented
- **Water detection** — True NDWI (McFeeters 1996) for multispectral, RGB color proxy for standard images
- **Vegetation detection** — True NDVI (Rouse et al. 1974) for multispectral, greenness proxy for RGB
- **Built-up detection** — True NDBI (Zha et al. 2003) for multispectral, urban color proxy for RGB
- **Change detection** — Pixel-difference pipeline for image pairs
- **Visual evidence maps** — Clear overlay showing detected areas
- **Multispectral GeoTIFF support** — Full raster loading with band metadata
- **Honest reporting** — Clearly labels heuristics vs. true spectral calculations

### 🔬 Experimental
- **RGB visual proxies** — Color-based approximations (NOT equivalent to true indices)
- **Image pair alignment** — Basic resize-based alignment for different-resolution images

### 🗺️ Planned
- **Vision-Language Model integration** — GPT-4V, LLaVA, or similar for natural language understanding
- **SAR support** — Synthetic Aperture Radar analysis
- **Temporal analysis** — Multi-date time series
- **Advanced registration** — Affine/projective image registration
- **Object detection** — Building/road/vehicle detection

## Supported Inputs

| Input Type | Bands Required | Analysis Supported |
|------------|---------------|-------------------|
| RGB (PNG, JPEG) | Red, Green, Blue | Visual proxies only (clearly labeled) |
| GeoTIFF (multispectral) | NIR + Red (NDVI), Green + NIR (NDWI), SWIR + NIR (NDBI) | True spectral indices |
| Sentinel-2 / Landsat | All bands | Full analysis |
| SAR | — | Detected but not yet analyzed |

## Installation

```bash
# Clone repository
git clone https://github.com/RishikGuthula/SatQuery-AI.git
cd SatQuery-AI

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

### Example Queries

**Single image analysis:**
- "Find water bodies in this image"
- "Detect vegetation areas"
- "Show built-up areas"
- "Calculate NDVI"
- "Describe this image"

**Change detection (requires two images):**
- "Detect changes between the two images"
- "Compare these images"

### Understanding Results

**True spectral index** (multispectral input):
```
Water detection using NDWI (McFeeters 1996).
Water coverage: 17.4% of image area (12,345 of 71,000 pixels).
Threshold: > 0.00
Method: True spectral index using Green and NIR bands.
```

**RGB heuristic** (standard image):
```
Water detection using RGB color heuristic (visual proxy).
Estimated water-like area: 15.2% of image area.
⚠️ This is NOT a true NDWI calculation. True NDWI requires
near-infrared (NIR) spectral data.
```

## Project Structure

```
SatQuery-AI/
├── app.py                    # Streamlit application
├── agent/
│   ├── __init__.py
│   └── controller.py         # Main pipeline orchestrator
├── core/
│   ├── __init__.py
│   ├── models.py             # RasterImage, AnalysisResult, Intent
│   ├── planner.py            # Query router / intent classifier
│   └── image_loader.py       # Unified image loading (PIL + rasterio)
├── tools/
│   ├── __init__.py
│   ├── registry.py           # Tool registry
│   ├── spectral.py           # Spectral index calculations
│   ├── water_detection.py    # Water detection tool
│   ├── vegetation_detection.py
│   ├── builtup_detection.py
│   └── change_detection.py   # Change detection pipeline
├── evidence/
│   ├── __init__.py
│   └── engine.py             # Visual evidence generation
├── vlm/
│   ├── __init__.py
│   └── base.py               # VLM abstraction (for future integration)
├── tests/
│   ├── test_router.py
│   ├── test_spectral.py
│   ├── test_change_detection.py
│   ├── test_image_loader.py
│   └── test_controller.py
├── .github/workflows/ci.yml  # CI pipeline
├── requirements.txt
└── README.md
```

## Architecture Principles

1. **AI decides WHAT** — The query planner determines intent
2. **Tools decide HOW** — Spectral tools perform the actual calculation
3. **Evidence shows WHERE** — The evidence engine visualizes results
4. **Honest reporting** — No fabricated confidence, no fake NDVI for RGB

## Limitations

- **RGB images** cannot be used for true NDVI/NDWI/NDBI (requires NIR/SWIR bands)
- **Change detection** uses simple pixel-difference (not advanced temporal analysis)
- **No SAR analysis** currently (SAR inputs are detected but not processed)
- **Image alignment** is basic (resize to match dimensions, not true georegistration)
- **No real VLM** currently integrated (interface is ready for future implementation)
- RGB "proxies" are **visual heuristics only** and should not be used for scientific analysis

## Evaluation Metrics

When multispectral data is available:
- **Coverage percentage** — Fraction of image classified as target class
- **Index statistics** — Mean, min, max of the computed index
- **Mask quality** — Binary classification at standard thresholds

For RGB inputs, these metrics reflect the heuristic approximation and are not scientifically validated.

## Roadmap

- [ ] Integrate VLM (GPT-4V / LLaVA) for natural language queries
- [ ] Add SAR analysis (sigma0 backscatter, speckle filtering)
- [ ] Implement advanced image registration for change detection
- [ ] Add multi-temporal analysis
- [ ] Support for custom band configurations
- [ ] Export results as GeoTIFF / Shapefile
- [ ] Batch processing for large areas

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_spectral.py -v
pytest tests/test_controller.py -v
```

## License

This project was developed for educational/hackathon purposes.

## Credits

Developed by the SatQuery AI team. Built with Streamlit, NumPy, Pillow, and Rasterio.
