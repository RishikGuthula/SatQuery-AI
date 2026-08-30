"""
SatQuery AI — Streamlit Application.

A multimodal remote-sensing assistant that analyzes satellite imagery
using spectral indices and change-detection tools.
"""

import io
import logging
import time

import streamlit as st
from PIL import Image

from agent.controller import process_query
from core.image_loader import ImageLoadError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")

# ── Header ──────────────────────────────────────────────────────────
st.title("🛰️ SatQuery AI")
st.subheader("Multimodal Remote-Sensing Assistant")
st.markdown(
    "Analyze satellite imagery using spectral indices "
    "(NDVI, NDWI, NDBI), change detection, and visual heuristics."
)
st.markdown("---")

# ── Image Upload Section ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📷 Primary Image")
    image1_file = st.file_uploader(
        "Upload Optical / Multispectral Image",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="img1",
        help="Accepts RGB images (PNG, JPEG) and multispectral GeoTIFFs.",
    )
    if image1_file:
        st.image(image1_file, caption="Primary Image", use_container_width=True)

with col2:
    st.markdown("### 📷 Secondary Image (optional)")
    image2_file = st.file_uploader(
        "Upload second image for change detection",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="img2",
        help="Required for change detection. Upload a second image of the same area.",
    )
    if image2_file:
        st.image(image2_file, caption="Secondary Image", use_container_width=True)

st.markdown("---")

# ── Query Section ───────────────────────────────────────────────────
st.markdown("### 💬 Ask your question")
query = st.text_input(
    "Enter your query about the satellite image(s):",
    placeholder="e.g., Find water bodies, Detect vegetation, Compare these images",
)

# Example queries
with st.expander("📝 Example queries"):
    st.markdown(
        """
        **Single image:**
        - "Find water bodies in this image"
        - "Detect vegetation areas"
        - "Show built-up areas"
        - "Calculate NDVI"
        - "Describe this image"

        **Two images:**
        - "Detect changes between the two images"
        - "Compare these images"

        **Unsupported (will be flagged):**
        - "Predict next year's crop yield"
        - "Identify specific buildings"
        """
    )

# ── Analyze Button ──────────────────────────────────────────────────
if st.button("🔍 Analyze", type="primary"):
    if not image1_file:
        st.error("⚠️ Please upload at least a Primary Image before submitting.")
    elif not query.strip():
        st.error("⚠️ Please enter a question about the image(s).")
    else:
        with st.spinner("🔄 Analyzing image(s)..."):
            # Read raw bytes
            img1_bytes = image1_file.read()
            img2_bytes = image2_file.read() if image2_file else None
            fname1 = image1_file.name
            fname2 = image2_file.name if image2_file else ""

            # Execute analysis
            t_start = time.time()
            result = process_query(
                query=query,
                image1_bytes=img1_bytes,
                image2_bytes=img2_bytes,
                filename1=fname1,
                filename2=fname2,
            )
            elapsed = time.time() - t_start

        # ── Results ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Answer
        st.markdown("### 💡 Answer")
        if result.answer.startswith("⚠️") or result.answer.startswith("❌"):
            st.warning(result.answer)
        else:
            st.success(result.answer)

        # Evidence
        if result.evidence is not None:
            st.markdown("### 🗺️ Visual Evidence Map")
            st.image(
                result.evidence,
                caption=f"Evidence — {result.tool_used}",
                use_container_width=True,
            )

        # Technical Details
        with st.expander("🔧 Technical Details"):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Analysis Info**")
                st.markdown(f"- **Tool used:** {result.tool_used}")
                st.markdown(f"- **Processing time:** {elapsed:.3f}s")
                if result.index_name:
                    st.markdown(f"- **Index:** {result.index_name}")
                if result.confidence is not None:
                    st.markdown(f"- **Confidence:** {result.confidence:.2f}")
                else:
                    st.markdown("- **Confidence:** N/A (not applicable)")

            with detail_col2:
                st.markdown("**Image Info**")
                meta = result.metadata
                if "image1_dimensions" in meta:
                    st.markdown(f"- **Primary image:** {meta['image1_dimensions']}")
                if "image1_sensor_type" in meta:
                    st.markdown(f"- **Sensor type:** {meta['image1_sensor_type']}")
                if "image1_bands" in meta:
                    bands_str = ", ".join(meta["image1_bands"])
                    st.markdown(f"- **Bands:** {bands_str}")
                if "image2_dimensions" in meta:
                    st.markdown(f"- **Secondary image:** {meta['image2_dimensions']}")
                if "coverage_percent" in meta:
                    st.markdown(f"- **Coverage:** {meta['coverage_percent']}%")
                if "changed_percent" in meta:
                    st.markdown(f"- **Changed area:** {meta['changed_percent']}%")

            if meta.get("requires_multispectral"):
                st.info(
                    "ℹ️ This analysis used an RGB color heuristic because the input "
                    "image does not contain the spectral bands required for a true "
                    "satellite index calculation. For accurate results, provide a "
                    "multispectral GeoTIFF with the appropriate bands."
                )

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>SatQuery AI — Remote Sensing Analysis Tool | "
    "Spectral indices require appropriate satellite data (NIR, SWIR bands). "
    "RGB inputs use visual heuristics only.</small>",
    unsafe_allow_html=True,
)
