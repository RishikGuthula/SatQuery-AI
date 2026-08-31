"""
SatQuery AI — Unified Agentic Remote-Sensing Assistant.

Streamlit application providing a single multimodal interface:
- Multi-sensor image loading (RGB, Multispectral GeoTIFF)
- Single-agent natural language understanding
- LLM structured query planning & multi-tool orchestration
- Remote GeoChat-7B GPU inference for visual reasoning
- Authoritative remote-sensing indices (NDVI, NDWI, NDBI) and change detection
- Transparent execution trace ("How the agent solved this")
"""

import logging
import time
import streamlit as st

from agent.controller import process_query
from core.image_loader import load_from_bytes
from core.models import SessionContext
from core.registry import get_registry
from llm.client import get_llm_client
from vlm.client import get_vlm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_display_image(uploaded_file):
    """
    Safely convert an uploaded file (JPEG, PNG, or mode I / 16-bit float GeoTIFF)
    to a 3-channel RGB uint8 PIL Image for Streamlit display only, preserving the
    original raw bytes and file pointer position for scientific processing.
    """
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        uploaded_file.seek(0)

        raster = load_from_bytes(data, filename=uploaded_file.name)
        return raster.to_pil()
    except Exception as e:
        logger.warning(f"Error preparing display image for {uploaded_file.name}: {e}")
        return None

st.set_page_config(
    page_title="SatQuery AI — Multimodal Remote-Sensing Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session Context in Streamlit state
if "session_context" not in st.session_state:
    st.session_state.session_context = SessionContext()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Sidebar: System Status & Capabilities ────────────────────────────
with st.sidebar:
    st.title("🛰️ SatQuery AI")
    st.markdown("**Single Agentic Multimodal Assistant**")
    st.markdown("---")

    st.markdown("### 🔌 System Readiness")

    # LLM Status
    llm = get_llm_client()
    if llm.is_available:
        st.success(f"🟢 **LLM Planner:** Active ({llm.provider.model})")
    else:
        st.info("🟡 **LLM Planner:** Offline (Rule-based fallback)")

    # GeoChat GPU Status
    vlm = get_vlm()
    if vlm.is_available():
        st.success("🟢 **GeoChat-7B GPU:** Connected")
    else:
        st.info("⚪ **GeoChat-7B GPU:** Standby / Offline")

    st.markdown("---")
    st.markdown("### 🛠️ Active Capabilities")
    reg = get_registry()
    for cap in reg.list_available():
        st.markdown(f"- **{cap.name.upper()}**: {cap.description[:45]}...")

    st.markdown("---")
    if st.button("🧹 Clear Conversation History", use_container_width=True):
        st.session_state.session_context = SessionContext()
        st.session_state.last_result = None
        st.rerun()

# ── Header ──────────────────────────────────────────────────────────
st.title("🛰️ SatQuery AI")
st.subheader("Multimodal Remote-Sensing Assistant")
st.markdown(
    "Upload satellite or aerial imagery and ask any question in natural language. "
    "The unified agent automatically selects authoritative spectral indices (NDVI, NDWI, NDBI), "
    "change detection algorithms, and remote GPU vision models (GeoChat-7B) to provide a grounded analysis."
)
st.markdown("---")

# ── Image Upload Section ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📷 Primary Image")
    image1_file = st.file_uploader(
        "Upload Optical / Multispectral / Multi-Modal Image",
        type=["tif", "tiff", "png", "jpg", "jpeg", "npz"],
        key="img1",
        help="Accepts standard RGB images (PNG, JPEG), multispectral GeoTIFFs, and 12-channel Sentinel-1+Sentinel-2 archives (.npz).",
    )
    if image1_file:
        disp1 = get_display_image(image1_file)
        if disp1 is not None:
            st.image(disp1, caption=f"Primary Image: {image1_file.name}", use_container_width=True)
        else:
            st.info(f"📄 Loaded file: {image1_file.name}")

with col2:
    st.markdown("### 📷 Secondary Image (optional)")
    image2_file = st.file_uploader(
        "Upload second image for change detection",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="img2",
        help="Upload a second image of the same area to run temporal change detection.",
    )
    if image2_file:
        disp2 = get_display_image(image2_file)
        if disp2 is not None:
            st.image(disp2, caption=f"Secondary Image: {image2_file.name}", use_container_width=True)
        else:
            st.info(f"📄 Loaded file: {image2_file.name}")

st.markdown("---")

# ── Query Section ───────────────────────────────────────────────────
st.markdown("### 💬 Ask the Agent")
query = st.text_input(
    "Enter your query about the satellite image(s):",
    placeholder="e.g., Find water bodies, Classify land cover with Sentinel-1 and Sentinel-2, Compare these images",
)

# Example queries expander
with st.expander("💡 Example Queries"):
    st.markdown(
        """
        * **Single Image Analysis:**
          - *"Find water bodies in this scene"*
          - *"Detect vegetation and compute NDVI"*
          - *"Show built-up areas and structures"*
          - *"Describe what you see in this satellite image"*
          - *"Find water and vegetation and explain the scene"*
        * **Multi-Modal Optical + SAR Analysis (BIFOLD RDNet Base):**
          - *"Classify the land cover using Sentinel-1 and Sentinel-2"*
          - *"Analyze this optical and SAR imagery"*
          - *(Note: BIFOLD requires 12 bands: Sentinel-1 VV, VH + Sentinel-2 B02-B08, B8A, B11, B12)*
        * **Dual Image Comparison (ChangeFormer):**
          - *"Compare these two images and highlight changes"*
          - *"Detect land-cover changes between before and after"*
        * **Conversational Follow-up:**
          - *"What about the vegetation coverage?"*
        """
    )


# ── Analyze Button & Pipeline ───────────────────────────────────────
if st.button("🔍 Analyze with Agent", type="primary", use_container_width=True):
    if not image1_file:
        st.error("⚠️ Please upload at least a Primary Image before running analysis.")
    elif not query.strip():
        st.error("⚠️ Please enter a question or instruction for the agent.")
    else:
        with st.spinner("🤖 Agent analyzing query, planning tools, and executing models..."):
            image1_file.seek(0)
            img1_bytes = image1_file.read()
            img2_bytes = None
            if image2_file:
                image2_file.seek(0)
                img2_bytes = image2_file.read()
            fname1 = image1_file.name
            fname2 = image2_file.name if image2_file else ""

            t_start = time.time()
            try:
                result = process_query(
                    query=query,
                    image1_bytes=img1_bytes,
                    image2_bytes=img2_bytes,
                    filename1=fname1,
                    filename2=fname2,
                    session_context=st.session_state.session_context,
                )
            except Exception as e:
                logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
                result = AnalysisResult(
                    answer=f"⚠️ Analysis could not be completed: {str(e)}",
                    evidence=None,
                    tool_used="error_handler",
                    metadata={"error": str(e)},
                )
            elapsed = time.time() - t_start
            st.session_state.last_result = result

# ── Render Results ──────────────────────────────────────────────────
if st.session_state.last_result is not None:
    result = st.session_state.last_result
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # Final Grounded Answer
    st.markdown("### 💡 Agent Synthesis")
    if result.answer.startswith("⚠️"):
        st.warning(result.answer)
    elif result.answer.startswith("❌"):
        st.error(result.answer)
    else:
        st.success(result.answer)

    # Visual Evidence Overlay Map
    if result.evidence is not None:
        st.markdown("### 🗺️ Visual Evidence Map")
        st.image(
            result.evidence,
            caption=f"Evidence Map — {result.tool_used}",
            use_container_width=True,
        )

    # ── Transparency: How the agent solved this ─────────────────────
    if result.trace is not None and result.trace.steps:
        with st.expander("🔍 How the agent solved this (Execution Trace)", expanded=True):
            st.markdown(f"**Planner Backend:** `{result.trace.planner_type.upper()}`")
            st.markdown(f"**Identified Intent:** `{result.trace.planned_intent}`")
            st.markdown(f"**Plan Rationale:** {result.trace.plan_reasoning}")
            st.markdown("#### Execution Steps:")

            for step in result.trace.steps:
                status_icon = "✅" if step.status == "success" else "⚠️" if step.status == "skipped" else "❌"
                st.markdown(
                    f"{status_icon} **Step {step.step_number}: {step.capability}** ({step.duration_seconds}s) — {step.description}"
                )
                if step.output_summary:
                    st.caption(f"↳ {step.output_summary}")

    # ── Technical Remote Sensing Details ────────────────────────────
    with st.expander("🔧 Technical & Remote Sensing Metadata"):
        dcol1, dcol2 = st.columns(2)

        with dcol1:
            st.markdown("**Analysis Information**")
            st.markdown(f"- **Primary tool:** {result.tool_used}")
            st.markdown(f"- **Total processing time:** {result.metadata.get('processing_time_seconds', 0.0):.3f}s")
            if result.index_name:
                st.markdown(f"- **Spectral index:** {result.index_name}")

        with dcol2:
            st.markdown("**Image Specifications**")
            meta = result.metadata
            if "image1_dimensions" in meta:
                st.markdown(f"- **Primary image size:** {meta['image1_dimensions']}")
            if "image1_sensor_type" in meta:
                st.markdown(f"- **Sensor modality:** {meta['image1_sensor_type']}")
            if "image1_bands" in meta:
                st.markdown(f"- **Detected bands:** {', '.join(meta['image1_bands'])}")
            if "coverage_percent" in meta:
                st.markdown(f"- **Computed feature coverage:** {meta['coverage_percent']}%")
            if "changed_percent" in meta:
                st.markdown(f"- **Changed area:** {meta['changed_percent']}%")
            elif "change_percentage" in meta:
                st.markdown(f"- **Changed area:** {meta['change_percentage']}%")
            if "alignment_method" in meta:
                st.markdown(f"- **Pair alignment method:** {meta['alignment_method']}")

        if result.metadata.get("requires_multispectral"):
            st.info(
                "ℹ️ **Scientific Note:** This analysis computed an RGB color heuristic because "
                "the input file lacks multispectral bands (NIR/SWIR). For true spectral index calculations, "
                "provide a multispectral GeoTIFF."
            )

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>SatQuery AI — Single Agentic Multimodal Remote-Sensing Assistant | "
    "Scientific calculations are authoritative. RGB inputs use visual proxies only.</small>",
    unsafe_allow_html=True,
)
