"""
SatQuery AI — Unified Agentic Remote-Sensing Assistant.

Streamlit application providing a production-grade multimodal interface:
- Multi-sensor image loading (RGB, Multispectral GeoTIFF, Sentinel-1+2 NPZ)
- Single-agent natural language understanding & domain validation
- Structured query planning & multi-tool orchestration
- Remote GeoChat-7B GPU inference for visual reasoning
- Authoritative remote-sensing indices (NDVI, NDWI, NDBI) and change detection (ChangeFormer)
- Grounded structured synthesis with scannable observations, evidence, and confidence
- Transparent execution tracing ("How the agent solved this")
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


# Page configuration
st.set_page_config(
    page_title="SatQuery AI — Multimodal Remote-Sensing Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Production-Grade CSS
st.markdown(
    """
    <style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Banner / Header */
    .sat-header-container {
        padding: 1.5rem 1.5rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
        color: #f8fafc;
    }
    .sat-header-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sat-header-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }

    /* Status Badges */
    .sat-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    .sat-badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .sat-badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .sat-badge-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .sat-badge-purple {
        background-color: rgba(139, 92, 246, 0.15);
        color: #a855f7;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    /* Structured Cards */
    .sat-card {
        background-color: #0b132b;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .sat-summary-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .sat-section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sat-observation-item {
        padding: 0.4rem 0;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.5;
        border-bottom: 1px solid rgba(51, 65, 85, 0.4);
    }
    .sat-observation-item:last-child {
        border-bottom: none;
    }
    .sat-evidence-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: #10b981;
        font-size: 0.88rem;
        padding: 0.2rem 0;
    }
    .sat-confidence-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #38bdf8;
    }

    /* Upload Container */
    .sat-upload-box {
        border: 1px dashed #334155;
        border-radius: 8px;
        padding: 1rem;
        background-color: rgba(15, 23, 42, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session Context in Streamlit state
if "session_context" not in st.session_state:
    st.session_state.session_context = SessionContext()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

# ── Sidebar: System Readiness & Tools ────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ SatQuery AI")
    st.caption("Agentic Remote-Sensing Intelligence")
    st.markdown("---")

    st.markdown("### 🔌 System Readiness")

    # LLM Planner Status
    llm = get_llm_client()
    if llm.is_available:
        model_name = getattr(llm.provider, "model", "Online")
        st.markdown(f'<span class="sat-badge sat-badge-success">🟢 LLM Planner Active</span>', unsafe_allow_html=True)
        st.caption(f"Model: `{model_name}`")
    else:
        st.markdown(f'<span class="sat-badge sat-badge-warning">🟡 LLM Planner Offline</span>', unsafe_allow_html=True)
        st.caption("Fallback: Deterministic Rule-Based Planner")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # GeoChat GPU Status
    vlm = get_vlm()
    if vlm.is_available():
        st.markdown(f'<span class="sat-badge sat-badge-purple">🟢 GeoChat-7B Active</span>', unsafe_allow_html=True)
        st.caption("Remote GPU vision reasoning")
    else:
        st.markdown(f'<span class="sat-badge sat-badge-info">⚪ GeoChat-7B Standby</span>', unsafe_allow_html=True)
        st.caption("Heuristic image overview fallback")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # Compute Device Detection
    reg = get_registry()
    try:
        import torch
        if torch.backends.mps.is_available():
            device_label = "Apple Silicon MPS"
        elif torch.cuda.is_available():
            device_label = f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            device_label = "CPU"
    except Exception:
        device_label = "CPU"

    # ChangeFormer Status
    cf_cap = reg.get("changeformer") or reg.get("change_detection")
    if cf_cap and cf_cap.is_available():
        st.markdown(f'<span class="sat-badge sat-badge-success">🟢 ChangeFormer Active</span>', unsafe_allow_html=True)
        st.caption(f"Bi-temporal Transformer ({device_label})")
    else:
        st.markdown(f'<span class="sat-badge sat-badge-info">⚪ ChangeFormer Standby</span>', unsafe_allow_html=True)

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # BIFOLD RDNet Status
    bifold_cap = reg.get("bifold") or reg.get("bifold_rdnet")
    if bifold_cap and bifold_cap.is_available():
        st.markdown(f'<span class="sat-badge sat-badge-success">🟢 BIFOLD RDNet Active</span>', unsafe_allow_html=True)
        st.caption(f"Optical + SAR 12-channel ({device_label})")
    else:
        st.markdown(f'<span class="sat-badge sat-badge-info">⚪ BIFOLD RDNet Standby</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Registered Capabilities")
    for cap in reg.list_available():
        st.markdown(f"- **{cap.name.upper()}**: {cap.description[:42]}...")

    st.markdown("---")
    if st.button("🧹 Clear Session Context", use_container_width=True):
        st.session_state.session_context = SessionContext()
        st.session_state.last_result = None
        st.session_state.active_query = ""
        st.rerun()

# ── Header Banner ───────────────────────────────────────────────────
st.markdown(
    """
    <div class="sat-header-container">
        <div class="sat-header-title">
            🛰️ SatQuery AI
            <span class="sat-badge sat-badge-info">Production Assistant</span>
        </div>
        <p class="sat-header-subtitle">
            Multimodal satellite imagery analysis powered by authoritative spectral calculations (NDVI, NDWI, NDBI), 
            ChangeFormer bi-temporal change detection, BIFOLD optical+SAR classification, and grounded LLM reasoning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Image Upload Section ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📷 Primary Satellite Image")
    image1_file = st.file_uploader(
        "Upload Optical / Multispectral / SAR Image",
        type=["tif", "tiff", "png", "jpg", "jpeg", "npz"],
        key="img1",
        help="Accepts standard RGB images (PNG, JPEG), multispectral GeoTIFFs (with Green, Red, NIR, SWIR), and 12-channel Sentinel-1+Sentinel-2 archives (.npz).",
    )
    if image1_file:
        disp1 = get_display_image(image1_file)
        if disp1 is not None:
            st.image(disp1, caption=f"Primary: {image1_file.name} ({disp1.width}x{disp1.height} px)", use_container_width=True)
        else:
            st.info(f"📄 Loaded file: {image1_file.name}")

with col2:
    st.markdown("#### 📷 Secondary Image (Optional for Change Detection)")
    image2_file = st.file_uploader(
        "Upload second temporal image or secondary spectral band",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="img2",
        help="Upload a second temporal image of the same region for bi-temporal change detection or a companion spectral GeoTIFF (e.g. B08 NIR).",
    )
    if image2_file:
        disp2 = get_display_image(image2_file)
        if disp2 is not None:
            st.image(disp2, caption=f"Secondary: {image2_file.name} ({disp2.width}x{disp2.height} px)", use_container_width=True)
        else:
            st.info(f"📄 Loaded file: {image2_file.name}")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ── Query Section with Quick Suggestion Pills ───────────────────────
st.markdown("#### 💬 Ask SatQuery AI")

# Quick suggestion chips
st.caption("Quick suggestions (click to load):")
sug_col1, sug_col2, sug_col3, sug_col4, sug_col5, sug_col6 = st.columns(6)

with sug_col1:
    if st.button("💧 Water (NDWI)", use_container_width=True):
        st.session_state.active_query = "Find water bodies in this scene"
with sug_col2:
    if st.button("🌿 Vegetation (NDVI)", use_container_width=True):
        st.session_state.active_query = "Detect vegetation and calculate NDVI"
with sug_col3:
    if st.button("🏙️ Built-up (NDBI)", use_container_width=True):
        st.session_state.active_query = "Find built-up urban structures"
with sug_col4:
    if st.button("🔄 ChangeFormer", use_container_width=True):
        st.session_state.active_query = "Compare these two images and detect changes"
with sug_col5:
    if st.button("🛰️ Optical + SAR", use_container_width=True):
        st.session_state.active_query = "Classify land cover using Sentinel-1 and Sentinel-2"
with sug_col6:
    if st.button("👁️ Describe Scene", use_container_width=True):
        st.session_state.active_query = "Describe what you see in this satellite image"

query = st.text_input(
    "Query input:",
    value=st.session_state.active_query,
    placeholder="e.g., What type of land use is visible? Find water bodies, Detect vegetation with NDVI...",
    label_visibility="collapsed",
)

# ── Analyze Button & Pipeline ───────────────────────────────────────
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
if st.button("🔍 Run Grounded Analysis", type="primary", use_container_width=True):
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
                    status="error",
                    summary=f"Analysis failed: {str(e)}",
                )
            st.session_state.last_result = result

# ── Render Structured Results ───────────────────────────────────────
if st.session_state.last_result is not None:
    result = st.session_state.last_result
    st.markdown("---")

    # Header with Status Badge
    status_str = result.status.lower() if hasattr(result, "status") and result.status else "success"
    if status_str == "out_of_scope":
        status_badge = '<span class="sat-badge sat-badge-warning">⚪ Out of Scope Query</span>'
    elif status_str == "insufficient_evidence":
        status_badge = '<span class="sat-badge sat-badge-warning">🟡 Insufficient Evidence</span>'
    elif status_str == "error":
        status_badge = '<span class="sat-badge sat-badge-warning">🔴 Error</span>'
    else:
        status_badge = '<span class="sat-badge sat-badge-success">🟢 Grounded Analysis Complete</span>'

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #f8fafc;">📊 Analysis Results</h3>
            {status_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Summary Card
    summary_text = getattr(result, "summary", "") or result.answer
    st.markdown(
        f"""
        <div class="sat-summary-card">
            <div class="sat-section-title">💡 Summary</div>
            <div style="color: #f1f5f9; font-size: 1rem; line-height: 1.6;">
                {summary_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Observations & Evidence Grid
    obs_list = getattr(result, "observations", [])
    feat_list = getattr(result, "key_visual_features", [])
    interpretation_text = getattr(result, "interpretation", "")
    limitations_list = getattr(result, "limitations", [])
    ev_list = getattr(result, "evidence_sources", []) or getattr(result, "evidence", [])
    if isinstance(ev_list, list) and ev_list and isinstance(ev_list[0], str):
        evidence_items = ev_list
    else:
        evidence_items = []

    conf_text = getattr(result, "confidence_level", "") or (f"{result.confidence * 100:.0f}%" if isinstance(result.confidence, float) else "Moderate")

    rcol1, rcol2 = st.columns([3, 2])

    with rcol1:
        if obs_list:
            st.markdown("#### 🔍 Key Observations")
            for obs in obs_list:
                st.markdown(f'<div class="sat-observation-item">• {obs}</div>', unsafe_allow_html=True)

        if feat_list:
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 🗺️ Visual Features & Land Cover")
            for feat in feat_list:
                st.markdown(f'<div class="sat-observation-item">• {feat}</div>', unsafe_allow_html=True)

        if interpretation_text:
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 💡 Interpretation & Analysis")
            st.markdown(f'<div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">{interpretation_text}</div>', unsafe_allow_html=True)

        if not obs_list and not feat_list and result.answer and not summary_text:
            st.markdown(result.answer)

    with rcol2:
        st.markdown("#### 📋 Evidence & Confidence")
        if evidence_items:
            for ev in evidence_items:
                st.markdown(f'<div class="sat-evidence-item">✓ {ev}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sat-evidence-item">✓ Primary satellite raster</div>', unsafe_allow_html=True)
            if result.tool_used:
                st.markdown(f'<div class="sat-evidence-item">✓ {result.tool_used}</div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"**Confidence Rating:**")
        st.markdown(f'<div class="sat-confidence-pill">{conf_text}</div>', unsafe_allow_html=True)

        if limitations_list:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("#### ⚠️ Limitations & Disclosures")
            for lim in limitations_list:
                st.markdown(f'<div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">⚠️ {lim}</div>', unsafe_allow_html=True)

    # 3. Visual Evidence Map Display
    if result.evidence is not None:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 🗺️ Visual Evidence Map")
        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            disp_orig = get_display_image(image1_file)
            if disp_orig is not None:
                st.image(disp_orig, caption="Original Input", use_container_width=True)
        with ev_col2:
            st.image(
                result.evidence,
                caption=f"Evidence Map — {result.tool_used} ({result.index_name or 'Detection Overlay'})",
                use_container_width=True,
            )

    # 4. Transparent Execution Trace ("How the agent solved this")
    if result.trace is not None and result.trace.steps:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        with st.expander("🔍 How the agent solved this (Transparent Execution Trace)", expanded=False):
            st.markdown(f"**Planner Backend:** `{result.trace.planner_type.upper()}`")
            st.markdown(f"**Identified Intent:** `{result.trace.planned_intent}`")
            st.markdown(f"**Plan Reasoning:** {result.trace.plan_reasoning}")
            st.markdown(f"**Total Execution Time:** `{result.trace.total_duration_seconds}s`")
            st.markdown("##### Execution Steps:")

            for step in result.trace.steps:
                status_icon = "✅" if step.status == "success" else "⚠️" if step.status == "skipped" else "❌"
                st.markdown(
                    f"{status_icon} **Step {step.step_number}: {step.capability}** ({step.duration_seconds}s) — {step.description}"
                )
                if step.output_summary:
                    st.caption(f"↳ {step.output_summary}")

    # 5. Technical Remote Sensing Metadata
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    with st.expander("🔧 Technical Remote Sensing & Scientific Metadata", expanded=False):
        dcol1, dcol2 = st.columns(2)
        meta = result.metadata or {}

        with dcol1:
            st.markdown("**Analysis Execution**")
            st.markdown(f"- **Primary tool:** `{result.tool_used}`")
            st.markdown(f"- **Processing time:** `{meta.get('processing_time_seconds', 0.0):.3f}s`")
            if result.index_name:
                st.markdown(f"- **Spectral index formula:** `{result.index_name}`")
            if getattr(result, "sources", None):
                st.markdown(f"- **Data & tool sources:** {', '.join(result.sources)}")

        with dcol2:
            st.markdown("**Raster Specifications**")
            if "image1_dimensions" in meta:
                st.markdown(f"- **Primary image size:** `{meta['image1_dimensions']}`")
            if "image1_sensor_type" in meta:
                st.markdown(f"- **Sensor modality:** `{meta['image1_sensor_type']}`")
            if "image1_bands" in meta:
                st.markdown(f"- **Detected spectral bands:** `{', '.join(meta['image1_bands'])}`")
            if "coverage_percent" in meta:
                st.markdown(f"- **Computed feature coverage:** **{meta['coverage_percent']}%**")
            if "changed_percent" in meta:
                st.markdown(f"- **Changed area:** **{meta['changed_percent']}%**")
            elif "change_percentage" in meta:
                st.markdown(f"- **Changed area:** **{meta['change_percentage']}%**")
            if "alignment_method" in meta:
                st.markdown(f"- **Pair alignment method:** `{meta['alignment_method']}`")

        if meta.get("requires_multispectral"):
            st.info(
                "ℹ️ **Scientific Integrity Note:** This analysis calculated an RGB color heuristic because "
                "the input file lacks multispectral bands (NIR/SWIR). For true spectral index calculations, "
                "upload a multispectral GeoTIFF."
            )

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.82rem;'>"
    "SatQuery AI — Unified Multimodal Remote-Sensing Assistant | "
    "Scientific calculations are authoritative. RGB inputs use visual proxies only."
    "</div>",
    unsafe_allow_html=True,
)
