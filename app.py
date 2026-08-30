import streamlit as st
from PIL import Image
import numpy as np
import sys
import os

# Add current folder to path so tools can be imported
sys.path.append(os.path.dirname(__file__))

try:
    from tools.spectral import analyze_image
    spectral_loaded = True
except Exception as e:
    spectral_loaded = False
    spectral_error = str(e)

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")

st.title("🛰️ SatQuery AI")
st.subheader("Interactive Vision-Language Assistant for Satellite Images")

st.markdown("---")

# Image Upload Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Primary Image")
    image1 = st.file_uploader("Upload Optical / SAR Image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="img1")

with col2:
    st.markdown("### Secondary Image (Optional)")
    image2 = st.file_uploader("Upload second image (for change detection or Optical+SAR)", type=["tif", "tiff", "png", "jpg", "jpeg"], key="img2")

# Query Section
st.markdown("### Ask your question")
query = st.text_input("Type your question in English", placeholder="Example: Where are the water bodies? or Show vegetation")

# Submit Button
if st.button("Analyze", type="primary"):
    if image1 is None:
        st.error("Please upload at least one image.")
    elif query.strip() == "":
        st.error("Please enter a question.")
    else:
        st.success("Processing started...")

        # Show uploaded images
        st.markdown("### Uploaded Images")
        cols = st.columns(2)

        with cols[0]:
            img1 = Image.open(image1)
            st.image(img1, caption="Primary Image", use_container_width=True)

        if image2:
            with cols[1]:
                img2 = Image.open(image2)
                st.image(img2, caption="Secondary Image", use_container_width=True)

        st.markdown("---")
        st.markdown("### Result")

        if not spectral_loaded:
            st.error(f"Could not load spectral tool. Error: {spectral_error}")
        else:
            try:
                # Important: reset file pointer
                image1.seek(0)
                result = analyze_image(image1, query)

                st.success(result["answer"])
                st.write(f"**Confidence:** {result['confidence']*100:.1f}%")

                if result["mask"] is not None:
                    st.markdown(f"### Visual Evidence ({result['index_name']})")
                    st.image(result["mask"], caption=result["index_name"], use_container_width=True, clamp=True)

                st.markdown("### Execution Trace")
                st.code(f"""Task: Spectral Analysis
Tool used: {result['index_name'] if result['index_name'] else 'None'}
Model: Spectral Indices (NDVI / NDWI / NDBI)
Confidence: {result['confidence']}""")

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.exception(e)

st.markdown("---")
st.caption("SatQuery AI | SIH 2026 Prototype")