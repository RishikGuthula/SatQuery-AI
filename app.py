import streamlit as st
from PIL import Image
from agent.controller import process_query

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")

st.title("🛰️ SatQuery AI")
st.subheader("Interactive Vision-Language Assistant for Satellite Images")
st.markdown("---")

# Image Upload Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Primary Image")
    image1_file = st.file_uploader("Upload Optical / SAR Image", type=["tif", "tiff", "png", "jpg", "jpeg"], key="img1")
    if image1_file:
        st.image(image1_file, caption="Primary Uploaded Image", use_container_width=True)

with col2:
    st.markdown("### Secondary Image")
    image2_file = st.file_uploader("Upload second image (for change detection or Optical+SAR)", type=["tif", "tiff", "png", "jpg", "jpeg"], key="img2")
    if image2_file:
        st.image(image2_file, caption="Secondary Uploaded Image", use_container_width=True)

st.markdown("---")

# Query Section
st.markdown("### Ask your question")
query = st.text_input("Enter your query about the satellite image(s):", placeholder="e.g., Highlight water bodies in this region")

if st.button("Analyze Query", type="primary"):
    if not image1_file:
        st.error("Please upload at least a Primary Image before submitting.")
    elif not query.strip():
        st.error("Please enter a question or query.")
    else:
        with st.spinner("Agent is analyzing image and routing tools..."):
            img1 = Image.open(image1_file)
            img2 = Image.open(image2_file) if image2_file else None
            
            # Execute agent workflow
            answer, evidence, trace = process_query(query, img1, img2)
            
            st.markdown("---")
            st.markdown("## Output Results")
            
            out_col1, out_col2 = st.columns(2)
            with out_col1:
                st.markdown("### Visual Evidence Map")
                st.image(evidence, caption="Generated Visual Evidence Map", use_container_width=True)
                
            with out_col2:
                st.markdown("### AI Answer")
                st.success(answer)
                
                st.markdown("### Execution Trace")
                st.code(trace)