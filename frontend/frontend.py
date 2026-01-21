"""
Streamlit Frontend for Wheat Detection Application
===================================================

This is the web interface for wheat head detection using YOLOv11.

Setup:
------
1. Make sure backend is running (python backend.py)
2. Install: pip install streamlit requests pillow
3. Run: streamlit run frontend.py

App will open at: http://localhost:8501
"""

import streamlit as st
import requests
from PIL import Image
import io
import time

# Page Configuration
st.set_page_config(
    page_title="Wheat Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 3.5rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stats-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .detection-number {
        font-size: 3rem;
        color: #2E7D32;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🌾 Wheat Head Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Wheat Detection using YOLOv11</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Confidence slider
    confidence = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Lower = more detections, Higher = more confident detections"
    )
    
    st.markdown("---")
    
    # API Status Check
    st.subheader("🔌 Backend Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded"):
                st.success("✅ Connected & Model Loaded")
            else:
                st.warning("⚠️ Connected but Model Not Loaded")
        else:
            st.error("❌ Backend Error")
    except:
        st.error("❌ Backend Not Running")
        st.info("Start backend: `python backend.py`")
    
    st.markdown("---")
    
    # Model Info
    with st.expander("📊 Model Information"):
        try:
            response = requests.get(f"{API_URL}/model-info", timeout=2)
            if response.status_code == 200:
                info = response.json()
                st.write(f"**Type:** {info.get('model_type', 'N/A')}")
                st.write(f"**Input Size:** {info.get('input_size', 'N/A')}px")
                st.write(f"**Classes:** {', '.join(info.get('classes', {}).values())}")
        except:
            st.write("Model info unavailable")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.info(
        "This application uses YOLOv11 deep learning model to detect "
        "wheat heads in images. Upload an image and adjust the confidence "
        "threshold to optimize detection results."
    )
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & FastAPI")

# Main Content Area
col1, col2 = st.columns([1, 1], gap="large")

# Left Column - Upload
with col1:
    st.subheader("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a wheat image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing wheat heads for detection",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        
        # Image info
        st.caption(f"📏 Size: {image.size[0]} × {image.size[1]} pixels")
        st.caption(f"📁 File: {uploaded_file.name}")
        
        # Detect Button
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Detect Wheat Heads", type="primary"):
            
            with st.spinner("🌾 Analyzing image... Please wait"):
                
                try:
                    # Prepare request
                    uploaded_file.seek(0)
                    files = {
                        "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                    }
                    params = {"confidence": confidence}
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔄 Uploading image...")
                    progress_bar.progress(30)
                    
                    # Call API
                    response = requests.post(
                        f"{API_URL}/detect",
                        files=files,
                        params=params,
                        timeout=60
                    )
                    
                    status_text.text("🤖 Running AI detection...")
                    progress_bar.progress(70)
                    
                    time.sleep(0.5)
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.result = result
                        st.session_state.detection_done = True
                        
                        # Success message
                        if result["count"] > 0:
                            st.success(f"✅ Found {result['count']} wheat heads!")
                        else:
                            st.warning("⚠️ No wheat heads detected. Try lowering the confidence threshold.")
                    
                    else:
                        st.error(f"❌ Detection failed: {response.text}")
                        st.session_state.detection_done = False
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend!")
                    st.info("Make sure backend is running: `python backend.py`")
                    st.session_state.detection_done = False
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.detection_done = False

# Right Column - Results
with col2:
    st.subheader("📊 Detection Results")
    
    if hasattr(st.session_state, 'detection_done') and st.session_state.detection_done:
        result = st.session_state.result
        
        # Statistics Cards
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
                <div class="stats-card">
                    <div class="detection-number">{result['count']}</div>
                    <div class="metric-label">Wheat Heads Detected</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            avg_conf = result['stats']['avg_confidence']
            st.markdown(f"""
                <div class="stats-card">
                    <div class="detection-number">{avg_conf:.2%}</div>
                    <div class="metric-label">Avg. Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Image dimensions
        st.info(f"📐 Image Size: {result['image_size']['width']} × {result['image_size']['height']} pixels")
        
        # Display Result Image
        try:
            img_filename = result['output_image']
            img_response = requests.get(f"{API_URL}/download/{img_filename}")
            
            if img_response.status_code == 200:
                result_image = Image.open(io.BytesIO(img_response.content))
                st.image(result_image, caption="🎯 Detection Results", use_container_width=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=img_response.content,
                    file_name=f"wheat_detection_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            else:
                st.error("Failed to load result image")
        
        except Exception as e:
            st.error(f"Error displaying results: {e}")
        
        # Detailed Detections
        if result['count'] > 0:
            with st.expander(f"🔍 View All {result['count']} Detections"):
                for detection in result['detections']:
                    col1_det, col2_det = st.columns([3, 1])
                    
                    with col1_det:
                        st.write(f"**Detection #{detection['id']}**")
                        bbox = detection['bbox']
                        st.write(f"📍 Position: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) → ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
                    
                    with col2_det:
                        conf_percent = detection['confidence'] * 100
                        st.metric("Confidence", f"{conf_percent:.1f}%")
                    
                    st.markdown("---")
        
        # Export Data Option
        if st.button("📄 Export Detection Data (JSON)"):
            import json
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"detection_data_{int(time.time())}.json",
                mime="application/json"
            )
    
    else:
        # Placeholder when no results
        st.info("👈 Upload an image and click 'Detect Wheat Heads' to see results here")
        
        # Sample instructions
        with st.expander("📖 How to Use"):
            st.markdown("""
            **Step 1:** Upload a wheat image using the file uploader
            
            **Step 2:** Adjust the confidence threshold if needed
            - Lower (0.1-0.2): More detections, may include false positives
            - Higher (0.4-0.6): Fewer detections, more confident results
            
            **Step 3:** Click "Detect Wheat Heads"
            
            **Step 4:** View results and download the annotated image
            """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>🌾 Wheat Detection System | "
    "Powered by YOLOv11 | FastAPI + Streamlit</p>",
    unsafe_allow_html=True
)