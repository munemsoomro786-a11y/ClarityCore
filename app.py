import streamlit as st
import cv2
import numpy as np
import os
import requests
import threading
import time

# 1. Page Configuration
st.set_page_config(page_title="ClarityCore | Image Upscaler", page_icon="🖼️", layout="wide")

# 2. Multi-User Stability
if 'lock' not in st.session_state:
    st.session_state.lock = threading.Lock()

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# 3. FIXED MODEL LOADING (No more repeated downloads)
@st.cache_resource(show_spinner=False)
def load_sr_model():
    model_dir = "models"
    model_path = os.path.join(model_dir, "FSRCNN_x4.pb")
    
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # ONLY download if the file is NOT there
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model for the first time..."):
            url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                else:
                    st.error("Could not download model from GitHub.")
                    return None
            except Exception as e:
                st.error(f"Download error: {e}")
                return None

    # Load from the local file
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("fsrcnn", 4)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return sr
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# 4. UI logic
st.title("🚀 ClarityCore AI Upscaler")

file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    if img is not None:
        target_width = st.sidebar.slider("Target Width", 800, 3840, 1920)
        
        if st.sidebar.button("⚡ Upscale", type="primary", use_container_width=True):
            start_time = time.time()
            
            with st.session_state.lock:
                with st.spinner("AI Reconstruction..."):
                    sr = load_sr_model()
                    if sr:
                        # Process image
                        h, w = img.shape[:2]
                        work_img = img[:,:,:3] if len(img.shape) > 2 and img.shape[2] == 4 else img
                        
                        # AI Upscale
                        upscaled = sr.upsample(work_img)
                        
                        # Resize to final target width
                        result = cv2.resize(upscaled, (target_width, int(target_width*(h/w))), interpolation=cv2.INTER_LANCZOS4)
                        
                        # Re-add Alpha channel if it existed
                        if len(img.shape) > 2 and img.shape[2] == 4:
                            alpha = cv2.resize(img[:, :, 3], (result.shape[1], result.shape[0]), interpolation=cv2.INTER_AREA)
                            result = cv2.merge([result, alpha])
                        
                        st.session_state.processed_image = result
                        duration = time.time() - start_time
                        st.sidebar.success(f"Fixed in {duration:.2f}s")

        # Display Result
        if st.session_state.processed_image is not None:
            res = st.session_state.processed_image
            # Convert colors for Streamlit display
            disp_img = cv2.cvtColor(res, cv2.COLOR_BGRA2RGBA if len(res.shape)>2 and res.shape[2]==4 else cv2.COLOR_BGR2RGB)
            st.image(disp_img, use_container_width=True)
            
            # Download buttons
            col_png, col_jpg = st.columns(2)
            with col_png:
                _, b_png = cv2.imencode('.png', res, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                st.download_button("Download PNG", b_png.tobytes(), "clarityCore.png", "image/png")
            with col_jpg:
                work_res = res[:,:,:3] if len(res.shape)>2 and res.shape[2]==4 else res
                _, b_jpg = cv2.imencode('.jpg', work_res, [cv2.IMWRITE_JPEG_QUALITY, 85])
                st.download_button("Download JPG", b_jpg.tobytes(), "clarityCore.jpg", "image/jpeg")
else:
    st.info("Upload an image to start.")