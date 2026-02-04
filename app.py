import streamlit as st
import cv2
import numpy as np
import os
import requests
import threading
import time

# 1. Page Configuration
st.set_page_config(page_title="ClarityCore AI Upscaler", page_icon="🖼️", layout="wide")

# 2. Multi-User Stability: Threading Lock
# Ensures the server doesn't crash when multiple people use it at once.
if 'lock' not in st.session_state:
    st.session_state.lock = threading.Lock()

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# 3. Model Loading Logic
@st.cache_resource(show_spinner=False)
def load_sr_model():
    model_dir = "models"
    model_path = os.path.join(model_dir, "FSRCNN_x4.pb")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists(model_path):
        url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"
        try:
            response = requests.get(url, timeout=20)
            with open(model_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"Model Download Error: {e}")
            return None

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
    return sr

# 4. Optimized File Export
def to_bytes(image, fmt):
    if fmt == 'PNG':
        # Balancing size (target 20MB) and quality
        _, b = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    else:
        # High quality but low size (target <10MB)
        temp = image[:, :, :3] if len(image.shape) > 2 and image.shape[2] == 4 else image
        _, b = cv2.imencode('.jpg', temp, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return b.tobytes()

# 5. User Interface
st.title("🚀 ClarityCore AI Upscaler")
st.markdown("Natural AI upscaling for clear, high-resolution results.")

file = st.sidebar.file_uploader("Upload Image (Max 5MB)", type=['jpg', 'png', 'jpeg'])

if file:
    try:
        # Read and decode image safely
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            st.error("Invalid image. Please upload a standard JPG or PNG file.")
        else:
            target_width = st.sidebar.slider("Target Width", 800, 3840, 1920)
            
            if st.sidebar.button("⚡ Upscale", type="primary", use_container_width=True):
                # Timer starts here
                start_time = time.time()
                
                # Lock prevents multiple users from hitting RAM at once
                with st.session_state.lock:
                    with st.spinner("AI is reconstructing pixels..."):
                        sr = load_sr_model()
                        if sr:
                            h, w = img.shape[:2]
                            target_h = int(target_width * (h / w))
                            
                            # Prepare image (ignore alpha for AI processing)
                            work_img = img[:,:,:3] if len(img.shape) > 2 and img.shape[2] == 4 else img
                            
                            # Step 1: AI Upscale
                            upscaled = sr.upsample(work_img)
                            
                            # Step 2: Smooth artifacts (Bilateral is faster/cleaner than sharpening)
                            smoothed = cv2.bilateralFilter(upscaled, d=7, sigmaColor=25, sigmaSpace=25)
                            
                            # Step 3: Final scaling to user width
                            result = cv2.resize(smoothed, (target_width, target_h), interpolation=cv2.INTER_AREA)
                            
                            # Step 4: Re-add Alpha channel if it existed
                            if len(img.shape) > 2 and img.shape[2] == 4:
                                alpha = cv2.resize(img[:, :, 3], (target_width, target_h), interpolation=cv2.INTER_AREA)
                                result = cv2.merge([result, alpha])
                            
                            st.session_state.processed_image = result
                            
                            # Calculate time taken
                            duration = time.time() - start_time
                            st.sidebar.success(f"Done! Processed in {duration:.2f} seconds.")

            # --- Layout Columns ---
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Original Image")
                # Fix color for Streamlit display
                disp_orig = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if len(img.shape)>2 and img.shape[2]==4 else cv2.COLOR_BGR2RGB)
                st.image(disp_orig, use_container_width=True)
            
            with c2:
                st.subheader("Upscaled Image")
                if st.session_state.processed_image is not None:
                    res = st.session_state.processed_image
                    disp_res = cv2.cvtColor(res, cv2.COLOR_BGRA2RGBA if len(res.shape)>2 and res.shape[2]==4 else cv2.COLOR_BGR2RGB)
                    st.image(disp_res, use_container_width=True)
                    
                    st.download_button("Download PNG", to_bytes(res, 'PNG'), "ClarityCore.png", "image/png", use_container_width=True)
                    st.download_button("Download JPG", to_bytes(res, 'JPG'), "claritycore_4k.jpg", "image/jpeg", use_container_width=True)
                    
    except Exception as e:
        st.error(f"A temporary error occurred: {e}")
        st.info("Try refreshing the page or using a smaller image.")
else:
    st.info("Please upload an image in the sidebar to begin upscaling.")