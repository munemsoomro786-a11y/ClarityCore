import streamlit as st
import cv2
import numpy as np
import os
import time
import requests

# Page configuration
st.set_page_config(page_title="ClarityCore AI Upscaler", page_icon="🖼️", layout="wide")

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

@st.cache_resource(show_spinner=False)
def load_sr_model():
    """Load model once and keep it in RAM."""
    model_dir = "models"
    model_path = os.path.join(model_dir, "FSRCNN_x4.pb")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists(model_path):
        url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
    return sr

def fast_process(image, target_w):
    """Optimized enhancement pipeline."""
    sr = load_sr_model()
    h, w = image.shape[:2]
    target_h = int(target_w * (h / w))
    
    has_alpha = image.shape[2] == 4 if len(image.shape) > 2 else False
    img = image[:, :, :3] if has_alpha else image.copy()

    # AI Reconstruction
    if sr:
        upscaled = sr.upsample(img)
    else:
        upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Fast Smoothing (Natural Look)
    result = cv2.bilateralFilter(upscaled, d=7, sigmaColor=25, sigmaSpace=25)
    result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if has_alpha:
        alpha = cv2.resize(image[:, :, 3], (target_w, target_h), interpolation=cv2.INTER_AREA)
        return cv2.merge([result, alpha])
    return result

def to_bytes(image, fmt):
    if fmt == 'PNG':
        _, b = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    else:
        temp = image[:, :, :3] if image.shape[2] == 4 else image
        _, b = cv2.imencode('.jpg', temp, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return b.tobytes()

# --- UI ---
st.title("🚀 ClarityCore AI Upscaler")
st.markdown("Upscale Your Images with AI Powered ClarityCore Upscaler.")

file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if file:
    img_arr = np.asarray(bytearray(file.read()), dtype=np.uint8)
    # This variable is now safely defined only after an upload
    original_image = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
    
    tw = st.sidebar.slider("Target Width", 800, 3840, 1920)
    
    if st.sidebar.button("⚡ Upscale", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            st.session_state.processed_image = fast_process(original_image, tw)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGRA2RGBA if len(original_image.shape)>2 and original_image.shape[2]==4 else cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("Upscaled Image")
        if st.session_state.processed_image is not None:
            res = st.session_state.processed_image
            st.image(cv2.cvtColor(res, cv2.COLOR_BGRA2RGBA if len(res.shape)>2 and res.shape[2]==4 else cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.download_button("Download PNG", to_bytes(res, 'PNG'), "clarity_4k.png", "image/png", use_container_width=True)
            st.download_button("Download JPG", to_bytes(res, 'JPG'), "clarity_4k.jpg", "image/jpeg", use_container_width=True)
else:
    st.info("Upload an image in the sidebar to get started!")