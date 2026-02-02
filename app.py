import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
import requests

# Page configuration
st.set_page_config(
    page_title="Image Upscaler",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

def download_model(url, filename):
    """Download model file with progress tracking."""
    try:
        with st.spinner(f"Downloading model (approx 37MB)... this may take a minute"):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False

def load_super_resolution_model(model_choice):
    """Load the super-resolution model, downloading it if necessary."""
    models_config = {
        "FSRCNN (Fast)": {
            "filename": "FSRCNN_x4.pb",
            "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
            "algo": "fsrcnn",
            "scale": 4
        },
        "EDSR (High Quality)": {
            "filename": "EDSR_x4.pb",
            "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
            "algo": "edsr",
            "scale": 4
        }
    }
    
    config = models_config[model_choice]
    model_dir = "models"
    model_path = os.path.join(model_dir, config["filename"])
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(model_path):
        st.info(f"Downloading {config['algo'].upper()} model...")
        if not download_model(config["url"], model_path):
            return None, "Failed to download model."
        st.success(f"{config['algo'].upper()} model downloaded successfully!")
    
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel(config["algo"], config["scale"])
        return sr, None
    except Exception as e:
        st.warning(f"Model file seems corrupted ({str(e)}). Attempting to re-download...")
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
            if download_model(config["url"], model_path):
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(model_path)
                sr.setModel(config["algo"], config["scale"])
                return sr, None
        except Exception as retry_e:
            return None, f"Error loading model after retry: {str(retry_e)}"
        return None, f"Error loading model: {str(e)}"

def upscale_image(image, target_width, model_choice):
    """Upscale image with transparency support."""
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)
    
    # Handle transparency
    has_alpha = image.shape[2] == 4
    if has_alpha:
        # Separate channels: BGR and Alpha
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        bgr = image

    sr_model, error = load_super_resolution_model(model_choice)
    
    if sr_model is None:
        result_bgr = cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        if has_alpha:
            result_alpha = cv2.resize(alpha, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            return cv2.merge([result_bgr, result_alpha])
        return result_bgr

    try:
        # Upscale BGR
        with st.spinner(f"Upscaling RGB layers..."):
            upscaled_bgr = sr_model.upsample(bgr)
            result_bgr = cv2.resize(upscaled_bgr, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        if has_alpha:
            # Upscale Alpha separately to maintain transparency
            with st.spinner(f"Upscaling transparency layer..."):
                result_alpha = cv2.resize(alpha, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
                return cv2.merge([result_bgr, result_alpha])
        return result_bgr
                                  
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

def image_to_bytes(image, format='PNG'):
    if format.upper() == 'PNG':
        success, encoded = cv2.imencode('.png', image)
    else:
        # Convert to BGR if it has alpha for JPG
        temp_img = image[:, :, :3] if image.shape[2] == 4 else image
        success, encoded = cv2.imencode('.jpg', temp_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return encoded.tobytes() if success else None

# Main UI
st.title("🖼️ Image Upscaler")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # FIXED: Use IMREAD_UNCHANGED to keep transparent background
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        st.session_state.original_image = original_image
        
        height, width = original_image.shape[:2]
        st.success(f"✓ Loaded: {width}x{height}")
        
        target_width = st.number_input("Width (pixels)", min_value=1, max_value=3840, value=min(width * 2, 3840))
        model_choice = st.selectbox("Model", ["FSRCNN (Fast)", "EDSR (High Quality)"])
        
        if st.button("🚀 Upscale Image", type="primary", use_container_width=True):
            st.session_state.processed_image = upscale_image(original_image, target_width, model_choice)
            st.success("✓ Done!")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original")
    if st.session_state.original_image is not None:
        # Convert for display while keeping transparency if present
        img = st.session_state.original_image
        mode = cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB
        st.image(cv2.cvtColor(img, mode), use_container_width=True)

with col2:
    st.subheader("Upscaled")
    if st.session_state.processed_image is not None:
        img_p = st.session_state.processed_image
        mode_p = cv2.COLOR_BGRA2RGBA if img_p.shape[2] == 4 else cv2.COLOR_BGR2RGB
        st.image(cv2.cvtColor(img_p, mode_p), use_container_width=True)
        
        col_jpg, col_png = st.columns(2)
        with col_jpg:
            st.download_button("Download JPG", image_to_bytes(img_p, 'JPG'), "upscaled.jpg", "image/jpeg")
        with col_png:
            st.download_button("Download PNG (Transparency)", image_to_bytes(img_p, 'PNG'), "upscaled.png", "image/png")
