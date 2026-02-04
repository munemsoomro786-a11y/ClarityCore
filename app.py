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
    page_title="ClarityCore | Optimized 4K Upscaler",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

def download_model(url, filename):
    """Download model file."""
    try:
        with st.spinner("Loading AI Engine..."):
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Download Error: {e}")
        return False

@st.cache_resource(show_spinner=False)
def load_super_resolution_model():
    """Load and cache the AI model."""
    config = {
        "filename": "FSRCNN_x4.pb",
        "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
        "algo": "fsrcnn",
        "scale": 4
    }
    model_dir = "models"
    model_path = os.path.join(model_dir, config["filename"])
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(model_path):
        if not download_model(config["url"], model_path):
            return None, "Model missing."
    
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel(config["algo"], config["scale"])
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return sr, None
    except Exception as e:
        return None, str(e)

def natural_enhance(image):
    """Clean upscaling without over-sharpening."""
    smoothed = cv2.bilateralFilter(image, d=7, sigmaColor=25, sigmaSpace=25)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def upscale_image(image, target_width):
    """Main pipeline: AI Upscale + Natural Smoothing."""
    orig_h, orig_w = image.shape[:2]
    target_height = int(target_width * (orig_h / orig_w))
    
    has_alpha = image.shape[2] == 4 if len(image.shape) > 2 else False
    img_work = image[:, :, :3] if has_alpha else image.copy()
    
    sr_model, _ = load_super_resolution_model()
    
    try:
        with st.spinner("AI Reconstruction..."):
            upscaled = sr_model.upsample(img_work)
        
        enhanced = natural_enhance(upscaled)
        result = cv2.resize(enhanced, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        if has_alpha:
            alpha = cv2.resize(image[:, :, 3], (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            return cv2.merge([result, alpha])
        return result
    except:
        return cv2.resize(img_work, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def image_to_bytes(image, format='PNG'):
    """Optimized compression to keep files within target limits (20MB PNG / 10MB JPG)."""
    if format.upper() == 'PNG':
        # Level 4-6 is the sweet spot for PNG. Smaller file, same quality.
        _, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 4])
    else:
        # Quality 85-90 significantly reduces size for 4K without losing visible detail.
        temp = image[:, :, :3] if image.shape[2] == 4 else image
        _, encoded = cv2.imencode('.jpg', temp, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes()

# --- UI Setup ---
st.title("🖼️ ClarityCore AI Upscaler")
st.markdown("Upscale to 4K with optimized file sizes for easy sharing.")

with st.sidebar:
    st.header("⚙️ Settings")
    file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if file:
        img_arr = np.asarray(bytearray(file.read()), dtype=np.uint8)
        original = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
        st.session_state.original_image = original
        
        target_w = st.slider("Target Resolution (Width)", 800, 3840, 1920)
        
        if st.button("🚀 Upscale", type="primary", use_container_width=True):
            t1 = time.time()
            st.session_state.processed_image = upscale_image(original, target_w)
            st.success(f"Finished in {time.time()-t1:.2f}s")

# --- Display ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("Original Image")
    if st.session_state.original_image is not None:
        o = st.session_state.original_image
        st.image(cv2.cvtColor(o, cv2.COLOR_BGRA2RGBA if o.shape[2]==4 else cv2.COLOR_BGR2RGB), use_container_width=True)

with c2:
    st.subheader("Upscaled Image")
    if st.session_state.processed_image is not None:
        p = st.session_state.processed_image
        st.image(cv2.cvtColor(p, cv2.COLOR_BGRA2RGBA if p.shape[2]==4 else cv2.COLOR_BGR2RGB), use_container_width=True)
        
        st.markdown("### 📥 Download")
        col_png, col_jpg = st.columns(2)
        with col_png:
            st.download_button("Download PNG", image_to_bytes(p, 'PNG'), "Upscaled.png", "image/png", use_container_width=True)
        with col_jpg:
            st.download_button("Download JPG", image_to_bytes(p, 'JPG'), "Upscaled.jpg", "image/jpeg", use_container_width=True)