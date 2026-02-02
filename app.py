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
    
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Check if model exists, if not, download it
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
        # If loading fails, it might be corrupted. Try one re-download.
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

def upscale_with_tiling(model, image, tile_size=400, padding=10):
    """
    Upscale image using tiling to avoid memory errors.
    """
    height, width = image.shape[:2]
    scale = 4
    
    output_height = height * scale
    output_width = width * scale
    output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y_start = max(0, y - padding)
            y_end = min(height, y + tile_size + padding)
            x_start = max(0, x - padding)
            x_end = min(width, x + tile_size + padding)
            
            tile = image[y_start:y_end, x_start:x_end]
            upscaled_tile = model.upsample(tile)
            
            out_y_start = (y_start * scale)
            in_y_offset_start = (y - y_start) * scale
            in_x_offset_start = (x - x_start) * scale
            
            target_y = y * scale
            target_x = x * scale
            
            valid_h = min((y_end - y_start) * scale - in_y_offset_start, output_height - target_y)
            if y + tile_size < height:
                 valid_h = min(tile_size * scale, output_height - target_y)
            else:
                 valid_h = (y_end - y_start) * scale - in_y_offset_start

            valid_w = min((x_end - x_start) * scale - in_x_offset_start, output_width - target_x)
            if x + tile_size < width:
                valid_w = min(tile_size * scale, output_width - target_x)
            else:
                 valid_w = (x_end - x_start) * scale - in_x_offset_start

            tile_crop = upscaled_tile[
                in_y_offset_start : in_y_offset_start + valid_h,
                in_x_offset_start : in_x_offset_start + valid_w
            ]
            
            output_image[
                target_y : target_y + valid_h,
                target_x : target_x + valid_w
            ] = tile_crop
            
    return output_image

def upscale_image(image, target_width, model_choice):
    """
    Upscale image to target width using super-resolution and interpolation.
    """
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)
    
    scale_factor = target_width / original_width
    
    st.info(f"Original size: {original_width}x{original_height}")
    st.info(f"Target size: {target_width}x{target_height}")
    st.info(f"Scale factor: {scale_factor:.2f}x")
    
    current_image = image.copy()
    
    if abs(scale_factor - 1.0) < 0.01:
        return current_image
    
    # Load super-resolution model
    sr_model, error = load_super_resolution_model(model_choice)
    
    if sr_model is None:
        st.warning(f"Super-resolution model not available: {error}")
        st.info("Using bicubic interpolation instead...")
        result = cv2.resize(current_image, (target_width, target_height), 
                          interpolation=cv2.INTER_CUBIC)
        return result
    
    # Use super-resolution for scaling
    try:
        # Optimization: Always perform a single pass of 4x Super Resolution.
        # Any further scaling (up or down) is handled by high-quality bicubic interpolation.
        # This avoids the exponential slowdown of multi-pass SR (e.g. processing the already upscaled 16x image).
        
        h, w = current_image.shape[:2]
        # Use tiling for larger images to prevent memory issues
        if h * w > 200000:
            with st.spinner(f"Applying super-resolution (4x) with tiling using {model_choice}..."):
                upscaled = upscale_with_tiling(sr_model, current_image)
        else:
            with st.spinner(f"Applying super-resolution (4x) using {model_choice}..."):
                upscaled = sr_model.upsample(current_image)
        
        # Resize to exact target dimensions (whether downscaling or upscaling further)
        with st.spinner("Final resizing to target resolution..."):
            result = cv2.resize(upscaled, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
                                  
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.info("Falling back to standard resizing due to error.")
        result = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    return result

def image_to_bytes(image, format='PNG'):
    """
    Convert OpenCV image to bytes buffer.
    
    Args:
        image: OpenCV image (BGR format)
        format: 'PNG' or 'JPG'
        
    Returns:
        Bytes buffer
    """
    if format.upper() == 'PNG':
        success, encoded = cv2.imencode('.png', image)
    else:  # JPG
        success, encoded = cv2.imencode('.jpg', image, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    if success:
        return encoded.tobytes()
    return None

# Main UI
st.title("🖼️ Image Upscaler")
st.markdown("Upload an image and upscale it to your desired width using AI super-resolution")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to upscale"
    )
    
    if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8) ✅ FIXED
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    # Every line that should only happen IF a file is uploaded must be aligned here

# Check if the image has transparency (4 channels)
if img.shape[2] == 4:
    st.write("✅ Transparency detected and preserved.")
        
        # Display original dimensions
        height, width = original_image.shape[:2]
        st.success(f"✓ Image loaded: {width}x{height}")
        
        # Target width input
        st.subheader("Target Width")
        target_width = st.number_input(
            "Width (pixels)",
            min_value=1,
            max_value=3840,
            value=min(width * 2, 3840),
            step=100,
            help="Maximum width is 3840 pixels (4K resolution)"
        )
        
        # Calculate target height
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        st.info(f"Target size: {target_width}x{target_height}")
        
        st.markdown("---")
        st.subheader("Model Selection")
        model_choice = st.selectbox(
            "Upscaling Model",
            ["FSRCNN (Fast)", "EDSR (High Quality)"],
            index=0,
            help="FSRCNN is much faster but slightly lower quality. EDSR provides best quality but is slower."
        )
        
        # Process button
        if st.button("🚀 Upscale Image", type="primary", use_container_width=True):
            start_time = time.time()
            
            with st.spinner("Processing..."):
                processed_image = upscale_image(original_image, target_width, model_choice)
                st.session_state.processed_image = processed_image
            
            processing_time = time.time() - start_time
            st.success(f"✓ Completed in {processing_time:.2f} seconds")
    else:
        st.info("👆 Upload an image to get started")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    if st.session_state.original_image is not None:
        # Convert BGR to RGB for display
        rgb_image = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
        st.image(rgb_image, use_container_width=True)
    else:
        st.info("No image uploaded yet")

with col2:
    st.subheader("Upscaled Image")
    if st.session_state.processed_image is not None:
        # Convert BGR to RGB for display
        rgb_processed = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
        st.image(rgb_processed, use_container_width=True)
        
        # Download buttons
        st.markdown("### 📥 Download")
        
        col_jpg, col_png = st.columns(2)
        
        with col_jpg:
            jpg_bytes = image_to_bytes(st.session_state.processed_image, 'JPG')
            if jpg_bytes:
                st.download_button(
                    label="Download JPG",
                    data=jpg_bytes,
                    file_name="upscaled_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        
        with col_png:
            png_bytes = image_to_bytes(st.session_state.processed_image, 'PNG')
            if png_bytes:
                st.download_button(
                    label="Download PNG",
                    data=png_bytes,
                    file_name="upscaled_image.png",
                    mime="image/png",
                    use_container_width=True
                )
    else:
        st.info("Upscaled image will appear here after processing")

# Footer with instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload an image** using the sidebar file uploader
    2. **Set target width** (maximum 3840 pixels for 4K resolution)
    3. **Click 'Upscale Image'** to process
    4. **Download** the result in JPG or PNG format
    
    **Note:** For best results, make sure to download the EDSR_x4.pb model file 
    and place it in the `models` folder. See README.md for instructions.
    """)
