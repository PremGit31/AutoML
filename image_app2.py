"""
Image Processing & Computer Vision Dashboard — image_app.py
============================================================
Unstructured Visual Data Exploration, Preprocessing, Feature Extraction,
Pattern Recognition, and Machine Learning.

Tabs:
  1. Visual Ingestion & Dataset Manager — Upload single/batch images or Zip archives,
                                          image metadata, RGB intensity histograms & pixel value heatmap matrix.
  2. Image Preprocessing Pipeline      — Resizing, color space conversions, spatial filtering,
                                          contrast enhancement (CLAHE, Equalization), thresholding
                                          (Otsu, Adaptive), and morphological operations.
  3. Feature Extraction Engine         — Edge detection (Canny, Sobel), keypoints (Harris, FAST, ORB),
                                          GLCM texture analysis, color channel statistics, Hu moments,
                                          HOG descriptors, and batch feature DataFrame exporter.
  4. Pattern Recognition & Vision ML   — Color quantization (K-Means palette), contour finding & object
                                          metrics (Area, Perimeter, Circularity), Watershed segmentation,
                                          PCA 2D/3D feature clustering, and Vision ML Classifiers.
  5. Structured Insights & Export      — Visual executive summary dashboard, processed images download,
                                          structured feature matrices (CSV/JSON), and HTML reports.
"""

import warnings
warnings.filterwarnings("ignore")

import io
import os
import zipfile
import base64
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageEnhance, ImageOps
import cv2
from scipy import stats, ndimage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ──────────────────────────────────────────────────────────────
# Page Config & Custom Styling (Matches home.py, app12.py, nlp_app2.py)
# ──────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="Image Processing & Computer Vision",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            color: #1A1A2E;
        }
        .stApp {
            background-color: #FAF6EF;
        }
        #MainMenu, footer { visibility: hidden; }
        .stAppDeployButton, [data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stToolbarActions"] {
            display: none !important;
            visibility: hidden !important;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #ECE6D8;
            box-shadow: 2px 0 12px rgba(0,0,0,0.03);
        }
        .dashboard-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.2rem;
            font-weight: 800;
            color: #1A1A1A;
            margin-bottom: 0.2rem;
        }
        .dashboard-subtitle {
            font-size: 0.95rem;
            color: #6B6B6B;
            margin-bottom: 1.2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #FFFFFF;
            padding: 6px;
            border-radius: 12px;
            border: 1px solid #ECE6D8;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #6B6B6B;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            padding: 8px 16px;
            transition: all 0.15s ease;
        }
        .stTabs [aria-selected="true"] {
            background: #1A1A2E !important;
            color: #FFFFFF !important;
        }
        .active-banner {
            background: linear-gradient(135deg, #FFFFFF, #F5F1E8);
            border: 1px solid #ECE6D8;
            border-left: 4px solid #1A1A2E;
            border-radius: 8px;
            padding: 12px 18px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            color: #374151;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        .stat-card {
            background: #FFFFFF;
            border: 1px solid #ECE6D8;
            border-radius: 10px;
            padding: 10px 14px;
            margin: 4px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            font-size: 13px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .stat-accent { border-left: 4px solid #9C3E77; }
        .stat-blue   { border-left: 4px solid #3B5B8C; }
        .stat-green  { border-left: 4px solid #10B981; }
        .stat-amber  { border-left: 4px solid #F59E0B; }
        
        .stButton > button {
            background: #1A1A2E;
            color: #FAF6EF;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 20px;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: #3A3A3A;
            color: #FFFFFF;
            transform: translateY(-1px);
        }
        .color-swatch {
            display: inline-block;
            width: 32px;
            height: 32px;
            border-radius: 6px;
            border: 1px solid #DDD;
            vertical-align: middle;
            margin-right: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Helper Functions: Image Presets & Operations
# ──────────────────────────────────────────────────────────────
def generate_sample_image(sample_name):
    """Generate synthetic high-quality test images."""
    if sample_name == "Geometric Shapes & Contours":
        img = np.zeros((400, 400, 3), dtype=np.uint8) + 240
        # Draw colored shapes
        cv2.circle(img, (100, 100), 50, (220, 50, 50), -1)      # Red circle
        cv2.rectangle(img, (220, 50), (350, 150), (50, 180, 50), -1) # Green rect
        # Triangle
        pts = np.array([[80, 350], [180, 250], [250, 350]], np.int32)
        cv2.fillPoly(img, [pts], (50, 80, 220))                  # Blue triangle
        # Ring
        cv2.circle(img, (300, 300), 45, (200, 50, 200), 12)     # Purple ring
        # Noise overlay
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    elif sample_name == "Cell Microscopy Simulator":
        img = np.full((400, 400, 3), (230, 220, 210), dtype=np.uint8)
        np.random.seed(42)
        # Generate cell nuclei
        for _ in range(12):
            cx, cy = np.random.randint(50, 350, 2)
            rad = np.random.randint(15, 35)
            color = (int(np.random.randint(40, 90)), int(np.random.randint(20, 60)), int(np.random.randint(100, 170)))
            cv2.circle(img, (cx, cy), rad, color, -1)
            # Inner core
            cv2.circle(img, (cx, cy), int(rad*0.5), (color[0]//2, color[1]//2, color[2]//2), -1)
        # Gaussian blur background
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return img

    elif sample_name == "Grayscale Texture Matrix":
        x = np.linspace(0, 8 * np.pi, 400)
        y = np.linspace(0, 8 * np.pi, 400)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y) * 127 + 128
        img_gray = Z.astype(np.uint8)
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    else: # Color Gradient Spectrum
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        for i in range(400):
            for j in range(400):
                img[i, j] = [int(i / 400 * 255), int(j / 400 * 255), int((i + j) / 800 * 255)]
        return img


def calc_glcm_features(gray_img):
    """Compute GLCM texture properties across 0, 45, 90, 135 degrees."""
    scaled = (gray_img / 16).astype(np.uint8)
    glcm = np.zeros((16, 16), dtype=np.float64)
    h, w = scaled.shape
    for r in range(h):
        for c in range(w - 1):
            i_val, j_val = scaled[r, c], scaled[r, c + 1]
            glcm[i_val, j_val] += 1
            glcm[j_val, i_val] += 1
    sum_g = np.sum(glcm)
    if sum_g > 0:
        glcm /= sum_g

    contrast = 0.0
    dissimilarity = 0.0
    homogeneity = 0.0
    energy = 0.0
    mean_i = 0.0
    for i in range(16):
        for j in range(16):
            p = glcm[i, j]
            contrast += p * ((i - j) ** 2)
            dissimilarity += p * abs(i - j)
            homogeneity += p / (1.0 + (i - j) ** 2)
            energy += p ** 2
            mean_i += i * p

    correlation = 0.0
    std_i = np.sqrt(np.sum([((i - mean_i) ** 2) * np.sum(glcm[i, :]) for i in range(16)]))
    if std_i > 0:
        for i in range(16):
            for j in range(16):
                correlation += glcm[i, j] * (i - mean_i) * (j - mean_i) / (std_i ** 2)

    return {
        "GLCM_Contrast": round(float(contrast), 4),
        "GLCM_Dissimilarity": round(float(dissimilarity), 4),
        "GLCM_Homogeneity": round(float(homogeneity), 4),
        "GLCM_Energy_ASM": round(float(energy), 4),
        "GLCM_Correlation": round(float(correlation), 4),
    }


def extract_hog_visualization(gray_img):
    """Compute simplified Histogram of Oriented Gradients (HOG)."""
    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    h, w = gray_img.shape
    cell_size = 16
    hog_vis = np.zeros((h, w), dtype=np.uint8)
    
    for r in range(0, h - cell_size, cell_size):
        for c in range(0, w - cell_size, cell_size):
            cell_mag = mag[r:r+cell_size, c:c+cell_size]
            cell_ang = angle[r:r+cell_size, c:c+cell_size]
            avg_ang = np.mean(cell_ang)
            avg_mag = np.mean(cell_mag)
            
            if avg_mag > 15:
                rad = np.radians(avg_ang)
                dx = int(np.cos(rad) * (cell_size / 2))
                dy = int(np.sin(rad) * (cell_size / 2))
                cx, cy = c + cell_size // 2, r + cell_size // 2
                cv2.line(hog_vis, (cx - dx, cy - dy), (cx + dx, cy + dy), 255, 1)

    hist_bins, _ = np.histogram(angle, bins=9, range=(0, 360), weights=mag)
    return hog_vis, hist_bins


def extract_full_image_features(rgb_img, filename="image"):
    """Extract a comprehensive structured feature vector from an RGB image."""
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY) if len(rgb_img.shape) == 3 else rgb_img
    
    if len(rgb_img.shape) == 3:
        r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
    else:
        r, g, b = gray, gray, gray

    features = {
        "Filename": filename,
        "Width": w,
        "Height": h,
        "Aspect_Ratio": round(w / h, 3),
        "Red_Mean": round(float(np.mean(r)), 2),
        "Red_Std": round(float(np.std(r)), 2),
        "Green_Mean": round(float(np.mean(g)), 2),
        "Green_Std": round(float(np.std(g)), 2),
        "Blue_Mean": round(float(np.mean(b)), 2),
        "Blue_Std": round(float(np.std(b)), 2),
        "Gray_Mean": round(float(np.mean(gray)), 2),
        "Gray_Std": round(float(np.std(gray)), 2),
        "Gray_Skewness": round(float(stats.skew(gray.ravel())), 3),
    }

    glcm_feats = calc_glcm_features(gray)
    features.update(glcm_feats)

    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        val = -1.0 * np.copysign(1.0, hu[i]) * np.log10(abs(hu[i])) if hu[i] != 0 else 0.0
        features[f"Hu_Moment_{i+1}"] = round(float(val), 4)

    edges = cv2.Canny(gray, 100, 200)
    features["Edge_Pixel_Ratio"] = round(float(np.count_nonzero(edges) / (h * w)), 4)

    return features


# ──────────────────────────────────────────────────────────────
# Helper: Convert NumPy image array to PNG bytes for download
# ──────────────────────────────────────────────────────────────
def img_to_png_bytes(img_arr):
    """Encode a NumPy image array (RGB, Gray, or any 2-D/3-D uint8) to PNG bytes."""
    if img_arr is None:
        return None
    arr = img_arr.copy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if len(arr.shape) == 2:
        encode_arr = arr
    elif arr.shape[2] == 3:
        encode_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        encode_arr = arr
    ok, buf = cv2.imencode(".png", encode_arr)
    return buf.tobytes() if ok else None


# ──────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────
if "raw_images" not in st.session_state:
    st.session_state.raw_images = {
        "Geometric_Shapes.png": generate_sample_image("Geometric Shapes & Contours"),
        "Microscopy_Cells.png": generate_sample_image("Cell Microscopy Simulator"),
        "Grayscale_Texture.png": generate_sample_image("Grayscale Texture Matrix"),
    }

if "active_image_name" not in st.session_state:
    st.session_state.active_image_name = "Geometric_Shapes.png"

if "select_key_id" not in st.session_state:
    st.session_state.select_key_id = 0

if "pipeline_history" not in st.session_state:
    st.session_state.pipeline_history = {}

if "processed_images" not in st.session_state:
    st.session_state.processed_images = dict(st.session_state.raw_images)

if "processed_file_ids" not in st.session_state:
    st.session_state.processed_file_ids = set()

for k in st.session_state.raw_images:
    if k not in st.session_state.pipeline_history:
        st.session_state.pipeline_history[k] = []


def process_uploaded_files(file_list):
    """Ingest uploaded image files or ZIP archives into session state once."""
    if not file_list:
        return
    if "processed_file_ids" not in st.session_state:
        st.session_state.processed_file_ids = set()

    newly_added = []
    for uf in file_list:
        file_id = f"{uf.name}_{uf.size}"
        if file_id in st.session_state.processed_file_ids:
            continue
        st.session_state.processed_file_ids.add(file_id)

        if uf.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(uf) as z:
                    for zname in z.namelist():
                        if zname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")):
                            bname = os.path.basename(zname)
                            if bname:
                                img_bytes = z.read(zname)
                                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                arr = np.array(pil_img)
                                st.session_state.raw_images[bname] = arr
                                st.session_state.processed_images[bname] = arr.copy()
                                st.session_state.pipeline_history[bname] = []
                                newly_added.append(bname)
            except Exception as e:
                st.error(f"Error extracting ZIP file {uf.name}: {e}")
        else:
            try:
                pil_img = Image.open(uf).convert("RGB")
                arr = np.array(pil_img)
                st.session_state.raw_images[uf.name] = arr
                st.session_state.processed_images[uf.name] = arr.copy()
                st.session_state.pipeline_history[uf.name] = []
                newly_added.append(uf.name)
            except Exception as e:
                st.error(f"Error loading image {uf.name}: {e}")

    if newly_added:
        last_name = newly_added[-1]
        st.session_state.active_image_name = last_name
        st.session_state.select_key_id += 1
        st.success(f"Successfully ingested {len(newly_added)} custom image(s)!")
        st.rerun()


# ──────────────────────────────────────────────────────────────
# Header & Navigation
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="dashboard-title">🖼️ Image Processing & Computer Vision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Transform raw visual data into structured insights — covering preprocessing, edge/texture feature extraction, contour analysis, and pattern recognition.</div>',
    unsafe_allow_html=True,
)

img_names = list(st.session_state.raw_images.keys())
if st.session_state.active_image_name not in img_names and len(img_names) > 0:
    st.session_state.active_image_name = img_names[0]

active_name = st.session_state.active_image_name
curr_img = st.session_state.processed_images.get(active_name, st.session_state.raw_images.get(active_name))
history_steps = st.session_state.pipeline_history.get(active_name, [])

banner_steps = " -> ".join([f"[{i+1}] {step}" for i, step in enumerate(history_steps)]) if history_steps else "None (Original Image)"
st.markdown(
    f"""
    <div class="active-banner">
        <strong>Active Image:</strong> {active_name} &nbsp;|&nbsp;
        <strong>Dimensions:</strong> {curr_img.shape[1]}x{curr_img.shape[0]} px &nbsp;|&nbsp;
        <strong>Channels:</strong> {curr_img.shape[2] if len(curr_img.shape)==3 else 1} &nbsp;|&nbsp;
        <strong>Applied Pipeline Steps:</strong> {banner_steps}
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Dataset & Active Selection")
    
    img_names = list(st.session_state.raw_images.keys())
    active_idx = img_names.index(active_name) if active_name in img_names else 0
    selected_name = st.selectbox(
        "Select Active Image",
        options=img_names,
        index=active_idx,
        key=f"sb_select_{st.session_state.select_key_id}",
    )
    if selected_name != st.session_state.active_image_name:
        st.session_state.active_image_name = selected_name
        st.rerun()

    st.markdown("---")
    st.markdown("### 📁 Rapid Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload Image(s) / ZIP",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "zip"],
        accept_multiple_files=True,
        key="sidebar_uploader",
    )
    if uploaded_files:
        process_uploaded_files(uploaded_files)



# ──────────────────────────────────────────────────────────────
# Main Application Tabs
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖼️ Visual Ingestion & Gallery",
    "⚙️ Image Preprocessing",
    "🔬 Feature Extraction Engine",
    "🎯 Pattern Recognition & Vision ML",
    "📊 Structured Insights & Export",
])

# ==============================================================================
# TAB 1: VISUAL INGESTION & GALLERY
# ==============================================================================
with tab1:
    st.markdown("### 🖼️ Visual Dataset Gallery & Matrix Inspector")
    
    with st.expander("📤 **Upload Custom Image(s) / ZIP Dataset** (Supports JPG, JPEG, PNG, WEBP, BMP, TIFF, ZIP)", expanded=False):
        t1_files = st.file_uploader(
            "Select or Drag & Drop Custom Images:",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "zip"],
            accept_multiple_files=True,
            key="t1_uploader"
        )
        if t1_files:
            process_uploaded_files(t1_files)

    col_gal, col_insp = st.columns([1, 1], gap="large")
    
    with col_gal:
        st.markdown("#### Image Gallery")
        gallery_cols = st.columns(2)
        for idx, (iname, img_arr) in enumerate(st.session_state.processed_images.items()):
            with gallery_cols[idx % 2]:
                st.image(img_arr, caption=iname, use_container_width=True)
                if st.button(f"Select", key=f"btn_sel_{iname}", use_container_width=True):
                    st.session_state.active_image_name = iname
                    st.session_state.select_key_id += 1
                    st.rerun()

    with col_insp:
        st.markdown(f"#### Active Image Inspector: `{active_name}`")
        st.image(curr_img, caption="Active Image Preview", use_container_width=True)
        
        h, w = curr_img.shape[:2]
        ch = curr_img.shape[2] if len(curr_img.shape) == 3 else 1
        size_kb = round((curr_img.nbytes) / 1024, 2)
        
        sc1, sc2 = st.columns(2)
        sc1.markdown(f'<div class="stat-card stat-blue"><strong>Width:</strong> {w} px</div>', unsafe_allow_html=True)
        sc2.markdown(f'<div class="stat-card stat-accent"><strong>Height:</strong> {h} px</div>', unsafe_allow_html=True)
        
        sc3, sc4 = st.columns(2)
        sc3.markdown(f'<div class="stat-card stat-green"><strong>Channels:</strong> {ch}</div>', unsafe_allow_html=True)
        sc4.markdown(f'<div class="stat-card stat-amber"><strong>Memory:</strong> {size_kb} KB</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Intensity Distribution & Pixel Grid Matrix Inspector")
    
    col_hist, col_pixel = st.columns([1, 1], gap="medium")
    
    with col_hist:
        st.markdown("#### RGB Channel Intensity Histogram")
        use_log = st.checkbox("Logarithmic Intensity Scale (enhances dark/bright wallpapers)", value=True, key=f"log_hist_{active_name}")
        
        fig_hist = go.Figure()
        if len(curr_img.shape) == 3 and curr_img.shape[2] >= 3:
            colors = ['red', 'green', 'blue']
            labels = ['Red Channel', 'Green Channel', 'Blue Channel']
            for i, col in enumerate(colors[:min(3, curr_img.shape[2])]):
                ch_data = curr_img[:, :, i].ravel()
                counts, bins = np.histogram(ch_data, bins=256, range=(0, 256))
                plot_y = np.log1p(counts) if use_log else counts
                fig_hist.add_trace(go.Scatter(x=bins[:-1], y=plot_y, name=labels[i], line=dict(color=col, width=2)))
        else:
            ch_data = curr_img.ravel()
            counts, bins = np.histogram(ch_data, bins=256, range=(0, 256))
            plot_y = np.log1p(counts) if use_log else counts
            fig_hist.add_trace(go.Scatter(x=bins[:-1], y=plot_y, name="Grayscale", line=dict(color="black", width=2)))
            
        fig_hist.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Pixel Intensity (0-255)",
            yaxis_title="Log1p(Pixel Count)" if use_log else "Pixel Count",
            template="plotly_white",
            height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_chart_{active_name}")

    with col_pixel:
        st.markdown("#### 🔍 Pixel Matrix Sub-Region Heatmap")
        st.write("Inspect numerical pixel values of a crop region:")
        
        max_x = max(0, w - 5)
        max_y = max(0, h - 5)
        
        default_x = max(0, min(max_x, (w // 2) - 7))
        default_y = max(0, min(max_y, (h // 2) - 7))
        
        # Use image-specific keys so slider max_val changes cleanly per active image
        cx_start = st.slider("Crop X Start", 0, max_x, default_x, key=f"heatmap_x_{active_name}")
        cy_start = st.slider("Crop Y Start", 0, max_y, default_y, key=f"heatmap_y_{active_name}")
        
        cx_start = min(cx_start, max_x)
        cy_start = min(cy_start, max_y)
        
        crop_region = curr_img[cy_start:min(h, cy_start + 15), cx_start:min(w, cx_start + 15)]
        
        if crop_region.size > 0:
            if len(crop_region.shape) == 3:
                if crop_region.shape[2] >= 4:
                    crop_gray = cv2.cvtColor(crop_region, cv2.COLOR_RGBA2GRAY)
                elif crop_region.shape[2] == 3:
                    crop_gray = cv2.cvtColor(crop_region, cv2.COLOR_RGB2GRAY)
                else:
                    crop_gray = crop_region[:, :, 0]
            else:
                crop_gray = crop_region
                
            fig_heat = px.imshow(
                crop_gray,
                labels=dict(x="X Offset", y="Y Offset", color="Intensity"),
                text_auto=True,
                color_continuous_scale="Viridis",
            )
            fig_heat.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=320)
            st.plotly_chart(fig_heat, use_container_width=True, key=f"heat_chart_{active_name}")
        else:
            st.info("Selected crop region is out of bounds.")


# ==============================================================================
# TAB 2: IMAGE PREPROCESSING PIPELINE
# ==============================================================================
with tab2:
    st.markdown("### ⚙️ Image Preprocessing Pipeline")
    st.write("Apply modular transformations sequentially to clean, enhance, and prepare visual data.")
    
    col_controls, col_preview = st.columns([1, 1.2], gap="large")
    
    with col_controls:
        st.markdown("#### Select & Parameterize Step")
        prep_action = st.selectbox(
            "Transformation Category",
            [
                "Resize & Scale",
                "Color Space Conversion",
                "Spatial Filtering & Noise Reduction",
                "Contrast & Exposure Enhancement",
                "Thresholding & Binarization",
                "Morphological Operations",
            ],
        )
        
        new_img = curr_img.copy()
        step_desc = ""

        if prep_action == "Resize & Scale":
            st.markdown("##### Resizing Parameters")
            scale_pct = st.slider("Scale Percentage", 10, 200, 100)
            interp_opt = st.selectbox("Interpolation", ["Bilinear", "Nearest Neighbor", "Bicubic", "Lanczos"])
            interp_map = {
                "Bilinear": cv2.INTER_LINEAR,
                "Nearest Neighbor": cv2.INTER_NEAREST,
                "Bicubic": cv2.INTER_CUBIC,
                "Lanczos": cv2.INTER_LANCZOS4,
            }
            new_w = max(1, int(w * (scale_pct / 100.0)))
            new_h = max(1, int(h * (scale_pct / 100.0)))
            new_img = cv2.resize(curr_img, (new_w, new_h), interpolation=interp_map[interp_opt])
            step_desc = f"Resized to {new_w}x{new_h} ({scale_pct}%, {interp_opt})"

        elif prep_action == "Color Space Conversion":
            st.markdown("##### Color Space Options")
            c_space = st.selectbox("Convert To", ["Grayscale", "HSV", "LAB", "YCrCb", "RGB Split (Red)", "RGB Split (Green)", "RGB Split (Blue)"])
            if c_space == "Grayscale":
                if len(curr_img.shape) == 3:
                    new_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
            elif c_space == "HSV":
                if len(curr_img.shape) == 3:
                    new_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2HSV)
            elif c_space == "LAB":
                if len(curr_img.shape) == 3:
                    new_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2LAB)
            elif c_space == "YCrCb":
                if len(curr_img.shape) == 3:
                    new_img = cv2.cvtColor(curr_img, cv2.COLOR_RGB2YCrCb)
            elif c_space == "RGB Split (Red)":
                new_img = curr_img[:,:,0] if len(curr_img.shape)==3 else curr_img
            elif c_space == "RGB Split (Green)":
                new_img = curr_img[:,:,1] if len(curr_img.shape)==3 else curr_img
            else:
                new_img = curr_img[:,:,2] if len(curr_img.shape)==3 else curr_img
            step_desc = f"Color Space: {c_space}"

        elif prep_action == "Spatial Filtering & Noise Reduction":
            st.markdown("##### Filter Settings")
            f_type = st.selectbox("Filter Type", ["Gaussian Blur", "Median Blur", "Bilateral Filter", "Box Blur"])
            k_size = st.slider("Kernel Size (Odd)", 3, 21, 5, step=2)
            if f_type == "Gaussian Blur":
                sigma = st.slider("Gaussian Sigma", 0.1, 5.0, 1.0)
                new_img = cv2.GaussianBlur(curr_img, (k_size, k_size), sigma)
            elif f_type == "Median Blur":
                new_img = cv2.medianBlur(curr_img, k_size)
            elif f_type == "Bilateral Filter":
                sig_color = st.slider("Sigma Color", 10, 150, 75)
                new_img = cv2.bilateralFilter(curr_img, k_size, sig_color, sig_color)
            else:
                new_img = cv2.blur(curr_img, (k_size, k_size))
            step_desc = f"{f_type} (Kernel {k_size}x{k_size})"

        elif prep_action == "Contrast & Exposure Enhancement":
            st.markdown("##### Enhancement Settings")
            c_method = st.selectbox("Method", ["Histogram Equalization", "CLAHE (Adaptive)", "Gamma Correction", "Brightness/Contrast Adjustment"])
            
            gray_input = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if len(curr_img.shape) == 3 else curr_img
            
            if c_method == "Histogram Equalization":
                new_img = cv2.equalizeHist(gray_input)
            elif c_method == "CLAHE (Adaptive)":
                clip_lim = st.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0)
                tile_size = st.slider("Grid Tile Size", 2, 16, 8)
                clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(tile_size, tile_size))
                new_img = clahe.apply(gray_input)
            elif c_method == "Gamma Correction":
                gamma = st.slider("Gamma Value", 0.1, 3.0, 1.2)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                new_img = cv2.LUT(curr_img, table)
            else:
                alpha = st.slider("Contrast (Alpha)", 0.5, 3.0, 1.2)
                beta = st.slider("Brightness (Beta)", -100, 100, 10)
                new_img = cv2.convertScaleAbs(curr_img, alpha=alpha, beta=beta)
            step_desc = f"Contrast: {c_method}"

        elif prep_action == "Thresholding & Binarization":
            st.markdown("##### Thresholding Options")
            t_type = st.selectbox("Type", ["Otsu's Thresholding", "Global Manual Threshold", "Adaptive Mean", "Adaptive Gaussian"])
            gray_input = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if len(curr_img.shape) == 3 else curr_img
            inv_flag = st.checkbox("Invert Binary Output", False)
            thresh_mode = cv2.THRESH_BINARY_INV if inv_flag else cv2.THRESH_BINARY

            if t_type == "Otsu's Thresholding":
                _, new_img = cv2.threshold(gray_input, 0, 255, thresh_mode + cv2.THRESH_OTSU)
            elif t_type == "Global Manual Threshold":
                t_val = st.slider("Threshold Value", 0, 255, 128)
                _, new_img = cv2.threshold(gray_input, t_val, 255, thresh_mode)
            elif t_type == "Adaptive Mean":
                blk_size = st.slider("Block Size (Odd)", 3, 31, 11, step=2)
                c_val = st.slider("Constant C", -10, 10, 2)
                new_img = cv2.adaptiveThreshold(gray_input, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_mode, blk_size, c_val)
            else:
                blk_size = st.slider("Block Size (Odd)", 3, 31, 11, step=2)
                c_val = st.slider("Constant C", -10, 10, 2)
                new_img = cv2.adaptiveThreshold(gray_input, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_mode, blk_size, c_val)
            step_desc = f"Binarization: {t_type}"

        else: # Morphological Operations
            st.markdown("##### Morphology Parameters")
            m_op = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"])
            m_shape = st.selectbox("Kernel Shape", ["Rectangle", "Ellipse", "Cross"])
            m_ksize = st.slider("Kernel Size", 1, 15, 3)
            m_iters = st.slider("Iterations", 1, 5, 1)

            shape_map = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
            kernel = cv2.getStructuringElement(shape_map[m_shape], (m_ksize, m_ksize))
            op_map = {
                "Erosion": cv2.MORPH_ERODE, "Dilation": cv2.MORPH_DILATE, "Opening": cv2.MORPH_OPEN,
                "Closing": cv2.MORPH_CLOSE, "Gradient": cv2.MORPH_GRADIENT, "Top Hat": cv2.MORPH_TOPHAT,
                "Black Hat": cv2.MORPH_BLACKHAT,
            }
            new_img = cv2.morphologyEx(curr_img, op_map[m_op], kernel, iterations=m_iters)
            step_desc = f"Morphology: {m_op} ({m_ksize}x{m_ksize})"

        st.markdown("---")
        btn_apply, btn_reset = st.columns(2)
        if btn_apply.button("✅ Apply Step to Active Image", use_container_width=True):
            st.session_state.processed_images[active_name] = new_img
            st.session_state.pipeline_history[active_name].append(step_desc)
            st.success(f"Applied: {step_desc}")
            st.rerun()

        if btn_reset.button("🔄 Reset to Original", use_container_width=True):
            st.session_state.processed_images[active_name] = st.session_state.raw_images[active_name].copy()
            st.session_state.pipeline_history[active_name] = []
            st.info("Reset active image to original state.")
            st.rerun()

    with col_preview:
        st.markdown("#### Step Live Preview (Before vs. After)")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.image(curr_img, caption="Before Step", use_container_width=True)
        with p_c2:
            st.image(new_img, caption="Live Transformed Preview", use_container_width=True)

        # ── Save current preview result ──────────────────────────────
        _prep_bytes = img_to_png_bytes(new_img)
        if _prep_bytes:
            st.download_button(
                label="💾 Save Edited Image (Current Preview)",
                data=_prep_bytes,
                file_name=f"preprocessed_{active_name}",
                mime="image/png",
                use_container_width=True,
                key="dl_prep_preview",
            )

        st.markdown("##### Applied Pipeline Log for Active Image")
        if history_steps:
            for idx, h_step in enumerate(history_steps):
                st.markdown(f"`Step {idx+1}:` {h_step}")
        else:
            st.write("No steps applied yet.")


# ==============================================================================
# TAB 3: FEATURE EXTRACTION ENGINE
# ==============================================================================
with tab3:
    st.markdown("### 🔬 Feature Extraction Engine")
    st.write("Extract structural, edge, keypoint, texture, and statistical features to construct a tabular feature vector.")

    gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if len(curr_img.shape) == 3 else curr_img

    fe_tab1, fe_tab2, fe_tab3, fe_tab4 = st.tabs([
        "⚡ Edges & Descriptors",
        "🎯 Keypoint Detection",
        "🧵 GLCM Texture & Hu Moments",
        "📦 Batch Dataset Extractor",
    ])

    with fe_tab1:
        st.markdown("#### Edge Detection & Gradient Descriptors")
        col_edge_ctrl, col_edge_view = st.columns([1, 1.2])
        
        with col_edge_ctrl:
            edge_algo = st.selectbox("Edge Algorithm", ["Canny Edge Detector", "Sobel Gradient (X+Y)", "Laplacian of Gaussian", "Scharr Filter"])
            if edge_algo == "Canny Edge Detector":
                t1 = st.slider("Canny Threshold 1", 0, 255, 100)
                t2 = st.slider("Canny Threshold 2", 0, 255, 200)
                edge_img = cv2.Canny(gray_curr, t1, t2)
            elif edge_algo == "Sobel Gradient (X+Y)":
                sx = cv2.Sobel(gray_curr, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray_curr, cv2.CV_64F, 0, 1, ksize=3)
                edge_img = cv2.magnitude(sx, sy)
                edge_img = np.uint8(np.clip(edge_img, 0, 255))
            elif edge_algo == "Laplacian of Gaussian":
                blur = cv2.GaussianBlur(gray_curr, (3, 3), 0)
                lap = cv2.Laplacian(blur, cv2.CV_64F)
                edge_img = np.uint8(np.abs(lap))
            else:
                sx = cv2.Scharr(gray_curr, cv2.CV_64F, 1, 0)
                sy = cv2.Scharr(gray_curr, cv2.CV_64F, 0, 1)
                edge_img = np.uint8(np.clip(cv2.magnitude(sx, sy), 0, 255))

        with col_edge_view:
            st.image(edge_img, caption=f"Result: {edge_algo}", use_container_width=True)
            _edge_bytes = img_to_png_bytes(edge_img)
            if _edge_bytes:
                st.download_button(
                    label="💾 Save Edge Detected Image",
                    data=_edge_bytes,
                    file_name=f"edge_{edge_algo.split()[0].lower()}_{active_name}",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_edge_img",
                )

        st.markdown("---")
        st.markdown("#### Histogram of Oriented Gradients (HOG)")
        hog_vis, hog_bins = extract_hog_visualization(gray_curr)
        c_hog1, c_hog2 = st.columns([1, 1])
        with c_hog1:
            st.image(hog_vis, caption="HOG Gradient Overlay", use_container_width=True)
            _hog_bytes = img_to_png_bytes(hog_vis)
            if _hog_bytes:
                st.download_button(
                    label="💾 Save HOG Visualization",
                    data=_hog_bytes,
                    file_name=f"hog_visualization_{active_name}",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_hog_img",
                )
        with c_hog2:
            fig_hog = px.bar(x=np.arange(0, 360, 40), y=hog_bins, labels=dict(x="Orientation Bins (°)", y="Magnitude Sum"))
            fig_hog.update_layout(title="HOG Orientation Histogram", margin=dict(l=10, r=10, t=30, b=10), height=250)
            st.plotly_chart(fig_hog, use_container_width=True)

    with fe_tab2:
        st.markdown("#### Keypoint & Corner Detection")
        c_kp_ctrl, c_kp_view = st.columns([1, 1.2])
        
        with c_kp_ctrl:
            kp_method = st.selectbox("Keypoint Detector", ["Harris Corner Detector", "FAST Keypoints", "ORB Features"])
            img_kp_overlay = curr_img.copy()
            if len(img_kp_overlay.shape) == 2:
                img_kp_overlay = cv2.cvtColor(img_kp_overlay, cv2.COLOR_GRAY2RGB)

            if kp_method == "Harris Corner Detector":
                blockSize = st.slider("Block Size", 2, 7, 2)
                ksize = st.slider("Aperture Parameter", 3, 7, 3, step=2)
                k_val = st.slider("Harris Free Parameter k", 0.01, 0.1, 0.04)
                dst = cv2.cornerHarris(gray_curr, blockSize, ksize, k_val)
                dst = cv2.dilate(dst, None)
                img_kp_overlay[dst > 0.01 * dst.max()] = [255, 0, 0]
                n_kp = np.count_nonzero(dst > 0.01 * dst.max())

            elif kp_method == "FAST Keypoints":
                fast_thresh = st.slider("FAST Threshold", 1, 100, 25)
                fast = cv2.FastFeatureDetector_create(threshold=fast_thresh)
                kp = fast.detect(gray_curr, None)
                img_kp_overlay = cv2.drawKeypoints(img_kp_overlay, kp, None, color=(0, 255, 0))
                n_kp = len(kp)

            else:
                orb = cv2.ORB_create(nfeatures=250)
                kp, des = orb.detectAndCompute(gray_curr, None)
                img_kp_overlay = cv2.drawKeypoints(img_kp_overlay, kp, None, color=(255, 0, 255))
                n_kp = len(kp) if kp else 0

            st.markdown(f'<div class="stat-card stat-green"><strong>Detected Keypoints:</strong> {n_kp}</div>', unsafe_allow_html=True)

        with c_kp_view:
            st.image(img_kp_overlay, caption=f"Keypoints Visualization ({kp_method})", use_container_width=True)
            _kp_bytes = img_to_png_bytes(img_kp_overlay)
            if _kp_bytes:
                st.download_button(
                    label="💾 Save Keypoint Overlay Image",
                    data=_kp_bytes,
                    file_name=f"keypoints_{kp_method.split()[0].lower()}_{active_name}",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_kp_img",
                )

    with fe_tab3:
        st.markdown("#### GLCM Texture Properties & Hu Invariant Moments")
        
        glcm_res = calc_glcm_features(gray_curr)
        c_glcm, c_hu = st.columns(2)
        
        with c_glcm:
            st.markdown("##### 🧵 Gray-Level Co-occurrence Matrix (GLCM)")
            glcm_df = pd.DataFrame(list(glcm_res.items()), columns=["Texture Metric", "Value"])
            st.dataframe(glcm_df, use_container_width=True, hide_index=True)
            _glcm_csv = glcm_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Save GLCM Features (CSV)",
                data=_glcm_csv,
                file_name=f"glcm_features_{active_name}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_glcm_csv",
            )

        with c_hu:
            st.markdown("##### 📐 Hu Invariant Moments (Log Scaled)")
            moments = cv2.moments(gray_curr)
            hu_raw = cv2.HuMoments(moments).flatten()
            hu_dict = {}
            for i in range(7):
                v = -1.0 * np.copysign(1.0, hu_raw[i]) * np.log10(abs(hu_raw[i])) if hu_raw[i] != 0 else 0.0
                hu_dict[f"Hu Moment {i+1}"] = round(float(v), 4)
            
            hu_df = pd.DataFrame(list(hu_dict.items()), columns=["Moment", "Log Value"])
            st.dataframe(hu_df, use_container_width=True, hide_index=True)
            _hu_csv = hu_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Save Hu Moments (CSV)",
                data=_hu_csv,
                file_name=f"hu_moments_{active_name}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_hu_csv",
            )

    with fe_tab4:
        st.markdown("#### 📦 Batch Dataset Extractor (Visual Data → Structured CSV)")
        st.write("Extract tabular features for all images in the session gallery:")
        
        if st.button("🚀 Extract Features for All Session Images"):
            batch_list = []
            for img_name, img_arr in st.session_state.processed_images.items():
                feat = extract_full_image_features(img_arr, filename=img_name)
                batch_list.append(feat)
            
            st.session_state.extracted_df = pd.DataFrame(batch_list)
            st.success(f"Extracted {len(batch_list)} feature vector rows with {st.session_state.extracted_df.shape[1]} metrics!")

        if "extracted_df" in st.session_state:
            st.dataframe(st.session_state.extracted_df, use_container_width=True)
            csv_data = st.session_state.extracted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Extracted Features CSV",
                data=csv_data,
                file_name="image_extracted_features.csv",
                mime="text/csv",
            )


# ==============================================================================
# TAB 4: PATTERN RECOGNITION & VISION ML
# ==============================================================================
with tab4:
    st.markdown("### 🎯 Pattern Recognition & Vision Machine Learning")
    
    pr_tab1, pr_tab2, pr_tab3, pr_tab4 = st.tabs([
        "🎨 Color Quantization (K-Means)",
        "🔍 Object Detection & Contours",
        "🌊 Watershed Segmentation",
        "🤖 Vision ML Classification",
    ])

    with pr_tab1:
        st.markdown("#### Color Palette Extraction via K-Means Clustering")
        c_k1, c_k2 = st.columns([1, 1.2])
        
        with c_k1:
            n_colors = st.slider("Number of Dominant Colors", 2, 8, 4)
            
            pixels = curr_img.reshape(-1, 3) if len(curr_img.shape) == 3 else np.column_stack([curr_img.ravel()]*3)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=5).fit(pixels)
            palette = kmeans.cluster_centers_.astype(np.uint8)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            percentages = counts / len(pixels) * 100
            
            st.markdown("##### Dominant Color Palette:")
            for idx in range(n_colors):
                hex_c = '#{:02x}{:02x}{:02x}'.format(palette[idx][0], palette[idx][1], palette[idx][2])
                st.markdown(
                    f'<span class="color-swatch" style="background-color:{hex_c};"></span> '
                    f'<strong>{hex_c}</strong> — RGB({palette[idx][0]}, {palette[idx][1]}, {palette[idx][2]}) — {percentages[idx]:.1f}%',
                    unsafe_allow_html=True,
                )

        with c_k2:
            target_shape = (curr_img.shape[0], curr_img.shape[1], 3)
            quantized = palette[kmeans.labels_].reshape(target_shape)
            st.image(quantized, caption=f"Quantized Image ({n_colors} Colors)", use_container_width=True)
            _quant_bytes = img_to_png_bytes(quantized)
            if _quant_bytes:
                st.download_button(
                    label="💾 Save Quantized Image",
                    data=_quant_bytes,
                    file_name=f"quantized_{n_colors}colors_{active_name}",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_quant_img",
                )

    with pr_tab2:
        st.markdown("#### Automatic Contour Discovery & Object Geometry Analysis")
        gray_cont = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if len(curr_img.shape) == 3 else curr_img
        
        _, thresh_c = cv2.threshold(gray_cont, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_img = curr_img.copy()
        if len(contour_img.shape) == 2:
            contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2RGB)

        contour_records = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 30:
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
                cv2.rectangle(contour_img, (x, y), (x + w_box, y + h_box), (255, 0, 0), 1)
                cv2.putText(contour_img, f"#{idx+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                contour_records.append({
                    "Object ID": f"#{idx+1}",
                    "Area (px²)": round(area, 1),
                    "Perimeter (px)": round(perimeter, 1),
                    "Circularity": round(circularity, 3),
                    "Bounding Box (X,Y,W,H)": f"({x},{y},{w_box},{h_box})",
                })

        c_cnt_v, c_cnt_t = st.columns([1, 1.1])
        with c_cnt_v:
            st.image(contour_img, caption=f"Discovered Objects (Count: {len(contour_records)})", use_container_width=True)
            _contour_bytes = img_to_png_bytes(contour_img)
            if _contour_bytes:
                st.download_button(
                    label="💾 Save Contour Overlay Image",
                    data=_contour_bytes,
                    file_name=f"contours_{active_name}",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_contour_img",
                )

        with c_cnt_t:
            st.markdown("##### Discovered Object Metrics Table")
            if contour_records:
                _cnt_df = pd.DataFrame(contour_records)
                st.dataframe(_cnt_df, use_container_width=True)
                _cnt_csv = _cnt_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Save Contour Metrics (CSV)",
                    data=_cnt_csv,
                    file_name=f"contour_metrics_{active_name}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_contour_csv",
                )
            else:
                st.write("No distinct contours found. Try thresholding in the Preprocessing tab first!")

    with pr_tab3:
        st.markdown("#### Watershed Image Segmentation Algorithm")
        st.write("Separate touching or overlapping object instances:")
        
        gray_w = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if len(curr_img.shape) == 3 else curr_img
        _, thresh_w = cv2.threshold(gray_w, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh_w, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        ws_img = curr_img.copy()
        if len(ws_img.shape) == 2:
            ws_img = cv2.cvtColor(ws_img, cv2.COLOR_GRAY2RGB)
        ws_img = cv2.watershed(ws_img, markers)
        
        out_ws = curr_img.copy()
        if len(out_ws.shape) == 2:
            out_ws = cv2.cvtColor(out_ws, cv2.COLOR_GRAY2RGB)
        out_ws[ws_img == -1] = [255, 0, 0]

        c_w1, c_w2 = st.columns(2)
        with c_w1:
            dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(dist_norm, caption="Distance Transform Map", use_container_width=True)
            _dist_bytes = img_to_png_bytes(dist_norm)
            if _dist_bytes:
                st.download_button(
                    label="💾 Save Distance Transform Map",
                    data=_dist_bytes,
                    file_name=f"distance_transform_{active_name}",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_dist_img",
                )
        with c_w2:
            st.image(out_ws, caption="Watershed Segmented Boundaries", use_container_width=True)
            _ws_bytes = img_to_png_bytes(out_ws)
            if _ws_bytes:
                st.download_button(
                    label="💾 Save Watershed Segmented Image",
                    data=_ws_bytes,
                    file_name=f"watershed_{active_name}",
                    mime="image/png",
                    use_container_width=True,
                    key="dl_watershed_img",
                )

    with pr_tab4:
        st.markdown("#### 🤖 Supervised Vision Machine Learning Classifier")
        st.write("Train ML models on extracted visual feature vectors:")

        if "extracted_df" not in st.session_state or len(st.session_state.extracted_df) < 3:
            batch_list = []
            for img_name, img_arr in st.session_state.processed_images.items():
                feat = extract_full_image_features(img_arr, filename=img_name)
                feat["Target_Class"] = "Shapes" if "Shape" in img_name else ("Cells" if "Cell" in img_name else "Pattern")
                batch_list.append(feat)
            st.session_state.extracted_df = pd.DataFrame(batch_list)

        df_ml = st.session_state.extracted_df.copy()
        if "Target_Class" not in df_ml.columns:
            df_ml["Target_Class"] = [f"Class_{(i%2)+1}" for i in range(len(df_ml))]

        st.dataframe(df_ml, use_container_width=True)

        col_ml_cfg, col_ml_res = st.columns([1, 1.1])
        with col_ml_cfg:
            model_type = st.selectbox("Vision Classifier", ["Random Forest", "Support Vector Classifier (SVC)", "K-Nearest Neighbors", "Logistic Regression"])
            
            numeric_cols = [c for c in df_ml.columns if c not in ["Filename", "Target_Class"]]
            X = df_ml[numeric_cols].fillna(0)
            y = df_ml["Target_Class"]

            if st.button("🚀 Train Vision Model"):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                if model_type == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=50, random_state=42)
                elif model_type == "Support Vector Classifier (SVC)":
                    clf = SVC(probability=True, random_state=42)
                elif model_type == "K-Nearest Neighbors":
                    clf = KNeighborsClassifier(n_neighbors=max(1, min(3, len(X)-1)))
                else:
                    clf = LogisticRegression()

                clf.fit(X_scaled, y)
                preds = clf.predict(X_scaled)
                acc = accuracy_score(y, preds)

                st.session_state.trained_clf = clf
                st.session_state.trained_scaler = scaler
                st.session_state.ml_acc = acc

        with col_ml_res:
            if "trained_clf" in st.session_state:
                st.markdown(f'<div class="stat-card stat-green"><strong>Model Accuracy:</strong> {st.session_state.ml_acc * 100:.1f}%</div>', unsafe_allow_html=True)
                st.markdown("##### Live Single Image Inference Tester:")
                curr_feats = extract_full_image_features(curr_img, filename=active_name)
                curr_x = pd.DataFrame([curr_feats])[numeric_cols].fillna(0)
                curr_scaled = st.session_state.trained_scaler.transform(curr_x)
                pred_label = st.session_state.trained_clf.predict(curr_scaled)[0]
                
                st.markdown(f"Predicted Class for `{active_name}`: **:blue[{pred_label}]**")

                # Save ML feature vector for current image
                _ml_feats_df = pd.DataFrame([curr_feats])
                _ml_feats_df["Predicted_Class"] = pred_label
                _ml_csv = _ml_feats_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Save ML Feature Vector (CSV)",
                    data=_ml_csv,
                    file_name=f"ml_features_{active_name}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_ml_feats_csv",
                )


# ==============================================================================
# TAB 5: STRUCTURED INSIGHTS & EXPORT STUDIO
# ==============================================================================
with tab5:
    st.markdown("### 📊 Executive Insights & Export Studio")
    
    e_col1, e_col2 = st.columns([1, 1])
    
    with e_col1:
        st.markdown("#### 🖼️ Visual Comparison Summary")
        st.image(st.session_state.raw_images[active_name], caption="Original Ingested Visual", use_container_width=True)
    
    with e_col2:
        st.markdown("#### ⚙️ Final Processed Visual")
        st.image(curr_img, caption="Fully Preprocessed & Feature-Analyzed Visual", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📥 Multi-Format Export Studio")
    
    exp1, exp2, exp3 = st.columns(3)
    
    with exp1:
        st.markdown("##### Export Processed Image")
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR) if len(curr_img.shape)==3 else curr_img)
        if is_success:
            st.download_button(
                label="📥 Download Processed Image (PNG)",
                data=buffer.tobytes(),
                file_name=f"processed_{active_name}",
                mime="image/png",
                use_container_width=True,
            )

    with exp2:
        st.markdown("##### Export Structured Features")
        if "extracted_df" in st.session_state:
            csv_b = st.session_state.extracted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Features DataFrame (CSV)",
                data=csv_b,
                file_name="visual_structured_features.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.write("Extract features in Tab 3 first.")

    with exp3:
        st.markdown("##### Export Executive HTML Report")
        html_report = f"""
        <html>
        <head><title>Image Analysis Report - {active_name}</title></head>
        <body style="font-family: Arial, sans-serif; margin: 40px; background: #FAF6EF;">
            <h2>Image Processing & Structured Vision Analysis Report</h2>
            <hr>
            <p><strong>Active Image:</strong> {active_name}</p>
            <p><strong>Resolution:</strong> {w} x {h} px</p>
            <p><strong>Applied Transformations:</strong> {banner_steps}</p>
            <h3>Extracted Key Metrics</h3>
            <pre>{pd.Series(extract_full_image_features(curr_img, active_name)).to_string()}</pre>
        </body>
        </html>
        """
        st.download_button(
            label="📄 Download Executive Report (HTML)",
            data=html_report.encode('utf-8'),
            file_name=f"report_{active_name}.html",
            mime="text/html",
            use_container_width=True,
        )