import os
import io
import time
import queue
import base64
import zipfile
import requests
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from components.visualizer import (
    render_predictions,
    render_predictions_bytes,
    create_video_from_frames,
    EMOTION_COLOR_MAP
)

# Page Setup
st.set_page_config(
    page_title="AI Facial Emotion Intelligence",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# System Constants
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = "USER"
if "full_name" not in st.session_state:
    st.session_state.full_name = ""
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

# Config State
if "FACE_CONF_THRESH" not in st.session_state:
    st.session_state.FACE_CONF_THRESH = 0.40
if "EMOTION_CONF_THRESH" not in st.session_state:
    st.session_state.EMOTION_CONF_THRESH = 0.00
if "VIDEO_FRAME_STRIDE" not in st.session_state:
    st.session_state.VIDEO_FRAME_STRIDE = 3
if "FACE_CROP_MARGIN" not in st.session_state:
    st.session_state.FACE_CROP_MARGIN = 0.15

# Modern CSS
st.markdown("""
<style>
    /* Global Reset & Colors */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F8FAFC;
        font-weight: 700;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        text-align: center;
        padding: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #3B82F6;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Role Badges */
    .role-badge-admin {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .role-badge-user {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Result states */
    .result-success { color: #10B981; font-weight: 600; }
    .result-warning { color: #F59E0B; font-weight: 600; }
    .result-error { color: #EF4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def get_auth_headers(token=None):
    if token:
        return {"Authorization": f"Bearer {token}"}
    try:
        return {"Authorization": f"Bearer {st.session_state.get('jwt_token', '')}"}
    except Exception:
        return {}

def call_predict_api(image_bytes: bytes, filename: str = "upload.jpg", is_stream: bool = False, token: str = None, conf_thresh: float = None, emo_thresh: float = None, margin: float = None):
    try:
        c_thresh = conf_thresh if conf_thresh is not None else st.session_state.get("FACE_CONF_THRESH", 0.4)
        e_thresh = emo_thresh if emo_thresh is not None else st.session_state.get("EMOTION_CONF_THRESH", 0.0)
        m_thresh = margin if margin is not None else st.session_state.get("FACE_CROP_MARGIN", 0.15)
    except:
        c_thresh, e_thresh, m_thresh = 0.4, 0.0, 0.15
        
    params = {
        "conf_thresh": c_thresh,
        "emotion_conf_thresh": e_thresh,
        "margin": m_thresh,
        "is_stream": is_stream
    }
    files = {"file": (filename, image_bytes, "image/jpeg")}
    return requests.post(f"{BACKEND_URL}/predict", params=params, files=files, headers=get_auth_headers(token), timeout=15)

def admin_auto_register():
    # Silently ensure admin account exists
    requests.post(f"{BACKEND_URL}/register", json={"username": "admin", "email": "admin@system.local", "password": "admin@123"})

# -------------------------------------------------------------------
# AUTHENTICATION UI
# -------------------------------------------------------------------
if not st.session_state.logged_in:
    st.markdown('<div class="main-title" style="text-align: center; margin-top: 5vh;">AI Facial Emotion Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title" style="text-align: center;">Secure Authentication Portal</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_mode = st.radio("Select Portal", ["User Login", "User Registration", "Admin Login"], horizontal=True, label_visibility="collapsed")
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if auth_mode == "User Login":
            st.markdown("### 👤 User Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", type="primary", use_container_width=True):
                    if username == "admin":
                        st.error("Please use the Admin Login portal.")
                    else:
                        resp = requests.post(f"{BACKEND_URL}/login", data={"username": username, "password": password})
                        if resp.status_code == 200:
                            st.session_state.jwt_token = resp.json()["access_token"]
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.role = "USER"
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
                            
        elif auth_mode == "User Registration":
            st.markdown("### 📝 User Registration")
            with st.form("reg_form"):
                full_name = st.text_input("Full Name")
                reg_user = st.text_input("Username")
                reg_pass = st.text_input("Password", type="password")
                reg_pass_conf = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Register", type="primary", use_container_width=True):
                    if reg_user.lower() == "admin":
                        st.error("Cannot register as admin.")
                    elif reg_pass != reg_pass_conf:
                        st.error("Passwords do not match.")
                    elif not full_name or not reg_user or not reg_pass:
                        st.error("All fields are required.")
                    else:
                        resp = requests.post(f"{BACKEND_URL}/register", json={"username": reg_user, "email": full_name, "password": reg_pass})
                        if resp.status_code == 200:
                            st.success("Registration successful! Please proceed to User Login.")
                        else:
                            st.error(resp.json().get("detail", "Registration failed."))
                            
        elif auth_mode == "Admin Login":
            st.markdown("### 🛡️ Admin Login")
            with st.form("admin_form"):
                admin_user = st.text_input("Username", value="admin", disabled=True)
                admin_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Admin Login", type="primary", use_container_width=True):
                    admin_auto_register()
                    resp = requests.post(f"{BACKEND_URL}/login", data={"username": "admin", "password": admin_pass})
                    if resp.status_code == 200:
                        st.session_state.jwt_token = resp.json()["access_token"]
                        st.session_state.logged_in = True
                        st.session_state.username = "admin"
                        st.session_state.role = "ADMIN"
                        st.rerun()
                    else:
                        st.error("Invalid Admin credentials.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# -------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------
with st.sidebar:
    if st.session_state.role == "ADMIN":
        st.markdown('<div class="role-badge-admin">🛡️ ADMIN</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="role-badge-user">👤 USER</div>', unsafe_allow_html=True)
        
    st.markdown(f"### Welcome, {st.session_state.username}")
    st.markdown("---")
    
    pages = [
        "🏠 Dashboard",
        "🖼️ Image Analysis",
        "📦 Batch / ZIP",
        "🎥 Video Analysis",
        "📷 Live Camera",
        "📜 History"
    ]
    
    if st.session_state.role == "ADMIN":
        pages.append("🛠️ Admin Panel")
        
    pages.extend([
        "👤 Profile",
        "⚙️ Settings"
    ])
    
    for page in pages:
        if st.button(page, use_container_width=True, type="primary" if st.session_state.current_page == page else "secondary"):
            st.session_state.current_page = page
            st.rerun()
            
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.session_state.logged_in = False
        st.session_state.jwt_token = ""
        st.session_state.current_page = "🏠 Dashboard"
        st.rerun()

# -------------------------------------------------------------------
# MAIN CONTENT AREA
# -------------------------------------------------------------------
page = st.session_state.current_page

if page == "🏠 Dashboard":
    st.markdown('<div class="main-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Welcome back, {st.session_state.username}</div>', unsafe_allow_html=True)
    
    try:
        resp = requests.get(f"{BACKEND_URL}/my-logs", headers=get_auth_headers(), timeout=5)
        history_len = len(resp.json()) if resp.status_code == 200 else 0
    except:
        history_len = 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="card metric-card"><div class="metric-value">{history_len}</div><div class="metric-label">Total Analyses</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="card metric-card"><div class="metric-value">{int(history_len*0.8)}</div><div class="metric-label">Images Analyzed</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="card metric-card"><div class="metric-value">{int(history_len*0.2)}</div><div class="metric-label">Videos Analyzed</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="card metric-card"><div class="metric-value">Active</div><div class="metric-label">System Status</div></div>', unsafe_allow_html=True)
    
    st.markdown("### Quick Actions")
    ac1, ac2, ac3, ac4 = st.columns(4)
    if ac1.button("🖼️ Image Analysis", use_container_width=True):
        st.session_state.current_page = "🖼️ Image Analysis"
        st.rerun()
    if ac2.button("📦 Batch Analysis", use_container_width=True):
        st.session_state.current_page = "📦 Batch / ZIP"
        st.rerun()
    if ac3.button("🎥 Video Analysis", use_container_width=True):
        st.session_state.current_page = "🎥 Video Analysis"
        st.rerun()
    if ac4.button("📷 Live Camera", use_container_width=True):
        st.session_state.current_page = "📷 Live Camera"
        st.rerun()

elif page == "🖼️ Image Analysis":
    st.markdown('<div class="main-title">Image Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload an image to analyze facial expressions</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Choose Image (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_image:
        image_bytes = uploaded_image.getvalue()
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_bytes, caption="Original Image", use_container_width=True)
        
        with col2:
            if getattr(st.session_state, "last_img_name", None) != uploaded_image.name:
                with st.spinner("⏳ Analyzing... Please wait."):
                    try:
                        resp = call_predict_api(image_bytes, uploaded_image.name)
                        if resp.status_code == 200:
                            st.session_state.img_data = resp.json()
                            st.session_state.last_img_name = uploaded_image.name
                            if st.session_state.img_data["total_faces"] > 0:
                                st.session_state.img_ann = render_predictions_bytes(image_bytes, st.session_state.img_data["detections"], human_detected=st.session_state.img_data.get("human_detected", True))
                            else:
                                st.session_state.img_ann = None
                        else:
                            st.error("⚠️ Something went wrong.")
                    except:
                        st.error("⚠️ Something went wrong.")
            
            data = getattr(st.session_state, "img_data", None)
            if data and getattr(st.session_state, "last_img_name", None) == uploaded_image.name:
                if not data.get("human_detected", True):
                    st.markdown('<p class="result-error">❌ No Human Detected</p>', unsafe_allow_html=True)
                    annotated_pil = render_predictions(image_bytes, [], human_detected=False)
                    st.image(annotated_pil, use_container_width=True)
                elif data["total_faces"] == 0:
                    st.markdown('<p class="result-warning">⚠️ No Face Detected</p>', unsafe_allow_html=True)
                    annotated_pil = render_predictions(image_bytes, [], human_detected=True)
                    st.image(annotated_pil, use_container_width=True)
                else:
                    st.markdown('<p class="result-success">✓ Human Detected</p>', unsafe_allow_html=True)
                    det = data["detections"][0]
                    st.markdown(f"### {det['emotion_label']}")
                    st.progress(det['emotion_confidence'])
                    st.caption(f"{det['emotion_confidence']*100:.1f}% Confidence")
                    if st.session_state.img_ann:
                        st.image(st.session_state.img_ann, use_container_width=True)
                        st.download_button(
                            label="📥 Download Annotated Image",
                            data=st.session_state.img_ann,
                            file_name=f"annotated_{uploaded_image.name}",
                            mime="image/jpeg",
                            use_container_width=True
                        )
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📦 Batch / ZIP":
    st.markdown('<div class="main-title">Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Process multiple images or ZIP archives</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    batch_files = st.file_uploader("Upload ZIP or Images", type=["zip", "jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    
    if batch_files and st.button("Process Batch", type="primary"):
        results = []
        st.session_state.batch_annotated_images = []
        with st.spinner("⏳ Analyzing... Please wait."):
            for b_file in batch_files:
                fn = b_file.name
                b_bytes = b_file.getvalue()
                if fn.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(b_bytes), "r") as z:
                        for member in z.infolist():
                            if not member.is_dir() and not os.path.basename(member.filename).startswith("."):
                                if os.path.splitext(member.filename)[1].lower() in [".jpg", ".png", ".webp"]:
                                    img_b = z.read(member)
                                    try:
                                        resp = call_predict_api(img_b, os.path.basename(member.filename))
                                        if resp.status_code == 200:
                                            d = resp.json()
                                            status = "✓ Human Detected" if d["total_faces"] > 0 else ("⚠️ No Face Detected" if d.get("human_detected", True) else "❌ No Human Detected")
                                            emo = d["detections"][0]["emotion_label"] if d["total_faces"] > 0 else "-"
                                            conf = f"{d['detections'][0]['emotion_confidence']*100:.1f}%" if d["total_faces"] > 0 else "-"
                                            results.append({"File": os.path.basename(member.filename), "Result": status, "Emotion": emo, "Confidence": conf, "Status": "Success"})
                                            annotated_pil = render_predictions(img_b, d["detections"], human_detected=d.get("human_detected", True))
                                            buf = io.BytesIO()
                                            annotated_pil.save(buf, format="JPEG")
                                            st.session_state.batch_annotated_images.append({"filename": os.path.basename(member.filename), "dl_bytes": buf.getvalue()})
                                    except:
                                        pass
                else:
                    try:
                        resp = call_predict_api(b_bytes, fn)
                        if resp.status_code == 200:
                            d = resp.json()
                            status = "✓ Human Detected" if d["total_faces"] > 0 else ("⚠️ No Face Detected" if d.get("human_detected", True) else "❌ No Human Detected")
                            emo = d["detections"][0]["emotion_label"] if d["total_faces"] > 0 else "-"
                            conf = f"{d['detections'][0]['emotion_confidence']*100:.1f}%" if d["total_faces"] > 0 else "-"
                            results.append({"File": fn, "Result": status, "Emotion": emo, "Confidence": conf, "Status": "Success"})
                            
                            annotated_pil = render_predictions(b_bytes, d["detections"], human_detected=d.get("human_detected", True))
                            buf = io.BytesIO()
                            annotated_pil.save(buf, format="JPEG")
                            st.session_state.batch_annotated_images.append({"filename": fn, "dl_bytes": buf.getvalue()})
                    except:
                        pass
        st.session_state.batch_res = results
        st.success("✓ Analysis completed successfully.")
        
    if getattr(st.session_state, "batch_res", None):
        st.markdown("#### 🖼️ Batch Annotated Outputs (4 Per Row)")
        results_with_images = getattr(st.session_state, "batch_annotated_images", [])
        if results_with_images:
            for row_idx in range(0, len(results_with_images), 4):
                row_items = results_with_images[row_idx:row_idx+4]
                cols = st.columns(4)
                for col_idx, item in enumerate(row_items):
                    with cols[col_idx]:
                        fn = item["filename"]
                        dl_bytes = item["dl_bytes"]
                        st.image(dl_bytes, caption=f"📄 {fn}", use_container_width=True)
                        st.download_button(label="📥 Download", data=dl_bytes, file_name=f"annotated_{fn}", mime="image/jpeg", key=f"dl_grid_{row_idx}_{col_idx}_{fn}", use_container_width=True)
            st.markdown("---")

        st.markdown("#### 📋 Batch Summary Table")
        df = pd.DataFrame(st.session_state.batch_res)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Batch CSV",
            data=csv,
            file_name="batch_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🎥 Video Analysis":
    st.markdown('<div class="main-title">Video Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyze pre-recorded video files</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        st.video(uploaded_video)
        if st.button("Analyze Video", type="primary"):
            with st.spinner("⏳ Analyzing... Please wait."):
                try:
                    files = {"file": (uploaded_video.name, uploaded_video.getvalue(), "video/mp4")}
                    params = {"frame_stride": st.session_state.VIDEO_FRAME_STRIDE, "conf_thresh": st.session_state.FACE_CONF_THRESH, "emotion_conf_thresh": st.session_state.EMOTION_CONF_THRESH, "margin": st.session_state.FACE_CROP_MARGIN}
                    resp = requests.post(f"{BACKEND_URL}/predict-video", params=params, files=files, headers=get_auth_headers(), timeout=180)
                    if resp.status_code == 200:
                        v_data = resp.json()
                        st.success("✓ Analysis completed successfully.")
                        if v_data.get("annotated_video_b64"):
                            st.video(base64.b64decode(v_data["annotated_video_b64"]))
                    else:
                        st.error("⚠️ Something went wrong.")
                except:
                    st.error("⚠️ Something went wrong.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📷 Live Camera":
    st.markdown('<div class="main-title">Live Camera</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time emotion detection</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    import threading
    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self, token, frame_stride, c_thresh, e_thresh, margin):
            self.frame_count = 0
            self.last_img_bgr = None
            self.log_queue = queue.Queue()
            self.token = token
            self.frame_stride = frame_stride
            self.c_thresh = c_thresh
            self.e_thresh = e_thresh
            self.margin = margin

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            if self.frame_count % self.frame_stride == 0 or self.last_img_bgr is None:
                success, buffer = cv2.imencode('.jpg', img_bgr)
                if success:
                    try:
                        resp = call_predict_api(buffer.tobytes(), "stream.jpg", is_stream=True, token=self.token, conf_thresh=self.c_thresh, emo_thresh=self.e_thresh, margin=self.margin)
                        if resp.status_code == 200:
                            data = resp.json()
                            dets = data.get("detections", [])
                            if len(dets) == 0:
                                self.log_queue.put({"Result": "⚠️ No Face Detected" if data.get("human_detected", True) else "❌ No Human Detected", "Emotion": "-", "Confidence": "-"})
                            else:
                                self.log_queue.put({
                                    "Result": "✓ Human Detected",
                                    "Emotion": dets[0]["emotion_label"],
                                    "Confidence": f"{dets[0]['emotion_confidence']*100:.1f}%"
                                })
                            ann_pil = render_predictions(buffer.tobytes(), dets, human_detected=data.get("human_detected", True))
                            img_rgb = np.array(ann_pil)
                            self.last_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        else:
                            self.last_img_bgr = img_bgr
                    except Exception:
                        self.last_img_bgr = img_bgr
            
            out_img = self.last_img_bgr if self.last_img_bgr is not None else img_bgr
            if not out_img.flags['C_CONTIGUOUS']:
                out_img = np.ascontiguousarray(out_img)
            return av.VideoFrame.from_ndarray(out_img, format="bgr24")

    # Get variables in main thread scope to avoid background thread exceptions
    jwt_tok = st.session_state.get("jwt_token", "")
    stride = st.session_state.get("VIDEO_FRAME_STRIDE", 3)
    c_thresh = st.session_state.get("FACE_CONF_THRESH", 0.4)
    e_thresh = st.session_state.get("EMOTION_CONF_THRESH", 0.0)
    m_thresh = st.session_state.get("FACE_CROP_MARGIN", 0.15)

    ctx = webrtc_streamer(
        key="emotion-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.stunprotocol.org:3478"]}
            ]
        },
        video_processor_factory=lambda: EmotionVideoProcessor(token=jwt_tok, frame_stride=stride, c_thresh=c_thresh, e_thresh=e_thresh, margin=m_thresh),
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 60}}, "audio": False},
        video_html_attrs={
            "style": {"width": "100%", "border-radius": "8px"},
            "autoPlay": True,
            "controls": False,
            "playsinline": True,
            "muted": True
        },
        async_processing=True
    )
    
    res_placeholder = st.empty()
    
    if "live_logs" not in st.session_state:
        st.session_state.live_logs = []

    if ctx.state.playing:
        if len(st.session_state.live_logs) > 0:
            st.session_state.live_logs = [] # Reset on new play
            
        st.markdown('<p class="result-success">● Camera Active (Results overlaying on video feed...)</p>', unsafe_allow_html=True)
        while True:
            if ctx.video_processor:
                try:
                    log = ctx.video_processor.log_queue.get(timeout=1.0)
                    # Add a timestamp
                    log["Timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.live_logs.append(log)
                except queue.Empty:
                    pass
            else:
                break
    else:
        if len(st.session_state.live_logs) > 0:
            st.success(f"Live session ended. Recorded {len(st.session_state.live_logs)} frames.")
            df = pd.DataFrame(st.session_state.live_logs)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Session CSV",
                data=csv,
                file_name="live_camera_session.csv",
                mime="text/csv",
                use_container_width=True
            )
            if st.button("Clear Session Data", use_container_width=True):
                st.session_state.live_logs = []
                st.rerun()
                
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📜 History":
    st.markdown('<div class="main-title">Analysis History</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    try:
        resp = requests.get(f"{BACKEND_URL}/my-logs", headers=get_auth_headers(), timeout=5)
        if resp.status_code == 200 and len(resp.json()) > 0:
            df = pd.DataFrame(resp.json())
            df['Date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['Result'] = df['total_faces'].apply(lambda x: "✓ Detected" if x > 0 else "⚠️ None")
            df['File'] = df['filename']
            df['Emotion'] = df['primary_emotion']
            st.dataframe(df[['File', 'Result', 'Emotion', 'Date']], use_container_width=True)
        else:
            st.markdown("📂 No analysis history yet.")
    except:
        st.error("⚠️ Something went wrong.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "👤 Profile":
    st.markdown('<div class="main-title">Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👤 User Information")
    st.write(f"**Full Name:** User")
    st.write(f"**Username:** {st.session_state.username}")
    st.write(f"**Account Type:** {st.session_state.role}")
    if st.button("Edit Profile"):
        st.info("Profile editing coming soon.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "⚙️ Settings":
    st.markdown('<div class="main-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Appearance")
    st.radio("Theme", ["Dark", "Light"])
    st.markdown("### Account")
    st.text_input("New Password", type="password")
    if st.button("Change Password"):
        st.success("Password updated (Mock).")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🛠️ Admin Panel":
    if st.session_state.role != "ADMIN":
        st.error("Access Denied")
        st.stop()
        
    st.markdown('<div class="main-title">Admin Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Welcome back, Administrator</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Pipeline Settings")
    st.session_state.FACE_CONF_THRESH = st.slider("Face Detection Confidence", 0.0, 1.0, st.session_state.FACE_CONF_THRESH)
    st.session_state.EMOTION_CONF_THRESH = st.slider("Emotion Classification Confidence", 0.0, 1.0, st.session_state.EMOTION_CONF_THRESH)
    st.session_state.VIDEO_FRAME_STRIDE = st.number_input("Video / Live Frame Subsampling (N)", 1, 30, st.session_state.VIDEO_FRAME_STRIDE)
    st.session_state.FACE_CROP_MARGIN = st.slider("Face Crop Margin", 0.0, 1.0, st.session_state.FACE_CROP_MARGIN)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🩺 System Status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health.status_code == 200:
            h_data = health.json()
            st.success("🟢 API Server Connected")
            st.write(f"Face Model Loaded: {h_data['face_model_loaded']}")
            st.write(f"Emotion Model Loaded: {h_data['emotion_model_loaded']}")
        else:
            st.warning("🟡 API Server Degraded")
    except:
        st.error("🔴 API Server Disconnected")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👥 All Users Global History")
    try:
        resp = requests.get(f"{BACKEND_URL}/all-logs", headers=get_auth_headers(), timeout=5)
        if resp.status_code == 200 and len(resp.json()) > 0:
            df = pd.DataFrame(resp.json())
            df['Date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['Result'] = df['total_faces'].apply(lambda x: "✓ Detected" if x > 0 else "⚠️ None")
            df['File'] = df['filename']
            df['Emotion'] = df['primary_emotion']
            df['User'] = df['username']
            st.dataframe(df[['User', 'File', 'Result', 'Emotion', 'Date']], use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Global CSV", data=csv, file_name="global_history.csv", mime="text/csv", use_container_width=True)
        else:
            st.markdown("📂 No global analysis history yet.")
    except:
        st.error("⚠️ Failed to load global history.")
    st.markdown('</div>', unsafe_allow_html=True)
