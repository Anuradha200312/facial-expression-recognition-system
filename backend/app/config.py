import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))

DEFAULT_HUMAN_MODEL_PATH = os.path.join(MODEL_DIR, "yolo11n.pt")
DEFAULT_FACE_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n-face.pt")
DEFAULT_EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "emotions_model.pt")

def resolve_model_paths():
    """Resolves face and emotion model paths using explicit env variables, default filenames, or auto-discovery."""
    human_path = os.getenv("HUMAN_MODEL_PATH", DEFAULT_HUMAN_MODEL_PATH)
    face_path = os.getenv("FACE_MODEL_PATH", DEFAULT_FACE_MODEL_PATH)
    emotion_path = os.getenv("EMOTION_MODEL_PATH", DEFAULT_EMOTION_MODEL_PATH)

    # Auto-discovery fallback if default files are not found
    if not os.path.exists(face_path) or not os.path.exists(emotion_path):
        if os.path.exists(MODEL_DIR):
            pt_files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
            for pt in pt_files:
                fname = os.path.basename(pt).lower()
                if not os.path.exists(human_path) and "yolo11n" in fname:
                    human_path = pt
                elif not os.path.exists(face_path) and ("face" in fname or "yolov8n-face" in fname):
                    face_path = pt
                elif not os.path.exists(emotion_path) and ("emotion" in fname or "cls" in fname):
                    emotion_path = pt

    return human_path, face_path, emotion_path

HUMAN_MODEL_PATH, FACE_MODEL_PATH, EMOTION_MODEL_PATH = resolve_model_paths()

HUMAN_CONF_THRESH = float(os.getenv("HUMAN_CONF_THRESH", "0.35"))
FACE_CONF_THRESH = float(os.getenv("FACE_CONF_THRESH", "0.40"))
EMOTION_CONF_THRESH = float(os.getenv("EMOTION_CONF_THRESH", "0.00"))
FACE_CROP_MARGIN = float(os.getenv("FACE_CROP_MARGIN", "0.15"))

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt"]
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_RESOLUTION = 20
DEFAULT_VIDEO_FRAME_STRIDE = 3
