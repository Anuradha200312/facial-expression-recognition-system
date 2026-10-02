import io
import os
import tempfile
import cv2
import numpy as np
from PIL import Image

EMOTION_COLOR_MAP = {
    "Angry": (239, 68, 68),     # Red
    "Disgust": (168, 85, 247),  # Purple
    "Fear": (249, 115, 22),    # Orange
    "Happy": (34, 197, 94),     # Green
    "Sad": (59, 130, 246),      # Blue
    "Surprise": (236, 72, 153), # Pink
    "Neutral": (107, 114, 128)  # Gray
}

def render_predictions(image_bytes: bytes, detections: list, human_detected: bool = True) -> Image.Image:
    """Draws bounding boxes and custom colored emotion tags onto image, including confidence scores."""
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img_bgr.shape

    # Calculate dynamic line thickness & font scale based on resolution
    thickness = max(2, int(min(w_img, h_img) / 300))
    font_scale = max(0.4, min(w_img, h_img) / 800)
    
    if not detections:
        # Draw massive text on the image
        text_scale = max(1.0, min(w_img, h_img) / 400)
        text_thick = max(2, int(min(w_img, h_img) / 150))
        if not human_detected:
            text = "NO HUMAN DETECTED"
            color = (0, 0, 255) # Red
        else:
            text = "NO FACE DETECTED"
            color = (0, 165, 255) # Orange
            
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, text_scale, text_thick)
        x = max(10, (w_img - tw) // 2)
        y = max(th + 20, int(h_img * 0.15))
        
        # Shadow for visibility
        cv2.putText(img_bgr, text, (x+2, y+2), cv2.FONT_HERSHEY_DUPLEX, text_scale, (0,0,0), text_thick+1, cv2.LINE_AA)
        cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, text_scale, color, text_thick, cv2.LINE_AA)

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        label = det["emotion_label"]
        emo_conf = det.get("emotion_confidence", 0.0)
        face_conf = det.get("face_confidence", 0.0)
        track_id = det.get("track_id") or (det.get("face_id", 0) + 1)

        # Color in BGR
        rgb_color = EMOTION_COLOR_MAP.get(label, (50, 150, 250))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        # Human Rectangle
        human_bbox = det.get("human_bbox")
        if human_bbox:
            hx1, hy1, hx2, hy2 = human_bbox["x1"], human_bbox["y1"], human_bbox["x2"], human_bbox["y2"]
            human_conf = det.get("human_confidence", 0.0)
            cv2.rectangle(img_bgr, (hx1, hy1), (hx2, hy2), (0, 255, 0), thickness)
            h_text = f"Human {human_conf*100:.1f}%"
            (htw, hth), _ = cv2.getTextSize(h_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img_bgr, (hx1, max(0, hy1 - hth - 10)), (hx1 + htw + 8, hy1), (0, 255, 0), -1)
            cv2.putText(img_bgr, h_text, (hx1 + 4, max(hth + 2, hy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        # Face Rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bgr_color, thickness)

        # Tag Text with Emotion Confidence
        tag_text = f"{label.capitalize()} {emo_conf*100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tag_y1 = max(0, y1 - th - 10)
        tag_y2 = y1
        cv2.rectangle(img_bgr, (x1, tag_y1), (x1 + tw + 8, tag_y2), bgr_color, -1)
        cv2.putText(img_bgr, tag_text, (x1 + 4, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def render_predictions_bytes(image_bytes: bytes, detections: list, format: str = "JPEG", human_detected: bool = True) -> bytes:
    """Returns encoded image bytes (JPEG/PNG) with rendered bounding boxes and confidence tags for downloading."""
    pil_img = render_predictions(image_bytes, detections, human_detected=human_detected)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()

def create_video_from_frames(frames_rgb: list, fps: float = 5.0) -> bytes:
    """Encodes a list of RGB numpy image frames into H.264 MP4 video bytes for video display and download."""
    if not frames_rgb:
        return b""
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_file.name
    out_file.close()
    try:
        import imageio
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=1)
        for frame in frames_rgb:
            writer.append_data(frame)
        writer.close()

        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                v_bytes = f.read()
            os.remove(out_path)
            return v_bytes
    except Exception:
        h, w, _ = frames_rgb[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                v_bytes = f.read()
            os.remove(out_path)
            return v_bytes
    return b""

