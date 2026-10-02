import base64
import time
import os
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from .detector import HumanDetector, FaceDetector, CentroidTracker
from .classifier import EmotionClassifier
from .preprocess import validate_and_decode_image, expand_bbox
from .config import DEFAULT_VIDEO_FRAME_STRIDE, HUMAN_CONF_THRESH, FACE_CONF_THRESH, EMOTION_CONF_THRESH

class Pipeline:
    def __init__(self, human_detector: HumanDetector = None, face_detector: FaceDetector = None, classifier: EmotionClassifier = None):
        self.human_detector = human_detector or HumanDetector()
        self.face_detector = face_detector or FaceDetector()
        self.classifier = classifier or EmotionClassifier()

    def process_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        human_conf_thresh: float = None,
        face_conf_thresh: float = None,
        emotion_conf_thresh: float = None,
        margin: float = None,
        is_stream: bool = False
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        start_time = time.time()
        img_bgr = validate_and_decode_image(image_bytes)

        if human_conf_thresh is None:
            human_conf_thresh = HUMAN_CONF_THRESH
        if face_conf_thresh is None:
            face_conf_thresh = FACE_CONF_THRESH
        if emotion_conf_thresh is None:
            emotion_conf_thresh = EMOTION_CONF_THRESH

        humans = self.human_detector.detect(img_bgr, conf_thresh=human_conf_thresh)
        print(f"[DEBUG] Detected humans: {humans}", flush=True)
        human_detected = len(humans) > 0
        
        detections = []
        annotated_bgr = img_bgr.copy()
        
        # We will collect global faces for tracking if in stream mode
        global_faces = []
        
        for human in humans:
            hx1, hy1, hx2, hy2 = human["bbox"]
            human_crop = img_bgr[hy1:hy2, hx1:hx2]
            
            # Detect faces inside the human crop
            local_faces = self.face_detector.detect(human_crop, conf_thresh=face_conf_thresh, margin=margin)
            print(f"[DEBUG] Local faces in human crop: {local_faces}", flush=True)
            
            for face in local_faces:
                fx1, fy1, fx2, fy2 = face["bbox"]
                # Convert to global coordinates
                gx1, gy1 = hx1 + fx1, hy1 + fy1
                gx2, gy2 = hx1 + fx2, hy1 + fy2
                
                global_faces.append({
                    "bbox": (gx1, gy1, gx2, gy2),
                    "det_conf": face["det_conf"],
                    "human_bbox": human["bbox"],
                    "human_conf": human.get("det_conf", 0.0)
                })
        if is_stream:
            # We track faces globally across frames
            tracked_faces = self.face_detector.tracker.update(global_faces)
        else:
            tracked_faces = global_faces

        for idx, face in enumerate(tracked_faces):
            gx1, gy1, gx2, gy2 = face["bbox"]
            face_crop = img_bgr[gy1:gy2, gx1:gx2]
            
            emo_res = self.classifier.classify(face_crop)
            if emo_res["cls_conf"] < emotion_conf_thresh:
                continue

            track_id = face.get("track_id", idx + 1)
            hx1, hy1, hx2, hy2 = face.get("human_bbox", (gx1, gy1, gx2, gy2))
            human_conf = face.get("human_conf", 0.0)

            detection_entry = {
                "face_id": idx,
                "track_id": track_id,
                "human_bbox": {"x1": hx1, "y1": hy1, "x2": hx2, "y2": hy2},
                "human_confidence": round(human_conf, 4),
                "bbox": {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2},
                "face_confidence": round(face["det_conf"], 4),
                "emotion_label": emo_res["label"],
                "emotion_confidence": round(emo_res["cls_conf"], 4),
                "all_scores": emo_res.get("all_scores")
            }
            detections.append(detection_entry)

            # Draw annotations
            # Human box (Green)
            cv2.rectangle(annotated_bgr, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
            h_text = f"Human {human_conf*100:.1f}%"
            (htw, hth), _ = cv2.getTextSize(h_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_bgr, (hx1, max(0, hy1 - hth - 8)), (hx1 + htw + 4, hy1), (0, 255, 0), -1)
            cv2.putText(annotated_bgr, h_text, (hx1 + 2, max(hth, hy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Face box (Blue)
            cv2.rectangle(annotated_bgr, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
            f_text = f"#{track_id} {emo_res['label']} {emo_res['cls_conf']*100:.1f}%"
            (ftw, fth), _ = cv2.getTextSize(f_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_bgr, (gx1, max(0, gy1 - fth - 8)), (gx1 + ftw + 4, gy1), (255, 0, 0), -1)
            cv2.putText(annotated_bgr, f_text, (gx1 + 2, max(fth, gy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        exec_time_ms = round((time.time() - start_time) * 1000, 2)

        result_payload = {
            "status": "success",
            "filename": filename,
            "total_faces": len(detections),
            "human_detected": human_detected,
            "detections": detections,
            "execution_time_ms": exec_time_ms
        }

        return result_payload, annotated_bgr

    def process_video_bytes(
        self,
        video_bytes: bytes,
        filename: str = "video.mp4",
        frame_stride: int = DEFAULT_VIDEO_FRAME_STRIDE,
        human_conf_thresh: float = None,
        face_conf_thresh: float = None,
        emotion_conf_thresh: float = None,
        margin: float = None
    ) -> Tuple[Dict[str, Any], bytes]:
        start_time = time.time()
        if human_conf_thresh is None:
            human_conf_thresh = HUMAN_CONF_THRESH
        if face_conf_thresh is None:
            face_conf_thresh = FACE_CONF_THRESH
        if emotion_conf_thresh is None:
            emotion_conf_thresh = EMOTION_CONF_THRESH
        if frame_stride is None or frame_stride < 1:
            frame_stride = DEFAULT_VIDEO_FRAME_STRIDE

        local_tracker = CentroidTracker()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as in_file:
            in_file.write(video_bytes)
            in_path = in_file.name

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = out_file.name
        out_file.close()

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            os.remove(in_path)
            raise ValueError("Could not open or decode uploaded video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_records = []
        annotated_frames_rgb = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                humans = self.human_detector.detect(frame, conf_thresh=human_conf_thresh)
                global_faces = []
                for human in humans:
                    hx1, hy1, hx2, hy2 = human["bbox"]
                    human_crop = frame[hy1:hy2, hx1:hx2]
                    local_faces = self.face_detector.detect(human_crop, conf_thresh=face_conf_thresh, margin=margin)
                    for face in local_faces:
                        fx1, fy1, fx2, fy2 = face["bbox"]
                        gx1, gy1 = hx1 + fx1, hy1 + fy1
                        gx2, gy2 = hx1 + fx2, hy1 + fy2
                        global_faces.append({
                            "bbox": (gx1, gy1, gx2, gy2),
                            "det_conf": face["det_conf"],
                            "human_bbox": human["bbox"],
                            "human_conf": human["det_conf"]
                        })
                
                tracked_faces = local_tracker.update(global_faces)
                annotated_frame = frame.copy()
                frame_faces = []

                for idx, face in enumerate(tracked_faces):
                    gx1, gy1, gx2, gy2 = face["bbox"]
                    face_crop = frame[gy1:gy2, gx1:gx2]
                    
                    emo_res = self.classifier.classify(face_crop)
                    if emo_res["cls_conf"] < emotion_conf_thresh:
                        continue

                    track_id = face.get("track_id", idx + 1)
                    hx1, hy1, hx2, hy2 = face.get("human_bbox", (gx1, gy1, gx2, gy2))
                    human_conf = face.get("human_conf", 0.0)

                    rec = {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "face_id": idx,
                        "track_id": track_id,
                        "human_bbox": {"x1": hx1, "y1": hy1, "x2": hx2, "y2": hy2},
                        "human_confidence": round(human_conf, 4),
                        "bbox": {"x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2},
                        "face_confidence": round(face["det_conf"], 4),
                        "emotion_label": emo_res["label"],
                        "emotion_confidence": round(emo_res["cls_conf"], 4)
                    }
                    frame_faces.append(rec)

                    thickness = max(2, int(min(w, h) / 300))
                    font_scale = max(0.4, min(w, h) / 800)

                    # Human
                    cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), thickness)
                    h_text = f"Human {human_conf*100:.1f}%"
                    (htw, hth), _ = cv2.getTextSize(h_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    cv2.rectangle(annotated_frame, (hx1, max(0, hy1 - hth - 8)), (hx1 + htw + 4, hy1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, h_text, (hx1 + 2, max(hth, hy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

                    # Face
                    cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), thickness)
                    f_text = f"#{track_id} {emo_res['label']} {emo_res['cls_conf']*100:.1f}%"
                    (ftw, fth), _ = cv2.getTextSize(f_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    cv2.rectangle(annotated_frame, (gx1, max(0, gy1 - fth - 8)), (gx1 + ftw + 4, gy1), (255, 0, 0), -1)
                    cv2.putText(annotated_frame, f_text, (gx1 + 2, max(fth, gy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                annotated_frames_rgb.append(annotated_rgb)

                frame_records.append({
                    "frame": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "human_detected": len(humans) > 0,
                    "total_faces": len(frame_faces),
                    "detections": frame_faces
                })

            frame_idx += 1

        cap.release()
        os.remove(in_path)

        out_bytes = b""
        if annotated_frames_rgb:
            try:
                import imageio
                out_fps = max(1.0, float(fps) / frame_stride)
                writer = imageio.get_writer(out_path, fps=out_fps, codec="libx264", macro_block_size=1)
                for f_rgb in annotated_frames_rgb:
                    writer.append_data(f_rgb)
                writer.close()

                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        out_bytes = f.read()
                    os.remove(out_path)
            except Exception as e:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, max(1.0, fps / frame_stride), (w, h))
                for f_rgb in annotated_frames_rgb:
                    f_bgr = cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR)
                    writer.write(f_bgr)
                writer.release()
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        out_bytes = f.read()
                    os.remove(out_path)

        exec_time_ms = round((time.time() - start_time) * 1000, 2)
        annotated_video_b64 = base64.b64encode(out_bytes).decode('utf-8') if out_bytes else ""

        payload = {
            "status": "success",
            "filename": filename,
            "total_frames_processed": len(frame_records),
            "frame_stride": frame_stride,
            "frame_timeline": frame_records,
            "execution_time_ms": exec_time_ms,
            "annotated_video_b64": annotated_video_b64
        }

        return payload, out_bytes
