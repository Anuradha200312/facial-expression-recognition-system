import cv2
import tempfile
import shutil
import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .schemas import PredictionResponse, HealthCheckResponse
from .pipeline import Pipeline
from .preprocess import ImageValidationError, extract_images_from_zip
from .config import DEFAULT_VIDEO_FRAME_STRIDE, FACE_CONF_THRESH, EMOTION_CONF_THRESH
from .database import engine, Base, get_db
from .models import User, DetectionLog
from .auth import auth_router, get_current_user

# Initialize database
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Face Emotion Recognition API",
    description="FastAPI service for 2-stage face detection and emotion classification.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

pipeline = Pipeline()

def save_log(db: Session, user: User, filename: str, payload: dict):
    # Extract highest level info
    total_faces = payload.get("total_faces", 0)
    primary_emotion = None
    if total_faces > 0 and payload.get("detections"):
        primary_emotion = payload["detections"][0].get("emotion_label")
    elif total_faces > 0 and payload.get("frame_timeline"):
        # For videos, just get first detection
        for frame in payload["frame_timeline"]:
            if frame.get("detections"):
                primary_emotion = frame["detections"][0].get("emotion_label")
                break
    
    log_entry = DetectionLog(
        user_id=user.id,
        filename=filename,
        total_faces=total_faces,
        primary_emotion=primary_emotion,
        execution_time_ms=payload.get("execution_time_ms"),
        full_json_payload=payload
    )
    db.add(log_entry)
    db.commit()

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Face Emotion Recognition API is running.",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
def health_check():
    return {
        "status": "ok",
        "service": "Face Emotion Recognition API",
        "face_model_loaded": pipeline.face_detector.is_loaded,
        "emotion_model_loaded": pipeline.classifier.is_loaded
    }

@app.get("/my-logs", tags=["Logs"])
def get_my_logs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    logs = db.query(DetectionLog).filter(DetectionLog.user_id == current_user.id).order_by(DetectionLog.timestamp.desc()).all()
    return [{"id": log.id, "filename": log.filename, "timestamp": log.timestamp, "total_faces": log.total_faces, "primary_emotion": log.primary_emotion, "execution_time_ms": log.execution_time_ms} for log in logs]

@app.get("/all-logs", tags=["Logs"])
def get_all_logs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.username != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    logs = db.query(DetectionLog).join(User).order_by(DetectionLog.timestamp.desc()).all()
    # Add username to response
    result = []
    for log in logs:
        user = db.query(User).filter(User.id == log.user_id).first()
        result.append({
            "id": log.id, 
            "username": user.username if user else "unknown",
            "filename": log.filename, 
            "timestamp": log.timestamp, 
            "total_faces": log.total_faces, 
            "primary_emotion": log.primary_emotion, 
            "execution_time_ms": log.execution_time_ms
        })
    return result


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    file: UploadFile = File(...),
    conf_thresh: float = Query(None, ge=0.0, le=1.0, description="Minimum face detection confidence threshold"),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0, description="Minimum emotion confidence threshold"),
    margin: float = Query(None, ge=0.0, le=0.5, description="Face crop expansion margin ratio"),
    is_stream: bool = Query(False, description="Whether input is live camera frame for tracking"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        contents = file.file.read()
        payload, _ = pipeline.process_bytes(
            contents,
            filename=file.filename or "uploaded.jpg",
            face_conf_thresh=conf_thresh,
            emotion_conf_thresh=emotion_conf_thresh,
            margin=margin,
            is_stream=is_stream
        )
        save_log(db, current_user, file.filename or "uploaded.jpg", payload)
        return payload
    except ImageValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

@app.post("/predict-annotated", tags=["Prediction"])
def predict_annotated(
    file: UploadFile = File(...),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5),
    is_stream: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        contents = file.file.read()
        payload, annotated_bgr = pipeline.process_bytes(
            contents,
            filename=file.filename or "uploaded.jpg",
            face_conf_thresh=conf_thresh,
            emotion_conf_thresh=emotion_conf_thresh,
            margin=margin,
            is_stream=is_stream
        )
        save_log(db, current_user, file.filename or "uploaded.jpg", payload)
        success, encoded_image = cv2.imencode(".jpg", annotated_bgr)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")
    except ImageValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

@app.post("/predict-zip", tags=["Prediction"])
def predict_zip(
    file: UploadFile = File(...),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        # Stream file to disk to prevent OOM
        shutil.copyfileobj(file.file, temp_zip)
        temp_zip.close()
        
        with open(temp_zip.name, "rb") as f:
            contents = f.read()
            
        valid_images, rejections = extract_images_from_zip(contents)

        results = []
        for filename, img_bytes in valid_images:
            payload, _ = pipeline.process_bytes(
                img_bytes,
                filename=filename,
                face_conf_thresh=conf_thresh,
                emotion_conf_thresh=emotion_conf_thresh,
                margin=margin
            )
            save_log(db, current_user, f"{file.filename}/{filename}", payload)
            results.append(payload)

        return {
            "status": "success",
            "archive_filename": file.filename,
            "total_extracted_images": len(valid_images),
            "total_rejected_files": len(rejections),
            "results": results,
            "rejections": rejections
        }
    except ImageValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process zip archive: {str(e)}")
    finally:
        if os.path.exists(temp_zip.name):
            os.remove(temp_zip.name)

@app.post("/predict-video", tags=["Prediction"])
def predict_video(
    file: UploadFile = File(...),
    frame_stride: int = Query(DEFAULT_VIDEO_FRAME_STRIDE, ge=1, le=30),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    in_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "video.mp4")[1])
    try:
        shutil.copyfileobj(file.file, in_file)
        in_file.close()

        payload, out_path = pipeline.process_video_file(
            in_path=in_file.name,
            filename=file.filename or "video.mp4",
            frame_stride=frame_stride,
            face_conf_thresh=conf_thresh,
            emotion_conf_thresh=emotion_conf_thresh,
            margin=margin
        )
        
        save_log(db, current_user, file.filename or "video.mp4", payload)
        
        if os.path.exists(out_path):
            os.remove(out_path)
            
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        if os.path.exists(in_file.name):
            os.remove(in_file.name)

@app.post("/predict-video-file", tags=["Prediction"])
def predict_video_file(
    file: UploadFile = File(...),
    frame_stride: int = Query(DEFAULT_VIDEO_FRAME_STRIDE, ge=1, le=30),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    in_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "video.mp4")[1])
    try:
        shutil.copyfileobj(file.file, in_file)
        in_file.close()

        payload, out_path = pipeline.process_video_file(
            in_path=in_file.name,
            filename=file.filename or "video.mp4",
            frame_stride=frame_stride,
            face_conf_thresh=conf_thresh,
            emotion_conf_thresh=emotion_conf_thresh,
            margin=margin
        )
        
        save_log(db, current_user, file.filename or "video.mp4", payload)

        with open(out_path, "rb") as f:
            out_bytes = f.read()
            
        os.remove(out_path)
        return Response(content=out_bytes, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video file processing failed: {str(e)}")
    finally:
        if os.path.exists(in_file.name):
            os.remove(in_file.name)
