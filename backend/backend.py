"""
FastAPI Backend for Wheat Detection Application
================================================

This backend serves a YOLOv11 model trained for wheat head detection.

Setup:
------
1. Place your trained model at: models/best.pt
   (Copy from: wheat_detection/yolo11m_run1/weights/best.pt)
2. Install: pip install fastapi uvicorn ultralytics opencv-python pillow python-multipart
3. Run: python backend.py

API will be available at: http://localhost:8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
from pathlib import Path
import uuid
import os
import shutil
from datetime import datetime

# Configuration
MODEL_PATH = "bestyolo_wheat.pt"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="🌾 Wheat Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
print("🔄 Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"💡 Copy your model: cp wheat_detection/yolo11m_run1/weights/best.pt {MODEL_PATH}")
    model = None


@app.get("/")
async def root():
    return {
        "message": "🌾 Wheat Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
async def model_info():
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": "YOLOv11 Medium",
        "classes": model.names,
        "input_size": 640
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = 0.25):
    """Detect wheat heads in image"""
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be 0.0-1.0")
    
    input_path = None
    output_path = None
    
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = Path(file.filename).suffix or ".jpg"
        
        input_path = UPLOAD_DIR / f"{timestamp}_{file_id}_input{ext}"
        output_path = UPLOAD_DIR / f"{timestamp}_{file_id}_output.jpg"
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run detection
        results = model.predict(
            source=str(input_path),
            conf=confidence,
            save=False,
            verbose=False
        )
        
        result = results[0]
        
        # Extract detections
        detections = []
        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "id": idx + 1,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1)
                },
                "class": "wheat_head"
            })
        
        # Save annotated image
        annotated = result.plot()
        cv2.imwrite(str(output_path), annotated)
        
        # Cleanup input
        os.remove(input_path)
        
        # Statistics
        avg_conf = sum(d["confidence"] for d in detections) / len(detections) if detections else 0
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "stats": {
                "total": len(detections),
                "avg_confidence": round(avg_conf, 3)
            },
            "output_image": output_path.name,
            "image_size": {
                "width": result.orig_shape[1],
                "height": result.orig_shape[0]
            }
        }
        
    except Exception as e:
        if input_path and input_path.exists():
            os.remove(input_path)
        if output_path and output_path.exists():
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🌾 Wheat Detection API")
    print("="*50)
    print(f"URL: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)