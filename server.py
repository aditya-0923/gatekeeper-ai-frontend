"""GateKeeper AI backend.

Run:  python server.py   (then open http://localhost:8000)

Put these files next to this script:
  yolov8n.pt
  face_detection_yunet_2023mar.onnx
  face_recognition_sface_2021dec.onnx
"""
import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

HERE = Path(__file__).parent
YUNET = str(HERE / "face_detection_yunet_2023mar.onnx")
SFACE = str(HERE / "face_recognition_sface_2021dec.onnx")
COSINE_MATCH = 0.363  # SFace's recommended cosine threshold
PERSON_CONF = 0.5

yolo = YOLO(str(HERE / "yolov8n.pt"))
detector = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.8, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE, "")
lock = threading.Lock()  # YuNet input size is shared state
enrolled: dict[str, np.ndarray] = {}  # name -> SFace feature

app = FastAPI(title="GateKeeper AI")


class EnrollIn(BaseModel):
    name: str
    image: str


class AnalyzeIn(BaseModel):
    image: str
    zone: list[list[float]]


def decode(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        arr = np.frombuffer(base64.b64decode(data), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        raise HTTPException(400, "The image could not be read.")
    return img


def find_faces(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    with lock:
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
    return faces if faces is not None else np.empty((0, 15))


def embed(img: np.ndarray, face: np.ndarray) -> np.ndarray:
    return recognizer.feature(recognizer.alignCrop(img, face))


def identify(feature: np.ndarray):
    best, best_score = None, 0.0
    for name, ref in enrolled.items():
        score = recognizer.match(feature, ref, cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best, best_score = name, score
    return (best, best_score) if best_score >= COSINE_MATCH else (None, best_score)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/people")
def people():
    return {"people": sorted(enrolled)}


@app.post("/api/enroll")
def enroll(body: EnrollIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Enter a name or ID.")
    img = decode(body.image)
    faces = find_faces(img)
    if len(faces) == 0:
        raise HTTPException(422, "No face found. Face the camera in good light and try again.")
    face = max(faces, key=lambda f: f[2] * f[3])  # largest face
    enrolled[name] = embed(img, face)
    return {"ok": True, "name": name}


@app.delete("/api/people/{name}")
def remove(name: str):
    enrolled.pop(name, None)
    return {"ok": True}


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    img = decode(body.image)
    if len(body.zone) < 3:
        raise HTTPException(400, "The zone needs at least 3 corners.")
    poly = np.array(body.zone, dtype=np.float32).reshape(-1, 1, 2)
    h, w = img.shape[:2]
    out = []
    for box in yolo(img, classes=[0], conf=PERSON_CONF, verbose=False)[0].boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        foot = ((x1 + x2) / 2, float(y2))
        inside = cv2.pointPolygonTest(poly, foot, False) >= 0
        name, score = None, 0.0
        # Face recognition only runs when a person is inside the zone.
        if inside and enrolled:
            top = max(y1, 0)
            head = img[top : top + max(int((y2 - y1) * 0.4), 1), max(x1, 0) : min(x2, w)]
            if head.shape[0] > 20 and head.shape[1] > 20:
                faces = find_faces(head)
                if len(faces):
                    face = max(faces, key=lambda f: f[2] * f[3])
                    name, score = identify(embed(head, face))
        out.append(
            {
                "box": [x1, y1, x2, y2],
                "inside": bool(inside),
                "name": name,
                "score": round(float(score), 3),
            }
        )
    return {"detections": out}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
