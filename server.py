"""GateKeeper AI backend.

Run:  python server.py   (then open http://localhost:8000)

Put these files next to this script (or point MODEL_DIR at their folder):
  yolov8n.pt
  face_detection_yunet_2023mar.onnx
  face_recognition_sface_2021dec.onnx

Settings are read from environment variables, so nothing needs editing:
  HOST, PORT          where to listen (default 127.0.0.1:8000)
  MODEL_DIR           folder holding the three model files (default: next to this script)
  DATA_DIR            where enrolled face data is saved (default: ./data)
  PERSIST_FACES       1 = keep enrolled faces across restarts, 0 = memory only (default 1)
  COSINE_MATCH        SFace match threshold (default 0.363, the value OpenCV recommends)
  PERSON_CONF         YOLO person confidence (default 0.5)
"""
from __future__ import annotations

import base64
import logging
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

log = logging.getLogger("gatekeeper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HERE = Path(__file__).parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", HERE))
DATA_DIR = Path(os.getenv("DATA_DIR", HERE / "data"))
PERSIST = os.getenv("PERSIST_FACES", "1") == "1"
COSINE_MATCH = float(os.getenv("COSINE_MATCH", "0.363"))
PERSON_CONF = float(os.getenv("PERSON_CONF", "0.5"))

IDENTITY_TTL = 3.0        # seconds a recognised person is remembered after they leave the frame
MAX_IMAGE_CHARS = 12_000_000  # ~9 MB of image data once base64-decoded
MAX_ZONE_POINTS = 50
MAX_NAME_LEN = 64
HEAD_FRACTION = 0.4       # top part of a person box that is searched for a face

YOLO_PATH = MODEL_DIR / "yolov8n.pt"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
FACES_FILE = DATA_DIR / "enrolled.npz"

for model_file in (YOLO_PATH, YUNET_PATH, SFACE_PATH):
    if not model_file.exists():
        raise SystemExit(
            f"Missing model file: {model_file.name}. See 'Download the models' in the README."
        )

yolo = YOLO(str(YOLO_PATH))
detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.8, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")

# One lock guards every model call: YuNet's input size and the YOLO tracker are shared state.
lock = threading.RLock()
enrolled: dict[str, np.ndarray] = {}   # name -> SFace feature, shape (1, 128)
identities: dict[int, dict] = {}       # tracker id -> {"name", "score", "seen"}


# ----------------------------------------------------------------------------- storage
def save_faces() -> None:
    if not PERSIST:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(enrolled)
    feats = np.vstack([enrolled[n] for n in names]) if names else np.empty((0, 128), np.float32)
    np.savez(FACES_FILE, names=np.array(names, dtype=str), feats=feats)


def load_faces() -> None:
    if not (PERSIST and FACES_FILE.exists()):
        return
    try:
        data = np.load(FACES_FILE, allow_pickle=False)
        for name, feat in zip(data["names"].tolist(), data["feats"]):
            enrolled[name] = feat.reshape(1, -1).astype(np.float32)
        log.info("Loaded %d enrolled face(s) from %s", len(enrolled), FACES_FILE)
    except Exception as exc:  # a damaged file should not stop the server
        log.warning("Could not read %s: %s", FACES_FILE, exc)


def reset_tracking() -> None:
    identities.clear()
    try:
        predictor = yolo.predictor
        if predictor is not None and getattr(predictor, "trackers", None):
            for tracker in predictor.trackers:
                tracker.reset()
    except Exception as exc:
        log.debug("Tracker reset skipped: %s", exc)


# ----------------------------------------------------------------------------- models
def decode(data: str) -> np.ndarray:
    if len(data) > MAX_IMAGE_CHARS:
        raise HTTPException(413, "The image is too large. Use a smaller camera resolution.")
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
    with lock:
        return recognizer.feature(recognizer.alignCrop(img, face))


def identify(feature: np.ndarray) -> tuple[str | None, float]:
    best, best_score = None, 0.0
    for name, ref in enrolled.items():
        score = recognizer.match(feature, ref, cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best, best_score = name, score
    return (best, best_score) if best_score >= COSINE_MATCH else (None, best_score)


def recognise_person(img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str | None, float]:
    """Look for a face in the head area of a person box and match it against enrolled faces."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    top = max(y1, 0)
    head = img[top : top + max(int((y2 - y1) * HEAD_FRACTION), 1), max(x1, 0) : min(x2, w)]
    if head.shape[0] <= 20 or head.shape[1] <= 20:
        return None, 0.0
    faces = find_faces(head)
    if not len(faces):
        return None, 0.0
    face = max(faces, key=lambda f: f[2] * f[3])
    return identify(embed(head, face))


# ----------------------------------------------------------------------------- API
app = FastAPI(title="GateKeeper AI")


class EnrollIn(BaseModel):
    name: str
    image: str


class AnalyzeIn(BaseModel):
    image: str
    zone: list[list[float]]
    # Which point of a person's box is tested against the zone:
    #   "feet"   bottom-centre of the box
    #   "center" middle of the box
    #   "auto"   feet, unless the box touches the bottom of the frame (feet cut off), then centre
    anchor: str = "auto"


@app.get("/api/health")
def health():
    return {"ok": True, "people": len(enrolled), "match_threshold": COSINE_MATCH}


@app.get("/api/people")
def people():
    return {"people": sorted(enrolled)}


@app.post("/api/enroll")
def enroll(body: EnrollIn):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Enter a name or ID.")
    if len(name) > MAX_NAME_LEN:
        raise HTTPException(400, f"Keep the name under {MAX_NAME_LEN} characters.")
    img = decode(body.image)
    faces = find_faces(img)
    if len(faces) == 0:
        raise HTTPException(422, "No face found. Face the camera in good light and try again.")
    face = max(faces, key=lambda f: f[2] * f[3])  # largest face
    with lock:
        enrolled[name] = embed(img, face)
        identities.clear()  # cached identities may no longer be correct
        save_faces()
    return {"ok": True, "name": name}


@app.delete("/api/people/{name}")
def remove(name: str):
    with lock:
        enrolled.pop(name, None)
        identities.clear()
        save_faces()
    return {"ok": True}


@app.post("/api/reset")
def reset():
    """Forget tracked people. The page calls this when monitoring starts."""
    with lock:
        reset_tracking()
    return {"ok": True}


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    started = time.perf_counter()
    if not 3 <= len(body.zone) <= MAX_ZONE_POINTS:
        raise HTTPException(400, f"The zone needs between 3 and {MAX_ZONE_POINTS} corners.")
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in body.zone):
        raise HTTPException(400, "Each zone corner must be an [x, y] pair.")
    if body.anchor not in ("auto", "feet", "center"):
        raise HTTPException(400, 'anchor must be "auto", "feet" or "center".')
    img = decode(body.image)
    img_h = img.shape[0]
    poly = np.array(body.zone, dtype=np.float32).reshape(-1, 1, 2)
    now = time.monotonic()
    out = []

    with lock:
        for tid in [t for t, v in identities.items() if now - v["seen"] > IDENTITY_TTL]:
            del identities[tid]

        result = yolo.track(
            img,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=PERSON_CONF,
            verbose=False,
        )[0]
        boxes = result.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cx = (x1 + x2) / 2
            feet_cut_off = y2 >= img_h - 4  # box touches the bottom edge of the frame
            if body.anchor == "center" or (body.anchor == "auto" and feet_cut_off):
                point = (cx, (y1 + y2) / 2)
            else:
                point = (cx, float(y2))
            inside = cv2.pointPolygonTest(poly, point, False) >= 0
            name, score = None, 0.0

            # Face recognition only runs for people inside the zone, and only until they are
            # recognised. After that the tracker id carries the identity, which saves compute.
            if inside and enrolled:
                cached = identities.get(tid) if tid is not None else None
                if cached:
                    name, score = cached["name"], cached["score"]
                    cached["seen"] = now
                else:
                    name, score = recognise_person(img, (x1, y1, x2, y2))
                    if name and tid is not None:
                        identities[tid] = {"name": name, "score": score, "seen": now}

            out.append(
                {
                    "track_id": tid,
                    "box": [x1, y1, x2, y2],
                    "point": [round(point[0]), round(point[1])],
                    "inside": bool(inside),
                    "name": name,
                    "score": round(float(score), 3),
                }
            )

    return {"detections": out, "ms": round((time.perf_counter() - started) * 1000)}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


load_faces()
# Warm up so the first live request is not slow.
yolo(np.zeros((480, 800, 3), np.uint8), verbose=False)
reset_tracking()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
