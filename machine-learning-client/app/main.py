"""Object detection service that stores detections in MongoDB."""
from __future__ import annotations

import base64
import os
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import pymongo
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
DB_NAME = os.getenv("MONGO_DBNAME", "ml_detections")
PORT = int(os.getenv("PORT", "8000"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "960"))
SAVE_EMPTY = os.getenv("SAVE_EMPTY", "0").lower() in {"1", "true", "yes"}

app = Flask(__name__)
CORS(app)
_model: YOLO | None = None
_db = None


def connect_to_db(max_attempts: int = 30, delay_seconds: int = 2):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
            print(f"Connected to MongoDB on attempt {attempt}.")
            return client[DB_NAME]
        except pymongo.errors.PyMongoError as exc:
            last_error = exc
            print(f"MongoDB not ready yet (attempt {attempt}/{max_attempts}): {exc}")
            time.sleep(delay_seconds)
    raise RuntimeError(f"Could not connect to MongoDB: {last_error}") from last_error


def get_db():
    global _db
    if _db is None:
        _db = connect_to_db()
    return _db


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def decode_base64_image(data_url: str) -> np.ndarray:
    if not data_url:
        raise ValueError("Missing image payload.")
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(payload)
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    h, w = img.shape[:2]
    if w > MAX_IMAGE_WIDTH:
        new_h = int(h * MAX_IMAGE_WIDTH / w)
        img = cv2.resize(img, (MAX_IMAGE_WIDTH, new_h))
    return img


def detect_objects(model: YOLO, frame: np.ndarray, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> list[dict[str, Any]]:
    results = model(frame, verbose=False)
    detections: list[dict[str, Any]] = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            cls_id = int(box.cls[0])
            detections.append({
                "label": str(model.names[cls_id]),
                "confidence": round(conf, 3),
                "bbox": [round(float(c), 1) for c in box.xyxy[0].tolist()],
            })
    return detections


def encode_frame_thumbnail(frame: np.ndarray, max_width: int = 320) -> str:
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("Could not encode frame.")
    return base64.b64encode(buf).decode("utf-8")


def save_detection_event(db, detections: list[dict[str, Any]], frame: np.ndarray, source: str) -> str:
    doc = {
        "timestamp": datetime.now(timezone.utc),
        "source": source,
        "num_objects": len(detections),
        "detections": detections,
        "image": encode_frame_thumbnail(frame),
    }
    result = db["detections"].insert_one(doc)
    return str(result.inserted_id)


@app.get('/health')
def health():
    try:
        get_db().command('ping')
        get_model()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@app.post('/detect')
def detect_route():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image')
    source = payload.get('source', 'browser-camera')
    try:
        frame = decode_base64_image(image_data)
        detections = detect_objects(get_model(), frame)
        inserted_id = None
        if detections or SAVE_EMPTY:
            inserted_id = save_detection_event(get_db(), detections, frame, source)
        return jsonify({
            'saved': bool(inserted_id),
            'inserted_id': inserted_id,
            'count': len(detections),
            'detections': detections,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


if __name__ == '__main__':
    get_db()
    get_model()
    app.run(host='0.0.0.0', port=PORT, debug=False)
