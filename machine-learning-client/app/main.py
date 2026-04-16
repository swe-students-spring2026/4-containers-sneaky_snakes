"""Object detection service that stores detections in MongoDB."""

from __future__ import annotations

import base64
import os
import threading
import time
from datetime import date, datetime, timezone
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

FOCUS_LABELS = {"person", "cell phone"}
PERSON_TIMEOUT = 30  # seconds gap before a sitting session is considered ended
NUDGE_THRESHOLD = 45 * 60  # 45 minutes in seconds

app = Flask(__name__)
CORS(app)
_model: YOLO | None = None
_db = None

# --- Session state (protected by _state_lock) ---
_state_lock = threading.Lock()
_sitting_since: float | None = None
_last_person_ts: float | None = None
_phone_visible: bool = False
_nudge_sent: bool = False
_phone_pickups_today: int = 0
_pickups_date: date = date.today()


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


def detect_objects(
    model: YOLO, frame: np.ndarray, confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> list[dict[str, Any]]:
    results = model(frame, verbose=False)
    detections: list[dict[str, Any]] = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            cls_id = int(box.cls[0])
            label = str(model.names[cls_id])
            if label not in FOCUS_LABELS:
                continue
            detections.append(
                {
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [round(float(c), 1) for c in box.xyxy[0].tolist()],
                }
            )
    return detections


def encode_frame_thumbnail(frame: np.ndarray, max_width: int = 320) -> str:
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("Could not encode frame.")
    return base64.b64encode(buf).decode("utf-8")


def save_detection_event(
    db, detections: list[dict[str, Any]], frame: np.ndarray, source: str
) -> str:
    doc = {
        "timestamp": datetime.now(timezone.utc),
        "source": source,
        "num_objects": len(detections),
        "detections": detections,
        "image": encode_frame_thumbnail(frame),
    }
    result = db["detections"].insert_one(doc)
    return str(result.inserted_id)


def save_focus_event(db, event_type: str, **extra) -> None:
    """Insert one event document into the focus_events collection."""
    doc = {"event_type": event_type, "timestamp": datetime.now(timezone.utc), **extra}
    db["focus_events"].insert_one(doc)


def upsert_focus_state(
    db,
    sitting_since_ts: float | None,
    nudge_active: bool,
    phone_visible: bool,
) -> None:
    """Persist the current session state so the web app can read it."""
    sitting_since_dt = (
        datetime.fromtimestamp(sitting_since_ts, tz=timezone.utc)
        if sitting_since_ts is not None
        else None
    )
    db["focus_state"].replace_one(
        {"_id": "current"},
        {
            "_id": "current",
            "sitting_since": sitting_since_dt,
            "nudge_active": nudge_active,
            "phone_visible": phone_visible,
            "updated_at": datetime.now(timezone.utc),
        },
        upsert=True,
    )


def process_focus_state(db, detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Update session tracking state and persist to MongoDB."""
    global _sitting_since, _last_person_ts, _phone_visible
    global _nudge_sent, _phone_pickups_today, _pickups_date

    now = time.time()
    person_in_frame = any(d["label"] == "person" for d in detections)
    phone_in_frame = any(d["label"] == "cell phone" for d in detections)

    with _state_lock:
        today = date.today()
        if today != _pickups_date:
            _phone_pickups_today = 0
            _pickups_date = today

        if person_in_frame:
            if _sitting_since is None:
                _sitting_since = now
                _nudge_sent = False
            _last_person_ts = now
        elif _sitting_since is not None and _last_person_ts is not None:
            if now - _last_person_ts > PERSON_TIMEOUT:
                _sitting_since = None
                _nudge_sent = False

        if phone_in_frame and not _phone_visible:
            _phone_pickups_today += 1
            session_secs = (now - _sitting_since) if _sitting_since else 0.0
            save_focus_event(
                db,
                "phone_pickup",
                session_duration_seconds=round(session_secs),
            )
        _phone_visible = phone_in_frame

        if _sitting_since is not None:
            session_duration = now - _sitting_since
            if session_duration >= NUDGE_THRESHOLD and not _nudge_sent:
                save_focus_event(
                    db,
                    "stand_up_nudge",
                    session_duration_seconds=round(session_duration),
                )
                _nudge_sent = True

        nudge_active = _sitting_since is not None and _nudge_sent
        session_duration_seconds = (
            round(now - _sitting_since) if _sitting_since is not None else 0
        )
        sitting_since_snapshot = _sitting_since
        pickups_snapshot = _phone_pickups_today

    upsert_focus_state(db, sitting_since_snapshot, nudge_active, phone_in_frame)

    return {
        "sitting_since_ts": sitting_since_snapshot,
        "session_duration_seconds": session_duration_seconds,
        "nudge_active": nudge_active,
        "phone_pickups_today": pickups_snapshot,
    }


@app.get("/health")
def health():
    try:
        get_db().command("ping")
        get_model()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@app.post("/detect")
def detect_route():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    source = payload.get("source", "browser-camera")
    try:
        frame = decode_base64_image(image_data)
        detections = detect_objects(get_model(), frame)
        inserted_id = None
        if detections or SAVE_EMPTY:
            inserted_id = save_detection_event(get_db(), detections, frame, source)
        focus = process_focus_state(get_db(), detections)
        return jsonify(
            {
                "saved": bool(inserted_id),
                "inserted_id": inserted_id,
                "count": len(detections),
                "detections": detections,
                "focus": focus,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    get_db()
    get_model()
    app.run(host="0.0.0.0", port=PORT, debug=False)
