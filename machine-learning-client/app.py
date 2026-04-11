"""Real-time object detection client using YOLOv8 and OpenCV."""

import os
import time
import base64
from datetime import datetime, timezone

import cv2
import pymongo
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


def get_camera(source=None):
    """Open camera (index) or video file. Reads VIDEO_SOURCE env var as fallback."""
    if source is None:
        source = os.getenv("VIDEO_SOURCE", "0")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def capture_frame(cap):
    """Read a single frame from the camera. Returns the frame or None."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def load_model(model_name="yolov8n.pt"):
    """Load and return a YOLOv8 model (weights auto-download on first call)."""
    return YOLO(model_name)


def detect_objects(model, frame, confidence_threshold=0.5):
    """Run YOLO inference on a frame. Returns list of detection dicts."""
    results = model(frame, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= confidence_threshold:
                cls_id = int(box.cls[0])
                detections.append(
                    {
                        "label": model.names[cls_id],
                        "confidence": round(conf, 3),
                        "bbox": [round(c, 1) for c in box.xyxy[0].tolist()],
                    }
                )
    return detections


def annotate_frame(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    return frame


def get_db():
    """Connect to MongoDB and return the database object."""
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DBNAME", "ml_detections")
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
    return client[db_name]


def encode_frame_thumbnail(frame, max_width=320):
    """Resize frame and encode as base64 JPEG for storage."""
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (max_width, int(h * max_width / w)))
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("utf-8")


def save_detections(db, detections, frame=None):
    """Save a detection event to MongoDB. Returns inserted doc ID."""
    doc = {
        "timestamp": datetime.now(timezone.utc),
        "detections": detections,
        "num_objects": len(detections),
    }
    if frame is not None:
        doc["image"] = encode_frame_thumbnail(frame)
    return db["detections"].insert_one(doc).inserted_id


def main():
    """Main loop: capture frames, detect objects, save to DB, display."""
    cap = get_camera()
    model = load_model()

    # Try connecting to MongoDB; proceed without it if unavailable
    db = None
    try:
        db = get_db()
        db.client.server_info()
        print("Connected to MongoDB.")
    except pymongo.errors.ServerSelectionTimeoutError:
        print("MongoDB not available — running without database.")
        db = None

    headless = os.getenv("HEADLESS", "").lower() in ("1", "true")
    last_save = 0
    try:
        while True:
            frame = capture_frame(cap)
            if frame is None:
                break
            detections = detect_objects(model, frame)

            if not headless:
                annotated = annotate_frame(frame.copy(), detections)
                cv2.imshow("ML Client - Detections", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            if detections and db is not None and (now - last_save) >= 1.0:
                doc_id = save_detections(db, detections, frame)
                print(f"Saved {len(detections)} detections, id={doc_id}")
                last_save = now
            elif detections:
                print(f"Detected: {[d['label'] for d in detections]}")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
