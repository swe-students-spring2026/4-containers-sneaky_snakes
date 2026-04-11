"""Real-time object detection client using YOLOv8 and OpenCV."""

import cv2
from ultralytics import YOLO


def get_camera(camera_index=0):
    """Open and return a VideoCapture object."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
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


def main():
    """Main loop: capture frames, detect objects, display results."""
    cap = get_camera()
    model = load_model()
    try:
        while True:
            frame = capture_frame(cap)
            if frame is None:
                break
            detections = detect_objects(model, frame)
            annotated = annotate_frame(frame, detections)
            cv2.imshow("ML Client - Detections", annotated)
            if detections:
                print(f"Detected: {[d['label'] for d in detections]}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
