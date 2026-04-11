"""Real-time object detection client using YOLOv8 and OpenCV."""

import cv2


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


def main():
    """Main loop: capture frames and display preview."""
    cap = get_camera()
    try:
        while True:
            frame = capture_frame(cap)
            if frame is None:
                break
            cv2.imshow("ML Client - Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
