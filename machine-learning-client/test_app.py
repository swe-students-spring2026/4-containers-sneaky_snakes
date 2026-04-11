import base64
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# Helpers 
def make_frame(h=480, w=640):
    """Return a dummy BGR frame (numpy array)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# Testing the camera functionality 
class TestGetCamera:
    @patch("cv2.VideoCapture")
    def test_opens_default_source(self, mock_vc):
        from app.main import get_camera
        mock_vc.return_value.isOpened.return_value = True
        cap = get_camera("0")
        mock_vc.assert_called_once_with(0)
        assert cap is not None

    @patch("cv2.VideoCapture")
    def test_raises_if_not_opened(self, mock_vc):
        from app.main import get_camera
        mock_vc.return_value.isOpened.return_value = False
        with pytest.raises(RuntimeError, match="Cannot open video source"):
            get_camera("0")

    @patch("cv2.VideoCapture")
    def test_accepts_file_path(self, mock_vc):
        from app.main import get_camera
        mock_vc.return_value.isOpened.return_value = True
        get_camera("video.mp4")
        mock_vc.assert_called_once_with("video.mp4")
        
        
        
# testing capture frame 

class TestCaptureFrame:
    def test_returns_frame_on_success(self):
        from app.main import capture_frame
        cap = MagicMock()
        frame = make_frame()
        cap.read.return_value = (True, frame)
        result = capture_frame(cap)
        assert result is frame

    def test_returns_none_on_failure(self):
        from app.main import capture_frame
        cap = MagicMock()
        cap.read.return_value = (False, None)
        assert capture_frame(cap) is None
        
# testing object detection 

class TestDetectObjects:
    def _make_mock_model(self, conf=0.9, cls_id=0, xyxy=(10, 20, 100, 200)):
        model = MagicMock()
        model.names = {0: "person"}

        box = MagicMock()
        box.conf = [conf]
        box.cls = [cls_id]
        xyxy_tensor = MagicMock()
        xyxy_tensor.tolist.return_value = list(xyxy)
        box.xyxy = [xyxy_tensor]

        result = MagicMock()
        result.boxes = [box]
        model.return_value = [result]
        return model

    def test_returns_detection_above_threshold(self):
        from app.main import detect_objects
        model = self._make_mock_model(conf=0.9)
        frame = make_frame()
        dets = detect_objects(model, frame, confidence_threshold=0.5)
        assert len(dets) == 1
        assert dets[0]["label"] == "person"
        assert dets[0]["confidence"] == 0.9

    def test_filters_below_threshold(self):
        from app.main import detect_objects
        model = self._make_mock_model(conf=0.3)
        frame = make_frame()
        dets = detect_objects(model, frame, confidence_threshold=0.5)
        assert dets == []

    def test_bbox_is_list_of_floats(self):
        from app.main import detect_objects
        model = self._make_mock_model(conf=0.8, xyxy=(10.5, 20.1, 100.9, 200.3))
        frame = make_frame()
        dets = detect_objects(model, frame)
        assert isinstance(dets[0]["bbox"], list)
        assert len(dets[0]["bbox"]) == 4

    def test_empty_results(self):
        from app.main import detect_objects
        model = MagicMock()
        result = MagicMock()
        result.boxes = []
        model.return_value = [result]
        dets = detect_objects(model, make_frame())
        assert dets == []