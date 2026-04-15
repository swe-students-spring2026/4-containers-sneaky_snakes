import base64
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import cv2
import app.main as main_module
from app.main import app as flask_app


@pytest.fixture(autouse=True)
def reset_globals():
    main_module._model = None
    main_module._db = None
    yield


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_data_url(frame=None):
    if frame is None:
        frame = make_frame()
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"


# decode_base64_image
class TestDecodeBase64Image:
    def test_decodes_valid_image(self):
        from app.main import decode_base64_image
        frame = decode_base64_image(make_data_url())
        assert frame is not None
        assert frame.shape[2] == 3

    def test_raises_on_empty(self):
        from app.main import decode_base64_image
        with pytest.raises(ValueError, match="Missing image payload"):
            decode_base64_image("")

    def test_resizes_if_too_wide(self):
        from app.main import decode_base64_image
        big = make_frame(480, 1920)
        result = decode_base64_image(make_data_url(big))
        assert result.shape[1] <= 960

    def test_decodes_without_data_prefix(self):
        from app.main import decode_base64_image
        _, buf = cv2.imencode(".jpg", make_frame())
        b64 = base64.b64encode(buf).decode()
        frame = decode_base64_image(b64)
        assert frame is not None


# detect_objects
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
        dets = detect_objects(self._make_mock_model(conf=0.9), make_frame(), confidence_threshold=0.5)
        assert len(dets) == 1
        assert dets[0]["label"] == "person"
        assert dets[0]["confidence"] == 0.9

    def test_filters_below_threshold(self):
        from app.main import detect_objects
        dets = detect_objects(self._make_mock_model(conf=0.3), make_frame(), confidence_threshold=0.5)
        assert dets == []

    def test_bbox_is_list_of_floats(self):
        from app.main import detect_objects
        dets = detect_objects(self._make_mock_model(conf=0.8), make_frame())
        assert isinstance(dets[0]["bbox"], list)
        assert len(dets[0]["bbox"]) == 4

    def test_empty_boxes(self):
        from app.main import detect_objects
        model = MagicMock()
        result = MagicMock()
        result.boxes = []
        model.return_value = [result]
        assert detect_objects(model, make_frame()) == []


# encode_frame_thumbnail
class TestEncodeFrameThumbnail:
    def test_returns_valid_base64(self):
        from app.main import encode_frame_thumbnail
        result = encode_frame_thumbnail(make_frame())
        assert isinstance(result, str)
        assert len(base64.b64decode(result)) > 0

    def test_respects_max_width(self):
        from app.main import encode_frame_thumbnail
        result = encode_frame_thumbnail(make_frame(), max_width=160)
        assert isinstance(result, str)


# save_detection_event
class TestSaveDetectionEvent:
    def test_inserts_and_returns_id(self):
        from app.main import save_detection_event
        db = MagicMock()
        db["detections"].insert_one.return_value.inserted_id = "abc123"
        result = save_detection_event(db, [], make_frame(), "test")
        assert result == "abc123"

    def test_doc_contains_num_objects(self):
        from app.main import save_detection_event
        db = MagicMock()
        db["detections"].insert_one.return_value.inserted_id = "x"
        dets = [{"label": "cat", "confidence": 0.8, "bbox": [0, 0, 1, 1]}] * 3
        save_detection_event(db, dets, make_frame(), "test")
        doc = db["detections"].insert_one.call_args[0][0]
        assert doc["num_objects"] == 3

    def test_doc_contains_image(self):
        from app.main import save_detection_event
        db = MagicMock()
        db["detections"].insert_one.return_value.inserted_id = "y"
        save_detection_event(db, [], make_frame(), "test")
        doc = db["detections"].insert_one.call_args[0][0]
        assert "image" in doc

    def test_doc_contains_source(self):
        from app.main import save_detection_event
        db = MagicMock()
        db["detections"].insert_one.return_value.inserted_id = "z"
        save_detection_event(db, [], make_frame(), "webcam")
        doc = db["detections"].insert_one.call_args[0][0]
        assert doc["source"] == "webcam"


# health route
class TestHealthRoute:
    @patch("app.main.get_model")
    @patch("app.main.get_db")
    def test_health_ok(self, mock_db, mock_model, client):
        mock_db.return_value.command.return_value = {"ok": 1}
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    @patch("app.main.get_db", side_effect=Exception("no db"))
    def test_health_fail(self, mock_db, client):
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.get_json()["ok"] is False


# detect route
class TestDetectRoute:
    @patch("app.main.save_detection_event", return_value="fake_id")
    @patch("app.main.detect_objects", return_value=[{"label": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1]}])
    @patch("app.main.get_model")
    @patch("app.main.get_db")
    @patch("app.main.decode_base64_image")
    def test_detect_returns_detections(self, mock_decode, mock_db, mock_model, mock_det, mock_save, client):
        mock_decode.return_value = make_frame()
        resp = client.post("/detect", json={"image": "data:image/jpeg;base64,abc", "source": "test"})
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 1

    @patch("app.main.decode_base64_image", side_effect=ValueError("bad image"))
    def test_detect_bad_image(self, mock_decode, client):
        resp = client.post("/detect", json={"image": "bad"})
        assert resp.status_code == 400

    @patch("app.main.save_detection_event", return_value=None)
    @patch("app.main.detect_objects", return_value=[])
    @patch("app.main.get_model")
    @patch("app.main.get_db")
    @patch("app.main.decode_base64_image")
    def test_detect_no_save_when_empty(self, mock_decode, mock_db, mock_model, mock_det, mock_save, client):
        mock_decode.return_value = make_frame()
        resp = client.post("/detect", json={"image": "data:image/jpeg;base64,abc"})
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 0


# get model
class TestGetModel:
    @patch("app.main.YOLO")
    def test_loads_model_once(self, mock_yolo):
        main_module._model = None
        main_module.get_model()
        main_module.get_model()
        mock_yolo.assert_called_once()
        
# connect_to_db
class TestConnectToDb:
    @patch("app.main.time.sleep")
    @patch("app.main.pymongo.MongoClient")
    def test_connects_successfully(self, mock_client, mock_sleep):
        mock_client.return_value.admin.command.return_value = {"ok": 1}
        from app.main import connect_to_db
        db = connect_to_db()
        assert db is not None

   


# get_db
class TestGetDb:
    @patch("app.main.connect_to_db")
    def test_returns_db_and_caches(self, mock_connect):
        mock_connect.return_value = MagicMock()
        main_module._db = None
        main_module.get_db()
        main_module.get_db()
        mock_connect.assert_called_once()


# decode_base64_image invalid
class TestDecodeBase64ImageInvalid:
    def test_raises_on_bad_bytes(self):
        from app.main import decode_base64_image
        bad_b64 = base64.b64encode(b"not an image").decode()
        with pytest.raises(ValueError, match="Could not decode image"):
            decode_base64_image(bad_b64)