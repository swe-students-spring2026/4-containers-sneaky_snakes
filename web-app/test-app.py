# Tests for the Flask web app.
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def fake_doc(label="person"):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "browser-camera",
        "num_objects": 1,
        "detections": [{"label": label, "confidence": 0.88, "bbox": [0, 0, 50, 80]}],
    }


def fake_stats():
    return {
        "total_snapshots": 3,
        "unique_labels": 2,
        "most_common": [{"label": "person", "count": 2}, {"label": "dog", "count": 1}],
    }


#  get_recent_detections 

class TestGetRecentDetections:
    @patch("app.get_collection")
    def test_returns_a_list(self, mock_col):
        from app import get_recent_detections
        mock_col.return_value.find.return_value.sort.return_value.limit.return_value = [fake_doc()]
        result = get_recent_detections()
        assert isinstance(result, list)

    @patch("app.get_collection")
    def test_limit_defaults_to_20(self, mock_col):
        from app import get_recent_detections
        mock_col.return_value.find.return_value.sort.return_value.limit.return_value = []
        get_recent_detections()
        mock_col.return_value.find.return_value.sort.return_value.limit.assert_called_with(20)

    @patch("app.get_collection")
    def test_custom_limit_works(self, mock_col):
        from app import get_recent_detections
        mock_col.return_value.find.return_value.sort.return_value.limit.return_value = []
        get_recent_detections(limit=5)
        mock_col.return_value.find.return_value.sort.return_value.limit.assert_called_with(5)

    @patch("app.get_collection")
    def test_sorted_by_timestamp_desc(self, mock_col):
        from app import get_recent_detections
        mock_col.return_value.find.return_value.sort.return_value.limit.return_value = []
        get_recent_detections()
        mock_col.return_value.find.return_value.sort.assert_called_with("timestamp", -1)


#  get_stats

class TestGetStats:
    @patch("app.get_collection")
    def test_has_right_keys(self, mock_col):
        from app import get_stats
        mock_col.return_value.count_documents.return_value = 0
        mock_col.return_value.distinct.return_value = []
        mock_col.return_value.aggregate.return_value = []
        result = get_stats()
        assert "total_snapshots" in result
        assert "unique_labels" in result
        assert "most_common" in result

    @patch("app.get_collection")
    def test_total_snapshots(self, mock_col):
        from app import get_stats
        mock_col.return_value.count_documents.return_value = 7
        mock_col.return_value.distinct.return_value = []
        mock_col.return_value.aggregate.return_value = []
        assert get_stats()["total_snapshots"] == 7

    @patch("app.get_collection")
    def test_unique_label_count(self, mock_col):
        from app import get_stats
        mock_col.return_value.count_documents.return_value = 0
        mock_col.return_value.distinct.return_value = ["person", "cat", "bottle"]
        mock_col.return_value.aggregate.return_value = []
        assert get_stats()["unique_labels"] == 3

    @patch("app.get_collection")
    def test_most_common_label_and_count(self, mock_col):
        from app import get_stats
        mock_col.return_value.count_documents.return_value = 0
        mock_col.return_value.distinct.return_value = []
        mock_col.return_value.aggregate.return_value = [{"_id": "person", "count": 9}]
        mc = get_stats()["most_common"]
        assert mc[0]["label"] == "person"
        assert mc[0]["count"] == 9

    @patch("app.get_collection")
    def test_empty_db(self, mock_col):
        from app import get_stats
        mock_col.return_value.count_documents.return_value = 0
        mock_col.return_value.distinct.return_value = []
        mock_col.return_value.aggregate.return_value = []
        stats = get_stats()
        assert stats["total_snapshots"] == 0
        assert stats["most_common"] == []


#  index route 

class TestIndex:
    @patch("app.get_stats")
    @patch("app.get_recent_detections")
    def test_200_ok(self, mock_dets, mock_stats, client):
        mock_dets.return_value = []
        mock_stats.return_value = fake_stats()
        assert client.get("/").status_code == 200

    @patch("app.get_stats")
    @patch("app.get_recent_detections")
    def test_renders_html(self, mock_dets, mock_stats, client):
        mock_dets.return_value = [fake_doc()]
        mock_stats.return_value = fake_stats()
        resp = client.get("/")
        assert b"<" in resp.data  # it's html

    @patch("app.get_stats")
    @patch("app.get_recent_detections")
    def test_shows_detection_label(self, mock_dets, mock_stats, client):
        mock_dets.return_value = [fake_doc("bicycle")]
        mock_stats.return_value = fake_stats()
        resp = client.get("/")
        assert b"bicycle" in resp.data

    @patch("app.get_stats", side_effect=Exception("mongo down"))
    @patch("app.get_recent_detections", side_effect=Exception("mongo down"))
    def test_db_error_still_loads(self, mock_dets, mock_stats, client):
        # page should not crash even if mongo is unreachable
        resp = client.get("/")
        assert resp.status_code == 200


#  /api/detections 

class TestApiDetections:
    @patch("app.get_recent_detections")
    def test_returns_200(self, mock_dets, client):
        mock_dets.return_value = [fake_doc()]
        assert client.get("/api/detections").status_code == 200

    @patch("app.get_recent_detections")
    def test_returns_json(self, mock_dets, client):
        mock_dets.return_value = [fake_doc()]
        resp = client.get("/api/detections")
        assert resp.content_type == "application/json"
        assert isinstance(resp.get_json(), list)

    @patch("app.get_recent_detections")
    def test_empty_returns_empty_list(self, mock_dets, client):
        mock_dets.return_value = []
        assert client.get("/api/detections").get_json() == []

    @patch("app.get_recent_detections", side_effect=Exception("db error"))
    def test_503_on_db_failure(self, mock_dets, client):
        assert client.get("/api/detections").status_code == 503


# ---- /api/stats ----

class TestApiStats:
    @patch("app.get_stats")
    def test_returns_200(self, mock_stats, client):
        mock_stats.return_value = fake_stats()
        assert client.get("/api/stats").status_code == 200

    @patch("app.get_stats")
    def test_json_has_expected_keys(self, mock_stats, client):
        mock_stats.return_value = fake_stats()
        data = client.get("/api/stats").get_json()
        assert "total_snapshots" in data
        assert "most_common" in data

    @patch("app.get_stats")
    def test_most_common_is_list(self, mock_stats, client):
        mock_stats.return_value = fake_stats()
        data = client.get("/api/stats").get_json()
        assert isinstance(data["most_common"], list)

    @patch("app.get_stats", side_effect=Exception("db error"))
    def test_503_on_db_failure(self, mock_stats, client):
        assert client.get("/api/stats").status_code == 503