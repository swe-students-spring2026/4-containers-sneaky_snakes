"""Flask dashboard that reads ML detections from MongoDB."""

from __future__ import annotations
import os
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template
from pymongo import MongoClient
from pymongo.errors import PyMongoError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
DB_NAME = os.environ.get("DB_NAME", "ml_detections")
COLLECTION_NAME = "detections"
PORT = int(os.environ.get("PORT", "5001"))
ML_API_URL = os.environ.get("ML_API_URL", "http://localhost:8000/detect")


def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client[DB_NAME][COLLECTION_NAME]


def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client[DB_NAME]


def get_recent_detections(limit: int = 20):
    docs = get_collection().find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return list(docs)


def get_stats():
    collection = get_collection()
    total = collection.count_documents({})
    unique_labels = collection.distinct("detections.label")
    most_common = list(
        collection.aggregate(
            [
                {"$unwind": "$detections"},
                {"$group": {"_id": "$detections.label", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5},
            ]
        )
    )
    return {
        "total_snapshots": total,
        "unique_labels": len(unique_labels),
        "most_common": [
            {"label": item["_id"], "count": item["count"]} for item in most_common
        ],
    }


def get_focus_stats():
    db = get_db()

    state = db["focus_state"].find_one({"_id": "current"}) or {}
    nudge_active = bool(state.get("nudge_active", False))
    sitting_since = state.get("sitting_since")

    if sitting_since is not None:
        now = datetime.now(timezone.utc)
        sitting_since = (
            sitting_since.replace(tzinfo=timezone.utc)
            if sitting_since.tzinfo is None
            else sitting_since
        )
        session_duration_seconds = max(0, round((now - sitting_since).total_seconds()))
    else:
        session_duration_seconds = 0

    today_midnight = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    phone_pickups_today = db["focus_events"].count_documents(
        {"event_type": "phone_pickup", "timestamp": {"$gte": today_midnight}}
    )

    return {
        "phone_pickups_today": phone_pickups_today,
        "session_duration_seconds": session_duration_seconds,
        "nudge_active": nudge_active,
    }


@app.route("/")
def index():
    try:
        detections = get_recent_detections()
        stats = get_stats()
        focus_stats = get_focus_stats()
        db_error = False
    except PyMongoError:
        detections = []
        stats = {"total_snapshots": 0, "unique_labels": 0, "most_common": []}
        focus_stats = {
            "phone_pickups_today": 0,
            "session_duration_seconds": 0,
            "nudge_active": False,
        }
        db_error = True
    return render_template(
        "index.html",
        detections=detections,
        stats=stats,
        focus_stats=focus_stats,
        db_error=db_error,
        ml_api_url=ML_API_URL,
    )


@app.route("/api/detections")
def api_detections():
    try:
        return jsonify(get_recent_detections())
    except PyMongoError:
        return jsonify({"error": "Could not reach database"}), 503


@app.route("/api/stats")
def api_stats():
    try:
        return jsonify(get_stats())
    except PyMongoError:
        return jsonify({"error": "Could not reach database"}), 503


@app.route("/api/focus")
def api_focus():
    try:
        return jsonify(get_focus_stats())
    except PyMongoError:
        return jsonify({"error": "Could not reach database"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
