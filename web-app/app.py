"""Flask web app for displaying object detection results from MongoDB."""

import os
from flask import Flask, render_template, jsonify
from pymongo import MongoClient
from pymongo.errors import PyMongoError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
DB_NAME = os.environ.get("DB_NAME", "detection_db")
COLLECTION_NAME = "detections"
PORT = int(os.environ.get("PORT", "5000"))

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def get_recent_detections(limit=20):
    """Return the most recent detection documents."""
    docs = collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return list(docs)


def get_stats():
    """Return summary statistics from the detections collection."""
    total = collection.count_documents({})

    pipeline = [
        {"$unwind": "$detections"},
        {"$group": {"_id": "$detections.label", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5},
    ]
    most_common = list(collection.aggregate(pipeline))
    unique_labels = collection.distinct("detections.label")

    return {
        "total_snapshots": total,
        "unique_labels": len(unique_labels),
        "most_common": [{"label": d["_id"], "count": d["count"]} for d in most_common],
    }


@app.route("/")
def index():
    """Render the main dashboard."""
    db_error = False
    detections = []
    stats = {"total_snapshots": 0, "unique_labels": 0, "most_common": []}
    try:
        detections = get_recent_detections()
        stats = get_stats()
    except PyMongoError:
        db_error = True
    return render_template(
        "index.html", detections=detections, stats=stats, db_error=db_error
    )


@app.route("/api/detections")
def api_detections():
    """Return recent detections as JSON."""
    try:
        return jsonify(get_recent_detections())
    except PyMongoError:
        return jsonify({"error": "Could not reach database"}), 503


@app.route("/api/stats")
def api_stats():
    """Return stats as JSON."""
    try:
        return jsonify(get_stats())
    except PyMongoError:
        return jsonify({"error": "Could not reach database"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
