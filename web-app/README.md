# Web App

Flask dashboard for displaying real-time object detection results from MongoDB.

## Running Locally

```bash
pip install -r requirements.txt
MONGO_URI=mongodb://localhost:27017 PORT=5001 python app.py
```

> Port 5000 is used by AirPlay Receiver on macOS. Use `PORT=5001` to avoid the conflict.

Open [http://localhost:5001].

## Running with Docker

```bash
docker build -t web-app .
docker run -p 5000:5000 -e MONGO_URI=mongodb://host.docker.internal:27017 web-app
```

Open [http://localhost:5000].

## Environment Variables



`MONGO_URI`  `mongodb://mongodb:27017`  MongoDB connection string  
`DB_NAME`    `detection_db`             Database name              
`PORT`       `5000`                     Port the server listens on 

## MongoDB Document Format

The ML client must store detection documents in this format:

```json
{
  "timestamp": "2026-04-11T12:00:00",
  "detections": [
    { "label": "person", "confidence": 0.95 },
    { "label": "chair", "confidence": 0.87 }
  ]
}
```

Collection name: `detections`
Database name: `detection_db` (or set via `DB_NAME`)
