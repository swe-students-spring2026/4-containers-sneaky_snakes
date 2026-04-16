[![lint-free](https://github.com/swe-students-spring2026/4-containers-sneaky_snakes/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/swe-students-spring2026/4-containers-sneaky_snakes/actions/workflows/lint.yml)
[![log github events](https://github.com/swe-students-spring2026/4-containers-sneaky_snakes/actions/workflows/event-logger.yml/badge.svg)](https://github.com/swe-students-spring2026/4-containers-sneaky_snakes/actions/workflows/event-logger.yml)


# Sneaky Snakes Real-Time Object Detection

A containerized system that captures webcam frames, runs YOLOv8 object detection, and displays results on a live web dashboard. It serves as a study monitor system that aims to create a study environment by warning the user about possible distraction when a phone is detected.

## Team

- Rohan Malhotra — [@Rohanmalhotra0](https://github.com/Rohanmalhotra0)
- Ryan Lu — [@CHEology](https://github.com/CHEology)
- James Huang — [@JamesHuang2004](https://github.com/JamesHuang2004)
- Jai — [@hyperjasm](https://github.com/hyperjasm)
- Yusef — [@YusefMoustafa](https://github.com/YusefMoustafa)

## Project Overview

This project is composed of three containerized services:

1. **machine-learning-client/**
   - receives webcam snapshots
   - runs YOLOv8 object detection
   - stores detection results in MongoDB

2. **web-app/**
   - serves the Flask dashboard
   - shows live detections, history, and summary statistics
   - reads stored detection events from MongoDB

3. **mongodb**
   - stores all detection records for later display and analysis

Together, these services demonstrate communication among containers using Docker Compose.

## Tech Stack

- **ML Client:** Python, OpenCV, YOLOv8 (ultralytics), pymongo
- **Web App:** Python, Flask, pymongo, Jinja2
- **Database:** MongoDB (Docker)

## Architecture and Data Flow

```text
Browser → ML Client → MongoDB → Web App → Browser
```

- The browser captures webcam frames and sends them to the ML client over HTTP.
- The ML client runs object detection on the image.
- The ML client stores the detection result in MongoDB.
- The web app reads stored results from MongoDB.
- The web app displays detections, history, and statistics in the browser.

### Container Communication

- **Browser → ML Client:** HTTP on port `8000`
- **ML Client → MongoDB:** MongoDB protocol on port `27017`
- **Web App → MongoDB:** MongoDB protocol on port `27017`
- **Browser → Web App:** HTTP on port `5001`

Inside Docker Compose, services communicate using service names rather than `localhost`:

- `mongodb:27017`
- `ml-client:8000`

## Repository Structure

```text
.
├── .github/workflows/
├── machine-learning-client/
├── web-app/
├── docker-compose.yml
└── README.md
```

## Environment Variables

This project uses environment variables to configure the two application containers.

### Web App

```env
MONGO_URI=mongodb://mongodb:27017
DB_NAME=ml_detections
PORT=5001
ML_API_URL=http://localhost:8000/detect
```

### Machine Learning Client

```env
MONGO_URI=mongodb://mongodb:27017
MONGO_DBNAME=ml_detections
PORT=8000
CONFIDENCE_THRESHOLD=0.4
MAX_IMAGE_WIDTH=960
SAVE_EMPTY=0
```

### Notes on Environment Variables

- `MONGO_URI` specifies how each service connects to MongoDB.
- `DB_NAME` / `MONGO_DBNAME` specify the database name used by the application.
- `PORT` determines the service port exposed by each application container.
- `ML_API_URL` is the endpoint used by the browser-facing app to send frames for detection.
- `CONFIDENCE_THRESHOLD` controls the minimum confidence required for a detection.
- `MAX_IMAGE_WIDTH` limits image size before sending to the ML service.
- `SAVE_EMPTY` controls whether frames with no detections should still be stored.

## Running the Full Application

Make sure Docker is running, then from the project root run:

```bash
docker compose up --build
```

This starts all three containers:

- `mongodb`
- `ml-client`
- `web-app`

After startup:

- Web dashboard: `http://localhost:5001`
- ML API: `http://localhost:8000/detect`
- MongoDB: `localhost:27017`

To stop the application:

```bash
docker compose down
```

## Running Individual Services for Development

### Start MongoDB only

```bash
docker compose up -d mongodb
```

### Run the machine-learning client locally

```bash
cd machine-learning-client
pip install -r requirements.txt
python -m app.main
```

### Run the web app locally

```bash
cd web-app
pip install -r requirements.txt
python app.py
```

## Database Contents

MongoDB stores detection events, including fields such as:

- `timestamp`
- `source`
- `num_objects`
- `detections`
- `image`

Each entry in `detections` may include:

- `label`
- `confidence`
- `bbox`

The `image` field is stored as a base64-encoded string so that detections can be reviewed later in the dashboard.


## Testing and CI

This repository includes GitHub Actions workflows for:

- linting
- logging GitHub events
- CI for the machine-learning client
- CI for the web app

The badges at the top of this README show the current status of those workflows.


