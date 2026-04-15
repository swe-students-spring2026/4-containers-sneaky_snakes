# Sneaky Snakes Real-Time Object Detection

A containerized system that captures webcam frames, runs YOLOv8 object detection, and displays results on a live web dashboard.

## How it works

- **ML Client** — captures frames, runs YOLO detection, saves results to MongoDB
- **Web App** — Flask dashboard showing detected objects, history, and stats
- **Database** — MongoDB stores all detection events

## Team

- Rohan Malhotra — [@Rohanmalhotra0](https://github.com/Rohanmalhotra0)
- Ryan Lu — [@CHEology](https://github.com/CHEology)
- James Huang — [@JamesHuang2004](https://github.com/JamesHuang2004)
- Jai — [@hyperjasm](https://github.com/hyperjasm)

## Tech Stack

- **ML Client**: Python, OpenCV, YOLOv8 (ultralytics), pymongo
- **Web App**: Python, Flask, pymongo, Jinja2
- **Database**: MongoDB (Docker)

## Running the app

```bash
docker-compose up --build
```

Make sure Docker is running. The web dashboard will be available at `http://localhost:5001`.

## Environment Variables

Copy `.env.example` to `.env` in both `machine-learning-client/` and `web-app/` and fill in the values.