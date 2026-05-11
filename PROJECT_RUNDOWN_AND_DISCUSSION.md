# Project Rundown And Discussion Notes

## One-Paragraph Rundown

This project is a traffic video analytics system. It uses YOLOv3 for object detection, SORT for multi-object tracking, and a line-crossing rule to count vehicles in a video, webcam feed, or stream. The system counts only vehicle classes such as `car`, `motorbike`, `bus`, and `truck`, but it still tracks and logs all detected object classes. This makes the output explainable: the project does not only produce an annotated video, it also creates timestamped logs and per-track summaries that show what was detected, how long it stayed in the scene, whether it crossed the line, and whether it contributed to the vehicle count.

## Current Capabilities

- Processes uploaded videos, local video files, webcam feeds, and RTSP/HTTP streams.
- Detects objects using YOLOv3 Darknet weights through OpenCV DNN.
- Tracks objects across frames using SORT and stable track IDs.
- Counts vehicle-class tracks once when they cross a configured line.
- Logs all tracked objects, including non-counted classes, for auditability.
- Estimates speed from centroid movement, FPS, and configurable meters-per-pixel calibration.
- Creates timestamped run folders under `output/runs/YYYYMMDD_HHMMSS/`.
- Produces an annotated output video, frame-level CSV log, per-track summary CSV, and run metadata JSON.
- Provides a Gradio web app for upload-and-process deployment.
- Supports Docker deployment for the web app.
- Supports YOLOv11 through Ultralytics as an optional detector backend.
- Supports ByteTrack when using the Ultralytics detector.
- Generates ROI, lane, direction, heatmap, HTML report, and PDF report artifacts.
- Provides a REST API for upload-based background processing.

## Project Architecture

```text
Video Source
    |
    v
Frame Reader
    |
    v
YOLOv3 Detector
    |
    v
Non-Maximum Suppression
    |
    v
SORT Tracker
    |
    v
Track Label Matching
    |
    +--> Vehicle Line Counter
    +--> Speed Estimator
    +--> Annotated Video Writer
    +--> Tracking CSV Logger
    +--> Track Summary Writer
    +--> Run Metadata Writer
```

## Important Design Decisions

### Tracking And Counting Are Separated

The system tracks all detected classes, but only counts traffic-relevant vehicle classes. This is stronger than filtering everything early because the logs preserve evidence for false positives, pedestrians, traffic lights, or other non-counted objects.

### Each Run Has Its Own Folder

Every run gets a timestamped folder, such as:

```text
output/runs/20260511_143000/
```

This prevents old outputs from being overwritten and makes each experiment reproducible.

### Counting Is One-Time Per Track

Once a vehicle track crosses the line, that track is marked as counted. This reduces double counting when a bounding box jitters around the line.

### Logs Make The Result Explainable

The project can answer why the final count happened. If the result is wrong, the logs help identify whether the issue came from detection, tracking, line crossing, calibration, or class filtering.

## 5-Minute Discussion Notes

### 1. Problem

Manual traffic counting is slow, inconsistent, and difficult to audit. The goal is to automate vehicle counting from traffic video while keeping the output explainable.

### 2. Approach

The system combines YOLOv3 for detection and SORT for tracking. YOLO finds objects in each frame, and SORT assigns persistent IDs so the same object can be followed over time.

### 3. Counting Logic

The system defines a counting line. When a tracked object crosses that line, it is counted only if its stable class label is one of the vehicle classes: `car`, `motorbike`, `bus`, or `truck`.

### 4. Explainability

Instead of only producing a number, every tracked object is logged with timestamp, class, confidence, bounding box, center point, speed estimate, and line-crossing status. This allows us to review the result after processing.

### 5. Deployment

The project has a command-line version, a real-time display mode, and a Gradio web app where a user uploads a video and receives the processed video and logs.

### 6. Limitations And Next Step

The main limitation is that the detector is currently YOLOv3. The best next step is upgrading to YOLOv11 and comparing detection quality and processing speed.

## 10-Minute Discussion Notes

### 1. Motivation

Traffic monitoring is useful for congestion analysis, road planning, and safety. A practical system should not only count vehicles, but also preserve enough evidence to explain and debug the count.

### 2. Detection Layer

YOLOv3 detects COCO classes in each frame. It outputs bounding boxes, class probabilities, and confidence scores. Non-maximum suppression removes duplicate boxes before tracking.

### 3. Tracking Layer

SORT uses Kalman filtering and assignment matching to keep object identities stable across frames. The benefit is that the system can reason about object movement instead of treating every frame independently.

### 4. Counting Layer

The line-crossing logic checks the movement of each track center between consecutive frames. If the center path intersects the configured line, the object becomes a crossing event. The project counts the track only once.

### 5. Logging Layer

The frame-level log records every observation. The summary log compresses each track into first timestamp, last timestamp, duration, label, max confidence, and line-crossing status. This makes the count auditable.

### 6. Runtime Modes

The same pipeline supports offline video processing, real-time display from webcam or stream, and browser upload through Gradio. Each run produces its own timestamped output folder.

### 7. Speed Estimation

Speed is estimated from centroid displacement:

```text
speed = pixel_distance * meters_per_pixel * fps
```

This is suitable for a prototype, but a real deployment needs scene calibration or homography for accurate speed.

### 8. Strengths

- Combines detection, tracking, counting, logging, and deployment.
- Separates tracking from counting for better debugging.
- Preserves run artifacts in timestamped folders.
- Provides both visual and structured outputs.
- Can run from CLI, live display, web app, or Docker.

### 9. Limitations

- YOLOv3 is older and less accurate than modern YOLO versions.
- SORT can switch IDs during occlusion.
- Speed is approximate without calibration.
- The counting line is manually configured.
- CPU performance may not reach true camera FPS.

### 10. Best Next Step

Upgrade the detector to YOLOv11, keep the same tracking and logging architecture, then compare YOLOv3 vs YOLOv11 on the same input video.

## Architecture Features Introduced Or Ready To Extend

### Detector Abstraction

Implemented for YOLOv3 and Ultralytics YOLOv11. The same idea can later support ONNX or TensorRT.

### Configuration Files

Partially implemented through ROI and lane JSON files. Model, line, calibration, and tracker settings are exposed through CLI arguments.

### Run Manager

Implemented through timestamped run folders and `run_metadata.json`.

### Background Job Queue

Implemented in `api.py` as background processing jobs for uploaded videos.

### REST API

Implemented in `api.py` with endpoints such as:

```text
POST /runs
GET /runs/{id}
GET /runs/{id}/artifacts
```

This makes the system usable by other apps.

### Database Storage

Store run metadata, final counts, class summaries, and artifact paths in SQLite or PostgreSQL. Keep videos and CSVs as filesystem artifacts.

### Evaluation Module

Add a script that compares predicted counts against manually labeled ground truth, producing precision, recall, count error, and class-level metrics.

### Model Registry

Implemented in run metadata and `compare_models.py`.

## YOLO-Focused Features Added

### YOLOv11 Upgrade

Implemented as an optional backend:

```text
python main.py --source video.mp4 --detector ultralytics --model yolo11n.pt
```

### YOLO Model Comparison

Implemented through `compare_models.py`, which writes `comparison.csv` and `comparison.json`.

### ROI-Based YOLO Detection

Implemented through `--roi` and `--roi-file`.

### Confidence Review

Implemented through `low_confidence_review` in the frame log and report metrics.

### Vehicle Class Breakdown

Implemented as class observations in `run_metadata.json`, `report.html`, and `report.pdf`.

### YOLO Detection Heatmap

Implemented through `detection_heatmap.png` and `detection_heatmap_overlay.png`.

### Per-Lane Counting

Implemented through `--lanes-file`.

### Direction Detection

Implemented through `direction` columns in the frame and summary logs.

### License Plate Extension

Partially implemented as a second-stage hook with `--save-crops`. It saves counted vehicle crops for later plate detection or OCR.

## General Features To Add

- Dashboard with total count, class distribution, and traffic over time. This is intentionally halted for now.
- Exportable PDF report for each run.
- CSV charts for average speed and peak traffic intervals.
- Multi-camera support.
- Manual correction screen for reviewing bad detections.
- Calibration wizard for speed estimation.
- Alerting when traffic exceeds a threshold.
- Storage browser for previous timestamped runs.
- Authentication for the web app if deployed publicly.
- GPU acceleration support through CUDA, ONNX Runtime, or TensorRT.

## Suggested TA Demo Flow

1. Open the web app and upload the sample video.
2. Show annotated preview frames during processing.
3. Open the timestamped output folder.
4. Show `output_video.mp4`.
5. Show `tracking_log.csv` and explain that every tracked object is auditable.
6. Show `track_summary.csv` and explain per-object duration and crossing status.
7. Explain the limitation of YOLOv3 and propose YOLOv11 upgrade as the next step.
