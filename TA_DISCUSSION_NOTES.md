# TA Discussion Notes

## Project Positioning

This project is a traffic video analytics prototype. It combines object detection, multi-object tracking, rule-based line crossing, real-time display, and structured logging to produce a vehicle count that can be inspected during or after the run.

The core idea is not only to display a count on the video, but also to make the count explainable. Every tracked object is stored with its timestamp, track ID, class label, position, speed estimate, and line-crossing status.

## Refined Tracking And Counting Idea

The system separates tracking from counting:

- It tracks all detected COCO object classes by default.
- It counts only traffic-relevant vehicle classes: `car`, `motorbike`, `bus`, and `truck`.
- It writes all tracked objects into CSV logs, even when they are not counted.

This is stronger than filtering everything early because it gives a complete audit trail. For example, if a person, bicycle, or traffic light is detected, that object is still visible in the log, but it does not affect the vehicle count. This makes debugging easier and makes the result more defensible in a discussion.

## Why This Matters

If the system only counted vehicles and discarded everything else, there would be no way to analyze false positives or explain why the count changed. By logging all tracked objects, the project can answer questions like:

- Which classes appeared in the scene?
- How long did each object stay visible?
- Did any non-vehicle object cross the counting line?
- Was a vehicle counted once or multiple times?
- Did the error come from detection, tracking, or line-crossing logic?

## Technical Flow

1. YOLOv3 detects objects in each frame.
2. Non-maximum suppression removes duplicate boxes.
3. SORT assigns stable track IDs across frames.
4. The system matches each track back to the most likely YOLO detection to recover its class label.
5. Every track observation is written to `tracking_log.csv`.
6. Each track is counted only once if its class is a vehicle and its centroid crosses the counting line.
7. A final per-track summary is written to `track_summary.csv`.

## Real-Time Capability

The same pipeline can run against a webcam index, a network stream, or a video file. In display mode, OpenCV shows the annotated frame immediately after processing, including bounding boxes, object labels, track IDs, the count line, current vehicle count, speed estimates, and processing FPS.

This makes the project useful both as an offline analysis tool and as a live demonstration. For a webcam demo, the command is:

```text
python main.py --source 0 --display --no-output --no-logs
```

For an IP camera or stream, `--source` can be an RTSP or HTTP URL.

## Deployment Capability

The project also has a Gradio web app. In that mode, a user uploads a video in the browser, the app streams annotated preview frames while YOLO is processing, and the final video plus logs become downloadable artifacts.

Each execution creates a separate timestamped run folder under `output/runs/`, such as:

```text
output/runs/20260511_143000/
```

This is important for a practical system because runs are reproducible and auditable. The folder contains the processed video, frame-level tracking log, per-track summary, and metadata describing the settings used for that run.

## What To Say About Speed Estimation

Speed is estimated using centroid displacement between frames:

```text
speed = pixel_distance * meters_per_pixel * fps
```

Then it is converted to km/h. The current `meters_per_pixel` value is configurable because real speed estimation needs scene calibration. This is an honest and technically sound way to present it: the prototype has the pipeline for speed estimation, while accurate real-world deployment requires camera calibration or known road measurements.

## Strengths

- Produces both visual output and structured CSV evidence.
- Avoids double-counting by counting each track only once.
- Counts only vehicle classes while preserving all object tracks for auditability.
- Uses timestamps, not just frame numbers, so results can be discussed in real time units.
- Exposes line coordinates, class filters, thresholds, and calibration through command-line options.
- Supports browser-based video upload through the Gradio deployment app.
- Stores each run in its own timestamped folder instead of overwriting previous outputs.

## Limitations To Acknowledge

- YOLOv3 is older than YOLOv11 and may miss or misclassify smaller vehicles.
- SORT is fast but can switch identities during occlusion.
- Speed accuracy depends on calibration.
- A single fixed count line works for this camera angle but should be configurable per scene.
- YOLOv3 on CPU may not reach full camera FPS, so live performance depends on hardware and model input size.

## Best Next Step

The strongest next improvement is upgrading the detector to YOLOv11 while keeping the same tracking and logging design. That would improve detection quality without losing the explainability layer created by the CSV logs.
