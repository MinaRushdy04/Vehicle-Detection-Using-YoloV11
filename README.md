# Vehicle Counting and Object Tracking with YOLOv3 + SORT

This project detects objects in traffic video with YOLOv3, tracks them across frames with SORT, counts only vehicle-class objects crossing a configured line, and logs every tracked object with timestamps for later analysis.

The important refinement is that counting and tracking are separated:

- **Counting target:** `car`, `motorbike`, `bus`, and `truck`.
- **Tracking/logging target:** all detected COCO classes by default.
- **Reason:** the system produces a clean traffic count while still keeping evidence for non-counted detections such as people, traffic lights, or false positives.

## What The Pipeline Does

1. Loads YOLOv3 Darknet files from `yolo-coco/`.
2. Reads a video file, webcam, or network stream.
3. Detects objects frame by frame.
4. Sends detections into SORT to maintain stable track IDs.
5. Assigns each track a class label using the matched YOLO detection.
6. Counts a track once when a countable vehicle crosses the yellow line.
7. Writes an annotated output video.
8. Writes CSV logs for every tracked object observation and each completed track.

## Outputs

By default, the script writes:

- `output/runs/YYYYMMDD_HHMMSS/output_video.mp4` - annotated video with boxes, track IDs, classes, speed estimates, and count line.
- `output/runs/YYYYMMDD_HHMMSS/tracking_log.csv` - frame-level log of every tracked object.
- `output/runs/YYYYMMDD_HHMMSS/track_summary.csv` - one row per track with label, first/last timestamp, duration, confidence, and whether it crossed the line.
- `output/runs/YYYYMMDD_HHMMSS/run_metadata.json` - source, settings, output paths, frame count, elapsed time, and final vehicle count.

Each run gets a separate timestamped folder, so results from a 2:30 PM run and a 2:40 PM run are kept independently.

`tracking_log.csv` includes:

- frame number
- timestamp in seconds
- track ID
- predicted object label
- whether the label is a countable vehicle class
- confidence
- bounding box
- centroid
- estimated speed
- line-crossing event
- running vehicle count

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download YOLOv3 weights:

```bash
bash download_weights
```

On Windows without `bash`/`wget`, download `yolov3.weights` from:

```text
https://pjreddie.com/media/files/yolov3.weights
```

Place it here:

```text
yolo-coco/yolov3.weights
```

## Usage

Run with an explicit video:

```bash
python main.py --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4"
```

If `--input` is omitted, the script first checks `input/input_video.mp4`, then the local test sample at `../test-sample/Vehicle Dataset Sample 2.mp4`.

Useful options:

```bash
python main.py ^
  --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4" ^
  --line 100 400 1000 400 ^
  --count-classes car,motorbike,bus,truck ^
  --track-classes all ^
  --meters-per-pixel 0.05
```

For a quick smoke test:

```bash
python main.py --max-frames 30
```

## Real-Time Display

Use `--source` with `--display` to process a live feed and show bounding boxes, labels, track IDs, count line, vehicle count, speed estimates, and processing FPS.

Webcam:

```bash
python main.py --source 0 --display --no-output --no-logs
```

RTSP or HTTP stream:

```bash
python main.py --source "rtsp://username:password@camera-ip/stream" --display --no-output
```

Video file played like a live feed:

```bash
python main.py --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4" --display --realtime
```

Press `q` in the display window to stop.

For better live performance on CPU, reduce the YOLO input size:

```bash
python main.py --source 0 --display --no-output --width 320 --height 320
```

## Web Demo Deployment

The project includes a Gradio web app for an upload-and-process workflow:

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

The web app accepts an uploaded video, streams annotated preview frames while processing, then exposes the final output video, tracking log, track summary, and run metadata from that run folder.

Docker deployment:

```bash
docker build -t vehicle-counting .
docker run --rm -p 7860:7860 vehicle-counting
```

## Design Notes For Discussion

The system intentionally tracks more than it counts. That makes the project stronger because the final count is not a black box: every detection and track can be audited in the CSV logs. If a wrong count appears, the log shows whether the issue came from detection, tracking, line crossing, or class filtering.

Vehicle counting uses a one-count-per-track rule, so the same object should not be counted repeatedly if it jitters around the line. Non-vehicle classes are still logged, but they do not increment the traffic counter.

Speed is estimated from centroid movement, video FPS, and a configurable `meters-per-pixel` calibration. This is suitable for demonstration and relative comparison, but a real deployment should calibrate the scene using known road distances or camera homography.

## Current Limitations

- The detector is YOLOv3, not YOLOv11.
- Speed is approximate unless the camera scene is calibrated.
- SORT can switch IDs during heavy occlusion.
- The count line is manually configured.
- Detection quality depends on lighting, camera angle, and video resolution.

## Strong Next Improvements

## Added Advanced Features

- Optional YOLOv11/Ultralytics detector support with `--detector ultralytics --model yolo11n.pt`.
- Optional ByteTrack support with `--tracker bytetrack` when using Ultralytics.
- Model comparison script for YOLOv3 vs YOLOv11 experiments.
- ROI filtering from inline polygon strings or JSON files.
- Per-lane assignment using lane polygon config files.
- Direction estimation from track movement.
- Low-confidence review flags in the logs.
- Detection heatmap and heatmap overlay artifacts.
- HTML and PDF report generation per run.
- Optional counted-vehicle crop export for plate/OCR follow-up experiments.
- REST API with background jobs through `api.py`.
- Previous-run browsing in the Gradio app.

The dashboard feature is intentionally halted for now.

## Strong Next Improvements

- Add homography-based speed calibration.
- Add a trained license-plate detector/OCR model for the saved vehicle crops.
- Add authentication if the Gradio app or API is exposed publicly.
- Add persistent database storage for run metadata.
- Add GPU/TensorRT deployment for higher FPS.
