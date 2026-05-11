# Running Manual

## Project Location

```powershell
D:\Projects\Yolov11\vehicle-counting
```

Open PowerShell and move into the project:

```powershell
cd D:\Projects\Yolov11\vehicle-counting
```

## 1. Environment Setup

Activate the existing virtual environment:

```powershell
.\.venv\Scripts\activate
```

If you need to recreate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Model Files

The project expects these YOLOv3 files:

```text
yolo-coco/coco.names
yolo-coco/yolov3.cfg
yolo-coco/yolov3.weights
```

If `yolov3.weights` is missing, download it:

```powershell
Invoke-WebRequest -Uri "https://pjreddie.com/media/files/yolov3.weights" -OutFile "yolo-coco\yolov3.weights"
```

## 3. Quick Smoke Test

Run only a few frames to check that everything works:

```powershell
python main.py --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4" --max-frames 10
```

The output will be placed in a new timestamped folder:

```text
output/runs/YYYYMMDD_HHMMSS/
```

## 4. Full Video Processing

Run the full sample video:

```powershell
python main.py --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4"
```

The run folder will contain:

```text
output_video.mp4
tracking_log.csv
track_summary.csv
run_metadata.json
```

## 5. Run Without Passing A Source

If the local sample video remains at `..\test-sample\Vehicle Dataset Sample 2.mp4`, this also works:

```powershell
python main.py
```

The script checks:

```text
input/input_video.mp4
..\test-sample\Vehicle Dataset Sample 2.mp4
```

## 6. Real-Time Webcam Mode

Run from webcam index `0` and show the annotated feed:

```powershell
python main.py --source 0 --display --no-output --no-logs
```

Press `q` in the display window to stop.

For better CPU performance:

```powershell
python main.py --source 0 --display --no-output --no-logs --width 320 --height 320
```

## 7. RTSP Or HTTP Stream

Run with a camera stream:

```powershell
python main.py --source "rtsp://username:password@camera-ip/stream" --display --no-output
```

Or with an HTTP stream:

```powershell
python main.py --source "http://camera-ip/video" --display --no-output
```

## 8. Display A Video Like A Live Feed

This plays a video in a display window and attempts to pace it near its source FPS:

```powershell
python main.py --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4" --display --realtime
```

## 9. Web App Mode

Start the Gradio web app:

```powershell
python app.py
```

Open this URL:

```text
http://127.0.0.1:7860
```

In the web app:

1. Upload a video.
2. Adjust confidence, NMS threshold, YOLO input size, count line, or calibration.
3. Click `Process Video`.
4. Watch annotated preview frames while processing.
5. Download the output video, tracking log, summary log, and metadata file.

## 10. Docker Mode

Build the image:

```powershell
docker build -t vehicle-counting .
```

Run the web app:

```powershell
docker run --rm -p 7860:7860 vehicle-counting
```

Open:

```text
http://127.0.0.1:7860
```

## 11. Useful CLI Options

### Run With YOLOv11

```powershell
python main.py --source "video.mp4" --detector ultralytics --model yolo11n.pt
```

### Run With YOLOv11 And ByteTrack

```powershell
python main.py --source "video.mp4" --detector ultralytics --model yolo11n.pt --tracker bytetrack
```

### Change Count Line

```powershell
python main.py --source "video.mp4" --line 100 400 1000 400
```

### Change Counted Classes

```powershell
python main.py --source "video.mp4" --count-classes car,bus,truck
```

### Track Only Vehicle Classes

```powershell
python main.py --source "video.mp4" --track-classes car,motorbike,bus,truck
```

### Track All Classes

```powershell
python main.py --source "video.mp4" --track-classes all
```

### Change Detection Confidence

```powershell
python main.py --source "video.mp4" --confidence 0.45
```

### Change Speed Calibration

```powershell
python main.py --source "video.mp4" --meters-per-pixel 0.05
```

### Use A Custom Run Folder

```powershell
python main.py --source "video.mp4" --run-dir "output\runs\demo_run"
```

### Use ROI Filtering

Inline ROI:

```powershell
python main.py --source "video.mp4" --roi "0,0;1280,0;1280,720;0,720"
```

ROI file:

```powershell
python main.py --source "video.mp4" --roi-file configs\roi.sample.json
```

### Use Per-Lane Counting

```powershell
python main.py --source "video.mp4" --lanes-file configs\lanes.sample.json
```

### Save Counted Vehicle Crops

```powershell
python main.py --source "video.mp4" --save-crops
```

### Disable Heatmap Generation

```powershell
python main.py --source "video.mp4" --no-heatmap
```

### Disable Output Video

```powershell
python main.py --source "video.mp4" --no-output
```

### Disable CSV Logs

```powershell
python main.py --source "video.mp4" --no-logs
```

## 12. Output File Meanings

### `output_video.mp4`

Annotated processed video with:

- bounding boxes
- track IDs
- class labels
- speed estimates
- counting line
- total vehicle count
- processing FPS

### `tracking_log.csv`

Frame-level object log. Each row represents one tracked object observation in one frame.

Important columns:

- `frame`
- `timestamp_sec`
- `track_id`
- `label`
- `confidence`
- `center_x`
- `center_y`
- `speed_kmh`
- `crossed_count_line`
- `vehicle_count_total`

### `track_summary.csv`

One row per tracked object. Useful for discussion because it shows how long each object was visible and whether it crossed the count line.

### `run_metadata.json`

Stores run settings, source path, output paths, processed frame count, final vehicle count, and elapsed time.

### `detection_heatmap.png`

Heatmap generated from vehicle center points.

### `detection_heatmap_overlay.png`

Heatmap blended on top of the source frame.

### `report.html`

HTML report with run metrics, class observations, lane observations, direction observations, and heatmap preview.

### `report.pdf`

PDF report for presenting or submitting the run summary.

### `vehicle_crops/`

Optional folder created by `--save-crops`. It stores counted vehicle crops that can be used later for license-plate or OCR experiments.

## 13. Model Comparison

Compare YOLOv3 and YOLOv11 on the same video:

```powershell
python compare_models.py --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4" --max-frames 120
```

The comparison artifacts are written to:

```text
output/comparisons/YYYYMMDD_HHMMSS/
```

The key files are:

```text
comparison.csv
comparison.json
```

Run only YOLOv3:

```powershell
python compare_models.py --source "video.mp4" --models yolov3:yolov3:yolov3 --max-frames 60
```

Run only YOLOv11:

```powershell
python compare_models.py --source "video.mp4" --models yolo11n:ultralytics:yolo11n.pt --max-frames 60
```

## 14. REST API Mode

Start the API server:

```powershell
python api.py
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

The main endpoints are:

```text
POST /runs
GET /runs
GET /runs/{job_id}
GET /runs/{job_id}/artifacts/{artifact_name}
```

The API processes uploaded videos as background jobs.

## 15. Troubleshooting

### `Missing YOLO file`

Make sure `yolo-coco/yolov3.weights` exists.

### `Could not open video source`

Check that the video path exists. If the path has spaces, wrap it in quotes.

### Webcam Does Not Open

Try another camera index:

```powershell
python main.py --source 1 --display --no-output --no-logs
```

### Real-Time Mode Is Slow

Lower YOLO input size:

```powershell
python main.py --source 0 --display --width 320 --height 320 --no-output --no-logs
```

Or run on a machine with GPU acceleration after migrating the detector to a GPU-supported YOLO backend.

### Web App Port Is Busy

If port `7860` is busy, change the port in `app.py`:

```python
demo.queue().launch(server_name="0.0.0.0", server_port=7861)
```

## 16. Recommended Demo Command

For the TA, this is the cleanest CLI demonstration:

```powershell
python main.py --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4" --max-frames 60
```

Then open the created folder under:

```text
output/runs/
```

For the deployment demonstration:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

For YOLOv11 comparison:

```powershell
python compare_models.py --source "D:\Projects\Yolov11\test-sample\Vehicle Dataset Sample 2.mp4" --max-frames 60
```
