import argparse
import csv
import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from sort import Sort


DEFAULT_COUNT_CLASSES = {"car", "motorbike", "bus", "truck"}
LABEL_ALIASES = {
    "motorbike": "motorcycle",
    "motorcycle": "motorbike",
    "aeroplane": "airplane",
    "airplane": "aeroplane",
    "sofa": "couch",
    "couch": "sofa",
    "tvmonitor": "tv",
    "tv": "tvmonitor",
}
DEFAULT_INPUT_CANDIDATES = [
    Path("input/input_video.mp4"),
    Path("../test-sample/Vehicle Dataset Sample 2.mp4"),
]
LOG_COLUMNS = [
    "frame",
    "timestamp_sec",
    "track_id",
    "label",
    "is_counted_vehicle_class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "lane",
    "direction",
    "speed_kmh",
    "low_confidence_review",
    "crossed_count_line",
    "vehicle_count_total",
]
SUMMARY_COLUMNS = [
    "track_id",
    "label",
    "is_counted_vehicle_class",
    "dominant_lane",
    "dominant_direction",
    "first_frame",
    "last_frame",
    "first_timestamp_sec",
    "last_timestamp_sec",
    "duration_sec",
    "observations",
    "max_confidence",
    "average_speed_kmh",
    "max_speed_kmh",
    "low_confidence_observations",
    "crossed_count_line",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count vehicles and log all tracked objects in a video using YOLOv3 + SORT."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source: file path, webcam index like 0, or RTSP/HTTP stream URL. Overrides --input.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Video file path. Kept for offline runs; use --source for webcam or stream feeds.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/runs"),
        help="Root folder used for timestamped run folders.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run folder. Defaults to output/runs/YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--summary-log", type=Path, default=None)
    parser.add_argument("--yolo-dir", type=Path, default=Path("yolo-coco"))
    parser.add_argument(
        "--detector",
        choices=["yolov3", "ultralytics"],
        default="yolov3",
        help="Detection backend. Use ultralytics with --model yolo11n.pt for YOLOv11.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model path/name, for example yolo11n.pt.",
    )
    parser.add_argument(
        "--tracker",
        choices=["sort", "bytetrack"],
        default="sort",
        help="Tracking backend. bytetrack requires --detector ultralytics.",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=416, help="YOLO network input width.")
    parser.add_argument("--height", type=int, default=416, help="YOLO network input height.")
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--max-age", type=int, default=3, help="SORT frames to keep unmatched tracks alive.")
    parser.add_argument("--min-hits", type=int, default=3, help="SORT hits before a track is considered stable.")
    parser.add_argument(
        "--count-classes",
        default="car,motorbike,bus,truck",
        help="Comma-separated COCO labels that should increment the vehicle counter.",
    )
    parser.add_argument(
        "--track-classes",
        default="all",
        help="Comma-separated COCO labels to track, or 'all' to log every detected class.",
    )
    parser.add_argument(
        "--line",
        type=int,
        nargs=4,
        default=[100, 400, 1000, 400],
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Counting line coordinates.",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=0.05,
        help="Approximate calibration used for speed estimation. Tune this per camera scene.",
    )
    parser.add_argument(
        "--calibration-pixels",
        type=float,
        default=None,
        help="Known pixel distance for speed calibration.",
    )
    parser.add_argument(
        "--calibration-meters",
        type=float,
        default=None,
        help="Known real-world meters matching --calibration-pixels.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI polygon as 'x,y;x,y;x,y'. Detections outside it are ignored.",
    )
    parser.add_argument(
        "--roi-file",
        type=Path,
        default=None,
        help="Optional JSON file containing an ROI polygon.",
    )
    parser.add_argument(
        "--lanes-file",
        type=Path,
        default=None,
        help="Optional JSON file mapping lane names to polygon points.",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.45,
        help="Tracked observations below this confidence are flagged for review.",
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=12.0,
        help="Minimum pixel movement before assigning a direction label.",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip heatmap artifact generation.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save counted vehicle crops as a second-stage hook for plate/OCR experiments.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick smoke tests.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the annotated frames in a live OpenCV window.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="For file sources, pace display near source FPS instead of processing as fast as possible.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not save the annotated output video. Useful for live feeds.",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Do not write CSV logs. Useful for long-running live feeds.",
    )
    parser.add_argument("--window-name", default="Vehicle Tracking")
    parser.add_argument("--quit-key", default="q", help="Key used to close the live window.")
    return parser.parse_args()


def split_labels(value):
    if value.lower() == "all":
        return None
    return {label.strip() for label in value.split(",") if label.strip()}


def resolve_label_set(requested_labels, available_labels):
    available = set(available_labels)
    resolved = set()
    unknown = set()
    for label in requested_labels:
        if label in available:
            resolved.add(label)
        elif LABEL_ALIASES.get(label) in available:
            resolved.add(LABEL_ALIASES[label])
        else:
            unknown.add(label)
    return resolved, unknown


def parse_points(value):
    points = []
    if not value:
        return points
    for part in value.split(";"):
        if not part.strip():
            continue
        x, y = part.split(",", 1)
        points.append((int(float(x.strip())), int(float(y.strip()))))
    return points


def normalize_points(points):
    return [(int(point[0]), int(point[1])) for point in points]


def load_roi(args):
    if args.roi:
        return parse_points(args.roi)
    if not args.roi_file:
        return None
    data = json.loads(Path(args.roi_file).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("roi", data.get("points", []))
    return normalize_points(data)


def load_lanes(lanes_file):
    if not lanes_file:
        return {}
    data = json.loads(Path(lanes_file).read_text(encoding="utf-8"))
    if "lanes" in data:
        data = data["lanes"]
    return {name: normalize_points(points) for name, points in data.items()}


def point_in_polygon(point, polygon):
    if not polygon:
        return True
    contour = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def assign_lane(point, lanes):
    for lane_name, polygon in lanes.items():
        if point_in_polygon(point, polygon):
            return lane_name
    return "unassigned" if lanes else ""


def estimate_direction(first_center, current_center, threshold):
    if not first_center:
        return "stationary"
    dx = current_center[0] - first_center[0]
    dy = current_center[1] - first_center[1]
    if math.hypot(dx, dy) < threshold:
        return "stationary"
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def most_common(counter, default=""):
    if not counter:
        return default
    return counter.most_common(1)[0][0]


def apply_calibration(args):
    if args.calibration_pixels and args.calibration_meters:
        args.meters_per_pixel = args.calibration_meters / args.calibration_pixels
        print(f"[INFO] calibrated meters_per_pixel: {args.meters_per_pixel:.6f}")


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def ccw(point_a, point_b, point_c):
    return (point_c[1] - point_a[1]) * (point_b[0] - point_a[0]) > (
        point_b[1] - point_a[1]
    ) * (point_c[0] - point_a[0])


def intersect(point_a, point_b, point_c, point_d):
    return ccw(point_a, point_c, point_d) != ccw(point_b, point_c, point_d) and ccw(
        point_a, point_b, point_c
    ) != ccw(point_a, point_b, point_d)


def bbox_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def is_readable_video(path):
    if not path.exists() or path.stat().st_size < 1024:
        return False
    cap = cv2.VideoCapture(str(path))
    is_valid = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    cap.release()
    return is_valid


def choose_input_path(input_arg):
    if input_arg:
        if not is_readable_video(input_arg):
            raise FileNotFoundError(f"Input video is missing or unreadable: {input_arg}")
        return input_arg

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if is_readable_video(candidate):
            return candidate

    candidates = ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"No readable input video found. Checked: {candidates}")


def looks_like_stream(source):
    source = source.lower()
    return source.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))


def resolve_capture_source(args):
    if args.source is not None:
        source = str(args.source).strip()
        if source.isdigit():
            return int(source), f"camera:{source}", True
        if looks_like_stream(source):
            return source, source, True

        path = Path(source)
        if not is_readable_video(path):
            raise FileNotFoundError(f"Video source is missing or unreadable: {source}")
        return str(path), str(path), False

    input_path = choose_input_path(args.input)
    return str(input_path), str(input_path), False


def create_timestamped_run_dir(output_root, run_dir=None):
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def prepare_run_outputs(args):
    run_dir = create_timestamped_run_dir(args.output_root, args.run_dir)
    args.output = Path(args.output) if args.output else run_dir / "output_video.mp4"
    args.log = Path(args.log) if args.log else run_dir / "tracking_log.csv"
    args.summary_log = Path(args.summary_log) if args.summary_log else run_dir / "track_summary.csv"

    if not args.no_output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_logs:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.summary_log.parent.mkdir(parents=True, exist_ok=True)

    return run_dir


def video_fourcc(output_path):
    suffix = Path(output_path).suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        return cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter_fourcc(*"MJPG")


def serialize_args(args):
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def write_run_metadata(run_dir, metadata):
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def write_heatmap_artifacts(run_dir, heatmap, base_frame):
    if heatmap is None or base_frame is None or not np.any(heatmap):
        return None, None

    normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_frame, 0.65, colored, 0.35, 0)
    heatmap_path = run_dir / "detection_heatmap.png"
    overlay_path = run_dir / "detection_heatmap_overlay.png"
    cv2.imwrite(str(heatmap_path), colored)
    cv2.imwrite(str(overlay_path), overlay)
    return str(heatmap_path), str(overlay_path)


def write_html_report(run_dir, result):
    report_path = run_dir / "report.html"
    class_rows = "".join(
        f"<tr><td>{label}</td><td>{count}</td></tr>"
        for label, count in result.get("class_observations", {}).items()
    )
    lane_rows = "".join(
        f"<tr><td>{lane}</td><td>{count}</td></tr>"
        for lane, count in result.get("lane_observations", {}).items()
    )
    direction_rows = "".join(
        f"<tr><td>{direction}</td><td>{count}</td></tr>"
        for direction, count in result.get("direction_observations", {}).items()
    )
    heatmap_img = ""
    if result.get("heatmap_overlay"):
        heatmap_img = f"<h2>Detection Heatmap</h2><img src='{Path(result['heatmap_overlay']).name}' />"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Vehicle Counting Run Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; min-width: 360px; }}
    td, th {{ border: 1px solid #c9d1d9; padding: 8px 10px; text-align: left; }}
    img {{ max-width: 900px; width: 100%; border: 1px solid #c9d1d9; }}
    code {{ background: #f3f4f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Vehicle Counting Run Report</h1>
  <p><strong>Run folder:</strong> <code>{result['run_dir']}</code></p>
  <p><strong>Source:</strong> <code>{result['source']}</code></p>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Final vehicle count</td><td>{result['final_vehicle_count']}</td></tr>
    <tr><td>Processed frames</td><td>{result['processed_frames']}</td></tr>
    <tr><td>Elapsed seconds</td><td>{result['elapsed_seconds']}</td></tr>
    <tr><td>Source FPS</td><td>{result['source_fps']}</td></tr>
    <tr><td>Low-confidence review events</td><td>{result['low_confidence_events']}</td></tr>
  </table>
  <h2>Class Observations</h2>
  <table><tr><th>Class</th><th>Observations</th></tr>{class_rows}</table>
  <h2>Lane Observations</h2>
  <table><tr><th>Lane</th><th>Observations</th></tr>{lane_rows}</table>
  <h2>Direction Observations</h2>
  <table><tr><th>Direction</th><th>Observations</th></tr>{direction_rows}</table>
  {heatmap_img}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)


def write_pdf_report(run_dir, result):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return None

    pdf_path = run_dir / "report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    y = height - 48
    lines = [
        "Vehicle Counting Run Report",
        f"Run folder: {result['run_dir']}",
        f"Source: {result['source']}",
        f"Final vehicle count: {result['final_vehicle_count']}",
        f"Processed frames: {result['processed_frames']}",
        f"Elapsed seconds: {result['elapsed_seconds']}",
        f"Low-confidence review events: {result['low_confidence_events']}",
        "",
        "Class observations:",
    ]
    for label, count in result.get("class_observations", {}).items():
        lines.append(f"  {label}: {count}")
    lines.append("")
    lines.append("Lane observations:")
    for lane, count in result.get("lane_observations", {}).items():
        lines.append(f"  {lane}: {count}")
    lines.append("")
    lines.append("Direction observations:")
    for direction, count in result.get("direction_observations", {}).items():
        lines.append(f"  {direction}: {count}")

    for line in lines:
        if y < 48:
            c.showPage()
            y = height - 48
        c.drawString(48, y, line[:110])
        y -= 16
    c.save()
    return str(pdf_path)


def load_yolo(yolo_dir):
    labels_path = yolo_dir / "coco.names"
    weights_path = yolo_dir / "yolov3.weights"
    config_path = yolo_dir / "yolov3.cfg"

    missing = [path for path in [labels_path, weights_path, config_path] if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing YOLO file(s): {missing_paths}")

    labels = labels_path.read_text(encoding="utf-8").strip().splitlines()
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return labels, net, output_layers


def load_ultralytics_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for --detector ultralytics. "
            "Install it with: pip install ultralytics"
        ) from exc
    print(f"[INFO] loading Ultralytics model: {model_path}")
    model = YOLO(model_path)
    names = model.names
    if isinstance(names, dict):
        labels = [names[index] for index in sorted(names)]
    else:
        labels = list(names)
    return labels, model


def detect_objects(frame, net, output_layers, labels, args, tracked_classes):
    frame_h, frame_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (args.width, args.height),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            label = labels[class_id]

            if confidence <= args.confidence:
                continue
            if tracked_classes is not None and label not in tracked_classes:
                continue

            box = detection[0:4] * np.array([frame_w, frame_h, frame_w, frame_h])
            center_x, center_y, width, height = box.astype("int")
            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(confidence)
            class_ids.append(class_id)

    detections = []
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, args.nms_threshold)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            detections.append(
                {
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "confidence": confidences[i],
                    "class_id": class_ids[i],
                    "label": labels[class_ids[i]],
                }
            )
    return detections


def detect_ultralytics(frame, model, labels, args, tracked_classes):
    results = model.predict(
        source=frame,
        imgsz=(args.height, args.width),
        conf=args.confidence,
        iou=args.nms_threshold,
        verbose=False,
    )
    return ultralytics_results_to_detections(results[0], labels, tracked_classes)


def track_ultralytics_bytetrack(frame, model, labels, args, tracked_classes):
    results = model.track(
        source=frame,
        imgsz=(args.height, args.width),
        conf=args.confidence,
        iou=args.nms_threshold,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    return ultralytics_results_to_detections(results[0], labels, tracked_classes, include_track_id=True)


def ultralytics_results_to_detections(result, labels, tracked_classes, include_track_id=False):
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = None
    if include_track_id and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)

    for index, (box, confidence, class_id) in enumerate(zip(xyxy, confs, classes)):
        label = labels[class_id]
        if tracked_classes is not None and label not in tracked_classes:
            continue
        detection = {
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "confidence": float(confidence),
            "class_id": int(class_id),
            "label": label,
        }
        if ids is not None:
            detection["track_id"] = int(ids[index])
        detections.append(detection)
    return detections


def load_detector(args):
    if args.detector == "ultralytics":
        labels, model = load_ultralytics_model(args.model)
        return labels, {"kind": "ultralytics", "model": model}

    labels, net, output_layers = load_yolo(args.yolo_dir)
    return labels, {"kind": "yolov3", "net": net, "output_layers": output_layers}


def get_detections(frame, detector, labels, args, tracked_classes):
    if args.tracker == "bytetrack":
        if detector["kind"] != "ultralytics":
            raise ValueError("--tracker bytetrack requires --detector ultralytics")
        return track_ultralytics_bytetrack(frame, detector["model"], labels, args, tracked_classes)

    if detector["kind"] == "ultralytics":
        return detect_ultralytics(frame, detector["model"], labels, args, tracked_classes)

    return detect_objects(frame, detector["net"], detector["output_layers"], labels, args, tracked_classes)


def match_track_to_detection(track_box, detections, used_detection_indexes):
    best_iou = 0.0
    best_index = None
    for index, detection in enumerate(detections):
        if index in used_detection_indexes:
            continue
        iou = bbox_iou(track_box, detection["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_index = index

    if best_index is None or best_iou < 0.1:
        return None

    used_detection_indexes.add(best_index)
    return detections[best_index]


def estimate_speed_kmh(previous_center, current_center, fps, meters_per_pixel):
    if not previous_center or fps <= 0 or meters_per_pixel <= 0:
        return None
    pixel_distance = math.dist(previous_center, current_center)
    meters_per_second = pixel_distance * meters_per_pixel * fps
    return meters_per_second * 3.6


def draw_label(frame, text, origin, color):
    x, y = origin
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_w, text_h = text_size
    y = max(y, text_h + 8)
    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 6),
        (x + text_w + 8, y + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def update_track_state(track_states, track_id, frame_index, timestamp, detection):
    state = track_states.setdefault(
        track_id,
        {
            "first_frame": frame_index,
            "first_time": timestamp,
            "last_frame": frame_index,
            "last_time": timestamp,
            "observations": 0,
            "class_counts": Counter(),
            "lane_counts": Counter(),
            "direction_counts": Counter(),
            "max_confidence": 0.0,
            "speed_sum": 0.0,
            "speed_observations": 0,
            "max_speed": 0.0,
            "low_confidence_observations": 0,
            "counted": False,
            "crossed": False,
            "first_center": None,
            "last_center": None,
        },
    )
    state["last_frame"] = frame_index
    state["last_time"] = timestamp
    state["observations"] += 1

    if detection:
        state["class_counts"][detection["label"]] += 1
        state["max_confidence"] = max(state["max_confidence"], detection["confidence"])

    return state


def most_common_label(state):
    if not state["class_counts"]:
        return "unknown"
    return state["class_counts"].most_common(1)[0][0]


def write_summary(summary_writer, track_states, count_classes):
    for track_id in sorted(track_states):
        state = track_states[track_id]
        label = most_common_label(state)
        avg_speed = (
            state["speed_sum"] / state["speed_observations"]
            if state["speed_observations"]
            else 0.0
        )
        summary_writer.writerow(
            {
                "track_id": track_id,
                "label": label,
                "is_counted_vehicle_class": label in count_classes,
                "dominant_lane": most_common(state["lane_counts"], ""),
                "dominant_direction": most_common(state["direction_counts"], "stationary"),
                "first_frame": state["first_frame"],
                "last_frame": state["last_frame"],
                "first_timestamp_sec": f"{state['first_time']:.3f}",
                "last_timestamp_sec": f"{state['last_time']:.3f}",
                "duration_sec": f"{state['last_time'] - state['first_time']:.3f}",
                "observations": state["observations"],
                "max_confidence": f"{state['max_confidence']:.4f}",
                "average_speed_kmh": f"{avg_speed:.2f}",
                "max_speed_kmh": f"{state['max_speed']:.2f}",
                "low_confidence_observations": state["low_confidence_observations"],
                "crossed_count_line": state["crossed"],
            }
        )


def run_pipeline(args, frame_callback=None, print_logs=True):
    apply_calibration(args)
    requested_count_classes = split_labels(args.count_classes) or DEFAULT_COUNT_CLASSES
    requested_tracked_classes = split_labels(args.track_classes)
    line_start = (args.line[0], args.line[1])
    line_end = (args.line[2], args.line[3])
    roi_polygon = load_roi(args)
    lanes = load_lanes(args.lanes_file)

    capture_source, source_label, is_live_source = resolve_capture_source(args)
    run_dir = prepare_run_outputs(args)

    def log(message):
        if print_logs:
            print(message)

    labels, detector = load_detector(args)
    count_classes, unknown_count_classes = resolve_label_set(requested_count_classes, labels)
    if unknown_count_classes:
        raise ValueError(f"Unknown count class label(s): {sorted(unknown_count_classes)}")
    tracked_classes = None
    if requested_tracked_classes:
        tracked_classes, unknown_track_classes = resolve_label_set(requested_tracked_classes, labels)
        if unknown_track_classes:
            raise ValueError(f"Unknown track class label(s): {sorted(unknown_track_classes)}")

    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits) if args.tracker == "sort" else None
    track_states = {}
    vehicle_count = 0
    class_counts = Counter()
    lane_counts = Counter()
    direction_counts = Counter()
    low_confidence_events = 0
    heatmap = None
    heatmap_base_frame = None
    crop_dir = run_dir / "vehicle_crops"
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(512, 3), dtype="uint8")

    video = cv2.VideoCapture(capture_source)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video source: {source_label}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = video.get(cv2.CAP_PROP_FPS)
    if not source_fps or not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = args.output_fps

    log(f"[INFO] run folder: {run_dir}")
    log(f"[INFO] source: {source_label}")
    if is_live_source or total_frames <= 0:
        log("[INFO] total frames: live/unknown")
    else:
        log(f"[INFO] total frames: {total_frames}")
    log(f"[INFO] source FPS: {source_fps:.2f}")
    if args.display:
        log(f"[INFO] live display enabled; press '{args.quit_key[:1] or 'q'}' to stop")

    writer = None
    frame_index = 0
    run_started_at = time.perf_counter()
    capture_started_at = time.perf_counter()
    previous_timestamp = None
    log_file = None
    summary_file = None
    log_writer = None
    summary_writer = None
    summary_written = False

    try:
        if not args.no_logs:
            log_file = args.log.open("w", newline="", encoding="utf-8")
            summary_file = args.summary_log.open("w", newline="", encoding="utf-8")
            log_writer = csv.DictWriter(log_file, fieldnames=LOG_COLUMNS)
            summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS)
            log_writer.writeheader()
            summary_writer.writeheader()

        tracking_scope = "all classes" if tracked_classes is None else ", ".join(sorted(tracked_classes))
        count_scope = ", ".join(sorted(count_classes))
        quit_key = ord(args.quit_key[:1] or "q")

        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            grabbed, frame = video.read()
            if not grabbed:
                break

            now = time.perf_counter()
            if is_live_source:
                timestamp = now - capture_started_at
            else:
                timestamp = frame_index / source_fps if source_fps > 0 else now - capture_started_at

            speed_fps = source_fps
            if previous_timestamp is not None and timestamp > previous_timestamp:
                speed_fps = 1.0 / (timestamp - previous_timestamp)

            frame = adjust_gamma(frame, gamma=args.gamma)
            if heatmap is None:
                heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
                heatmap_base_frame = frame.copy()

            detections = get_detections(frame, detector, labels, args, tracked_classes)
            if roi_polygon:
                detections = [
                    detection
                    for detection in detections
                    if point_in_polygon(bbox_center(detection["bbox"]), roi_polygon)
                ]

            if args.tracker == "bytetrack":
                tracks = np.asarray(
                    [
                        [*detection["bbox"], detection["track_id"]]
                        for detection in detections
                        if "track_id" in detection
                    ],
                    dtype=np.float32,
                )
                if tracks.size == 0:
                    tracks = np.empty((0, 5), dtype=np.float32)
            else:
                dets = np.asarray(
                    [[*detection["bbox"], detection["confidence"]] for detection in detections],
                    dtype=np.float32,
                )
                if dets.size == 0:
                    dets = np.empty((0, 5), dtype=np.float32)
                tracks = tracker.update(dets)
            used_detection_indexes = set()

            for track in tracks:
                x1, y1, x2, y2 = [float(value) for value in track[:4]]
                track_id = int(track[4])
                track_box = [x1, y1, x2, y2]
                detection = match_track_to_detection(track_box, detections, used_detection_indexes)
                state = update_track_state(track_states, track_id, frame_index, timestamp, detection)
                stable_label = most_common_label(state)
                label = detection["label"] if detection else stable_label
                count_label = stable_label if stable_label != "unknown" else label
                confidence = detection["confidence"] if detection else 0.0
                center = bbox_center(track_box)
                previous_center = state["last_center"]
                if state["first_center"] is None:
                    state["first_center"] = center
                lane = assign_lane(center, lanes)
                direction = estimate_direction(
                    state["first_center"],
                    center,
                    args.direction_threshold,
                )
                speed_kmh = estimate_speed_kmh(
                    previous_center,
                    center,
                    speed_fps,
                    args.meters_per_pixel,
                )
                crossed_line = bool(
                    previous_center and intersect(previous_center, center, line_start, line_end)
                )
                is_vehicle = count_label in count_classes
                low_confidence_review = 0 < confidence < args.low_confidence_threshold

                if label != "unknown":
                    class_counts[label] += 1
                if lane:
                    lane_counts[lane] += 1
                    state["lane_counts"][lane] += 1
                if direction:
                    direction_counts[direction] += 1
                    state["direction_counts"][direction] += 1
                if speed_kmh is not None:
                    state["speed_sum"] += speed_kmh
                    state["speed_observations"] += 1
                    state["max_speed"] = max(state["max_speed"], speed_kmh)
                if low_confidence_review:
                    low_confidence_events += 1
                    state["low_confidence_observations"] += 1
                if is_vehicle and heatmap is not None:
                    cv2.circle(heatmap, center, 10, 1.0, -1)

                if crossed_line:
                    state["crossed"] = True
                if crossed_line and is_vehicle and not state["counted"]:
                    vehicle_count += 1
                    state["counted"] = True
                    if args.save_crops:
                        x1c, y1c = max(0, int(x1)), max(0, int(y1))
                        x2c, y2c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                        crop = frame[y1c:y2c, x1c:x2c]
                        if crop.size:
                            cv2.imwrite(str(crop_dir / f"track_{track_id}_frame_{frame_index}.jpg"), crop)

                color = [int(value) for value in colors[track_id % len(colors)]]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                speed_text = f" {speed_kmh:.1f}km/h" if speed_kmh is not None else ""
                draw_label(frame, f"ID {track_id} {label}{speed_text}", (int(x1), int(y1) - 6), color)

                if previous_center:
                    cv2.line(frame, previous_center, center, color, 2)

                if log_writer is not None:
                    log_writer.writerow(
                        {
                            "frame": frame_index,
                            "timestamp_sec": f"{timestamp:.3f}",
                            "track_id": track_id,
                            "label": label,
                            "is_counted_vehicle_class": is_vehicle,
                            "confidence": f"{confidence:.4f}",
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                        "center_x": center[0],
                        "center_y": center[1],
                        "lane": lane,
                        "direction": direction,
                        "speed_kmh": "" if speed_kmh is None else f"{speed_kmh:.2f}",
                        "low_confidence_review": low_confidence_review,
                        "crossed_count_line": crossed_line,
                        "vehicle_count_total": vehicle_count,
                    }
                )

                state["last_center"] = center

            processing_elapsed = max(time.perf_counter() - run_started_at, 1e-6)
            processing_fps = (frame_index + 1) / processing_elapsed

            if roi_polygon:
                cv2.polylines(frame, [np.asarray(roi_polygon, dtype=np.int32)], True, (255, 255, 0), 2)
            for lane_name, lane_polygon in lanes.items():
                cv2.polylines(frame, [np.asarray(lane_polygon, dtype=np.int32)], True, (255, 0, 255), 2)
                if lane_polygon:
                    cv2.putText(
                        frame,
                        lane_name,
                        lane_polygon[0],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2,
                    )
            cv2.line(frame, line_start, line_end, (0, 255, 255), 4)
            cv2.putText(
                frame,
                f"Vehicle count: {vehicle_count}",
                (20, 70),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                frame,
                f"Counting: {count_scope} | Tracking: {tracking_scope}",
                (20, 115),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Processing FPS: {processing_fps:.1f}",
                (20, 155),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            if not args.no_output and writer is None:
                fourcc = video_fourcc(args.output)
                writer = cv2.VideoWriter(
                    str(args.output),
                    fourcc,
                    args.output_fps,
                    (frame.shape[1], frame.shape[0]),
                    True,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {args.output}")

            if writer is not None:
                writer.write(frame)

            if frame_callback is not None:
                keep_running = frame_callback(
                    frame,
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp,
                        "vehicle_count": vehicle_count,
                        "processing_fps": processing_fps,
                        "run_dir": run_dir,
                    },
                )
                if keep_running is False:
                    break

            if args.display:
                cv2.imshow(args.window_name, frame)
                delay_ms = 1
                if args.realtime and not is_live_source and source_fps > 0:
                    delay_ms = max(1, int(1000 / source_fps))
                if cv2.waitKey(delay_ms) & 0xFF == quit_key:
                    log("[INFO] stopped from display window")
                    break

            previous_timestamp = timestamp
            frame_index += 1

        if summary_writer is not None:
            write_summary(summary_writer, track_states, count_classes)
            summary_written = True
    except KeyboardInterrupt:
        log("[INFO] interrupted by user")
    finally:
        if summary_writer is not None and not summary_written:
            write_summary(summary_writer, track_states, count_classes)
        if writer is not None:
            writer.release()
        video.release()
        if args.display:
            cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()
        if summary_file is not None:
            summary_file.close()

    heatmap_path = None
    heatmap_overlay_path = None
    if not args.no_heatmap:
        heatmap_path, heatmap_overlay_path = write_heatmap_artifacts(run_dir, heatmap, heatmap_base_frame)

    elapsed = time.perf_counter() - run_started_at
    result = {
        "run_dir": str(run_dir),
        "source": source_label,
        "processed_frames": frame_index,
        "final_vehicle_count": vehicle_count,
        "elapsed_seconds": round(elapsed, 2),
        "output_video": None if args.no_output else str(args.output),
        "tracking_log": None if args.no_logs else str(args.log),
        "track_summary": None if args.no_logs else str(args.summary_log),
        "heatmap": heatmap_path,
        "heatmap_overlay": heatmap_overlay_path,
        "vehicle_crops": str(crop_dir) if args.save_crops else None,
        "class_observations": dict(class_counts),
        "lane_observations": dict(lane_counts),
        "direction_observations": dict(direction_counts),
        "low_confidence_events": low_confidence_events,
        "roi_enabled": bool(roi_polygon),
        "lanes_enabled": bool(lanes),
        "source_fps": round(source_fps, 3),
        "is_live_source": is_live_source,
        "arguments": serialize_args(args),
    }
    result["html_report"] = write_html_report(run_dir, result)
    result["pdf_report"] = write_pdf_report(run_dir, result)
    metadata_path = write_run_metadata(run_dir, result)
    result["metadata"] = str(metadata_path)

    log("[INFO] cleaning up...")
    log(f"[INFO] processed frames: {frame_index}")
    log(f"[INFO] final vehicle count: {vehicle_count}")
    log(f"[INFO] run metadata: {metadata_path}")
    log(f"[INFO] output video: {'skipped' if args.no_output else args.output}")
    log(f"[INFO] frame log: {'skipped' if args.no_logs else args.log}")
    log(f"[INFO] track summary: {'skipped' if args.no_logs else args.summary_log}")
    log(f"[INFO] report: {result['html_report']}")
    log(f"[INFO] elapsed seconds: {elapsed:.2f}")
    return result


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
