import argparse
import csv
import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from main import run_pipeline


def parse_model_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) == 1:
        name = parts[0]
        if name == "yolov3":
            return name, "yolov3", "yolov3"
        return name, "ultralytics", name
    if len(parts) == 2:
        name, detector = parts
        model = "yolov3" if detector == "yolov3" else name
        return name, detector, model
    return parts[0], parts[1], parts[2]


def build_args(base_args, name, detector, model, comparison_dir):
    return Namespace(
        source=base_args.source,
        input=None,
        output_root=comparison_dir / "runs",
        run_dir=comparison_dir / "runs" / name,
        output=None,
        log=None,
        summary_log=None,
        yolo_dir=Path("yolo-coco"),
        detector=detector,
        model=model,
        tracker="sort",
        confidence=base_args.confidence,
        nms_threshold=base_args.nms_threshold,
        gamma=1.5,
        width=base_args.width,
        height=base_args.height,
        output_fps=15.0,
        max_age=3,
        min_hits=3,
        count_classes="car,motorbike,bus,truck",
        track_classes="all",
        line=base_args.line,
        meters_per_pixel=base_args.meters_per_pixel,
        calibration_pixels=None,
        calibration_meters=None,
        roi=base_args.roi,
        roi_file=base_args.roi_file,
        lanes_file=base_args.lanes_file,
        low_confidence_threshold=0.45,
        direction_threshold=12.0,
        no_heatmap=False,
        save_crops=False,
        max_frames=base_args.max_frames,
        display=False,
        realtime=False,
        no_output=True,
        no_logs=False,
        window_name="Vehicle Tracking",
        quit_key="q",
    )


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO detector backends on the same video.")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov3:yolov3:yolov3", "yolo11n:ultralytics:yolo11n.pt"],
        help="Model specs as name:detector:model. Example: yolov3:yolov3:yolov3 yolo11n:ultralytics:yolo11n.pt",
    )
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--height", type=int, default=416)
    parser.add_argument("--meters-per-pixel", type=float, default=0.05)
    parser.add_argument("--line", type=int, nargs=4, default=[100, 400, 1000, 400])
    parser.add_argument("--roi", default=None)
    parser.add_argument("--roi-file", type=Path, default=None)
    parser.add_argument("--lanes-file", type=Path, default=None)
    args = parser.parse_args()

    comparison_dir = Path("output/comparisons") / datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in args.models:
        name, detector, model = parse_model_spec(spec)
        print(f"[INFO] comparing {name} ({detector}, {model})")
        try:
            result = run_pipeline(build_args(args, name, detector, model, comparison_dir))
            rows.append(
                {
                    "name": name,
                    "detector": detector,
                    "model": model,
                    "processed_frames": result["processed_frames"],
                    "final_vehicle_count": result["final_vehicle_count"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "fps": round(result["processed_frames"] / max(result["elapsed_seconds"], 1e-6), 3),
                    "low_confidence_events": result["low_confidence_events"],
                    "run_dir": result["run_dir"],
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": name,
                    "detector": detector,
                    "model": model,
                    "processed_frames": 0,
                    "final_vehicle_count": "",
                    "elapsed_seconds": "",
                    "fps": "",
                    "low_confidence_events": "",
                    "run_dir": "",
                    "error": str(exc),
                }
            )

    csv_path = comparison_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = comparison_dir / "comparison.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[INFO] comparison CSV: {csv_path}")
    print(f"[INFO] comparison JSON: {json_path}")


if __name__ == "__main__":
    main()
