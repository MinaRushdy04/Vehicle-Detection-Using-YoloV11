import queue
import threading
from argparse import Namespace
from pathlib import Path

import cv2
import gradio as gr

from main import run_pipeline


def build_args(
    video_path,
    detector,
    model,
    tracker,
    confidence,
    nms_threshold,
    width,
    height,
    line_x1,
    line_y1,
    line_x2,
    line_y2,
    meters_per_pixel,
    max_frames,
):
    return Namespace(
        source=str(video_path),
        input=None,
        output_root=Path("output/runs"),
        run_dir=None,
        output=None,
        log=None,
        summary_log=None,
        yolo_dir=Path("yolo-coco"),
        detector=detector,
        model=model,
        tracker=tracker,
        confidence=float(confidence),
        nms_threshold=float(nms_threshold),
        gamma=1.5,
        width=int(width),
        height=int(height),
        output_fps=15.0,
        max_age=3,
        min_hits=3,
        count_classes="car,motorbike,bus,truck",
        track_classes="all",
        line=[int(line_x1), int(line_y1), int(line_x2), int(line_y2)],
        meters_per_pixel=float(meters_per_pixel),
        calibration_pixels=None,
        calibration_meters=None,
        roi=None,
        roi_file=None,
        lanes_file=None,
        low_confidence_threshold=0.45,
        direction_threshold=12.0,
        no_heatmap=False,
        save_crops=False,
        max_frames=None if not max_frames else int(max_frames),
        display=False,
        realtime=False,
        no_output=False,
        no_logs=False,
        window_name="Vehicle Tracking",
        quit_key="q",
    )


def process_upload(
    video_path,
    detector,
    model,
    tracker,
    confidence,
    nms_threshold,
    width,
    height,
    preview_every,
    line_x1,
    line_y1,
    line_x2,
    line_y2,
    meters_per_pixel,
    max_frames,
):
    if not video_path:
        yield None, "Upload a video first.", None, None, None, None, None, None, None
        return

    events = queue.Queue()
    last_preview = None
    args = build_args(
        video_path,
        detector,
        model,
        tracker,
        confidence,
        nms_threshold,
        width,
        height,
        line_x1,
        line_y1,
        line_x2,
        line_y2,
        meters_per_pixel,
        max_frames,
    )
    preview_every = max(1, int(preview_every))

    def on_frame(frame, metrics):
        if metrics["frame_index"] % preview_every != 0:
            return None
        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        status = (
            f"Run: `{metrics['run_dir']}`  \n"
            f"Frame: `{metrics['frame_index']}`  \n"
            f"Vehicle count: `{metrics['vehicle_count']}`  \n"
            f"Processing FPS: `{metrics['processing_fps']:.1f}`"
        )
        events.put(("frame", preview, status))
        return None

    def worker():
        try:
            result = run_pipeline(args, frame_callback=on_frame, print_logs=False)
            events.put(("done", result))
        except Exception as exc:
            events.put(("error", str(exc)))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event = events.get()
        if event[0] == "frame":
            _, last_preview, status = event
            yield last_preview, status, None, None, None, None, None, None, None
        elif event[0] == "done":
            result = event[1]
            status = (
                f"Done.  \n"
                f"Run: `{result['run_dir']}`  \n"
                f"Frames: `{result['processed_frames']}`  \n"
                f"Final vehicle count: `{result['final_vehicle_count']}`  \n"
                f"Elapsed seconds: `{result['elapsed_seconds']}`"
            )
            yield (
                last_preview,
                status,
                result["output_video"],
                result["tracking_log"],
                result["track_summary"],
                result["metadata"],
                result["html_report"],
                result["pdf_report"],
                result["heatmap_overlay"],
            )
            break
        else:
            yield last_preview, f"Error: {event[1]}", None, None, None, None, None, None, None
            break


def list_runs():
    root = Path("output/runs")
    if not root.exists():
        return []
    return [path.name for path in sorted(root.iterdir(), reverse=True) if path.is_dir()]


def refresh_runs():
    return gr.update(choices=list_runs())


def load_run(run_name):
    if not run_name:
        return "Select a run.", None, None, None, None, None, None
    run_dir = Path("output/runs") / run_name
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return f"No metadata found for `{run_name}`.", None, None, None, None, None, None
    status = metadata_path.read_text(encoding="utf-8")
    return (
        f"Loaded `{run_name}`\n\n```json\n{status}\n```",
        str(run_dir / "output_video.mp4") if (run_dir / "output_video.mp4").exists() else None,
        str(run_dir / "tracking_log.csv") if (run_dir / "tracking_log.csv").exists() else None,
        str(run_dir / "track_summary.csv") if (run_dir / "track_summary.csv").exists() else None,
        str(metadata_path),
        str(run_dir / "report.html") if (run_dir / "report.html").exists() else None,
        str(run_dir / "detection_heatmap_overlay.png") if (run_dir / "detection_heatmap_overlay.png").exists() else None,
    )


with gr.Blocks(title="Vehicle Counting") as demo:
    gr.Markdown("# Vehicle Counting")

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Input Video", sources=["upload"])
            run_button = gr.Button("Process Video", variant="primary")

            with gr.Accordion("Detection", open=True):
                detector = gr.Dropdown(["yolov3", "ultralytics"], value="yolov3", label="Detector")
                model = gr.Textbox(value="yolo11n.pt", label="Ultralytics Model")
                tracker = gr.Dropdown(["sort", "bytetrack"], value="sort", label="Tracker")
                confidence = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confidence")
                nms_threshold = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="NMS Threshold")
                width = gr.Slider(256, 608, value=416, step=32, label="YOLO Width")
                height = gr.Slider(256, 608, value=416, step=32, label="YOLO Height")

            with gr.Accordion("Counting Line", open=False):
                line_x1 = gr.Number(value=100, label="X1", precision=0)
                line_y1 = gr.Number(value=400, label="Y1", precision=0)
                line_x2 = gr.Number(value=1000, label="X2", precision=0)
                line_y2 = gr.Number(value=400, label="Y2", precision=0)

            with gr.Accordion("Run", open=False):
                meters_per_pixel = gr.Number(value=0.05, label="Meters Per Pixel")
                preview_every = gr.Slider(1, 30, value=5, step=1, label="Preview Every N Frames")
                max_frames = gr.Number(value=0, label="Max Frames", precision=0)

        with gr.Column(scale=2):
            preview = gr.Image(label="Live Preview", type="numpy")
            status = gr.Markdown()

            with gr.Row():
                final_video = gr.Video(label="Output Video")
                tracking_log = gr.File(label="Tracking Log")
                summary_log = gr.File(label="Track Summary")
                metadata = gr.File(label="Run Metadata")
                report = gr.File(label="HTML Report")
                pdf_report = gr.File(label="PDF Report")
                heatmap = gr.Image(label="Heatmap Overlay")

    with gr.Accordion("Previous Runs", open=False):
        with gr.Row():
            previous_runs = gr.Dropdown(choices=list_runs(), label="Run Folder")
            refresh_button = gr.Button("Refresh Runs")
            load_button = gr.Button("Load Run")
        previous_status = gr.Markdown()
        with gr.Row():
            previous_video = gr.Video(label="Previous Output Video")
            previous_log = gr.File(label="Previous Tracking Log")
            previous_summary = gr.File(label="Previous Track Summary")
            previous_metadata = gr.File(label="Previous Metadata")
            previous_report = gr.File(label="Previous HTML Report")
            previous_heatmap = gr.Image(label="Previous Heatmap")

    run_button.click(
        process_upload,
        inputs=[
            video,
            detector,
            model,
            tracker,
            confidence,
            nms_threshold,
            width,
            height,
            preview_every,
            line_x1,
            line_y1,
            line_x2,
            line_y2,
            meters_per_pixel,
            max_frames,
        ],
        outputs=[preview, status, final_video, tracking_log, summary_log, metadata, report, pdf_report, heatmap],
    )
    refresh_button.click(refresh_runs, outputs=previous_runs)
    load_button.click(
        load_run,
        inputs=previous_runs,
        outputs=[
            previous_status,
            previous_video,
            previous_log,
            previous_summary,
            previous_metadata,
            previous_report,
            previous_heatmap,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
