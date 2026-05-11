import shutil
import threading
import uuid
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from main import run_pipeline


app = FastAPI(title="Vehicle Counting API")
JOBS = {}


def build_job_args(source_path, run_dir, max_frames=None, detector="yolov3", model="yolo11n.pt"):
    return Namespace(
        source=str(source_path),
        input=None,
        output_root=Path("output/runs"),
        run_dir=Path(run_dir),
        output=None,
        log=None,
        summary_log=None,
        yolo_dir=Path("yolo-coco"),
        detector=detector,
        model=model,
        tracker="sort",
        confidence=0.35,
        nms_threshold=0.25,
        gamma=1.5,
        width=416,
        height=416,
        output_fps=15.0,
        max_age=3,
        min_hits=3,
        count_classes="car,motorbike,bus,truck",
        track_classes="all",
        line=[100, 400, 1000, 400],
        meters_per_pixel=0.05,
        calibration_pixels=None,
        calibration_meters=None,
        roi=None,
        roi_file=None,
        lanes_file=None,
        low_confidence_threshold=0.45,
        direction_threshold=12.0,
        no_heatmap=False,
        save_crops=False,
        max_frames=max_frames,
        display=False,
        realtime=False,
        no_output=False,
        no_logs=False,
        window_name="Vehicle Tracking",
        quit_key="q",
    )


def run_job(job_id, source_path, run_dir, max_frames, detector, model):
    JOBS[job_id]["status"] = "running"
    try:
        args = build_job_args(source_path, run_dir, max_frames, detector, model)
        result = run_pipeline(args, print_logs=False)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result
    except Exception as exc:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)


@app.post("/runs")
async def create_run(
    video: UploadFile = File(...),
    max_frames: int | None = None,
    detector: str = "yolov3",
    model: str = "yolo11n.pt",
):
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    upload_dir = Path("output/uploads") / job_id
    run_dir = Path("output/runs") / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    source_path = upload_dir / video.filename

    with source_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "source": str(source_path),
        "run_dir": str(run_dir),
        "result": None,
        "error": None,
    }
    threading.Thread(
        target=run_job,
        args=(job_id, source_path, run_dir, max_frames, detector, model),
        daemon=True,
    ).start()
    return JOBS[job_id]


@app.get("/runs")
def list_runs():
    run_root = Path("output/runs")
    runs = []
    if run_root.exists():
        for run_dir in sorted(run_root.iterdir(), reverse=True):
            metadata = run_dir / "run_metadata.json"
            runs.append(
                {
                    "run_dir": str(run_dir),
                    "metadata": str(metadata) if metadata.exists() else None,
                }
            )
    return {"jobs": list(JOBS.values()), "run_folders": runs}


@app.get("/runs/{job_id}")
def get_run(job_id: str):
    return JOBS.get(job_id, {"error": "unknown job"})


@app.get("/runs/{job_id}/artifacts/{artifact_name}")
def get_artifact(job_id: str, artifact_name: str):
    run_dir = Path("output/runs") / job_id
    artifact_path = run_dir / artifact_name
    if not artifact_path.exists():
        return {"error": "artifact not found"}
    return FileResponse(str(artifact_path))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
