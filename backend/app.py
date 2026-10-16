"""Phoenix v2 backend — TTCC retention analysis, local on this machine.

Endpoints:
  GET  /health                 – liveness + model env status
  POST /api/analyze            – upload a video (+caption); returns a job id
  GET  /api/report/{rid}       – job status while running, report when done
  GET  /api/reports            – list finished reports

Real TTCC inference runs in a background thread (checkpoint load takes a
couple of minutes on MPS). When the model environment is unavailable the
job falls back to a demo curve and is clearly marked `demo: true`.
Scores are heuristic, within-video normalised, uncalibrated.
"""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from phoenix import ttcc
from phoenix.scorer import analyse

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "backend" / "out"
SAMPLES_DIR = ROOT / "samples_v2"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

STAGES = [
    "Uploading video and caption",
    "Processing the video",
    "Analysing video content",
    "Analysing audio",
    "Matching script and caption to the video",
    "Building the retention graph",
    "Extracting emotional peaks",
    "Scoring against viral formulas",
]

app = FastAPI(title="Project Phoenix API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# model-load lock: TTCC loads the checkpoint per run; don't stack runs
INFER_LOCK = threading.Lock()


def _demo_curve(T: int, seed: float) -> list[float]:
    """A plausible-feeling retention curve for demo mode (no model run)."""
    rng = np.random.default_rng(int(seed * 1000) % 2**32)
    t = np.arange(T)
    base = 88 * np.exp(-0.035 * t) + 12
    wobble = np.sin(t / 3.0 + seed) * 3
    boosts = np.zeros(T)
    slots = list(range(max(T - 2, 1)))
    for k in rng.choice(slots, size=min(3, len(slots)), replace=False):
        boosts[k:k + 3] += rng.uniform(4, 9)
    curve = np.clip(base + wobble + boosts + rng.normal(0, 1.2, T), 4, 100)
    return [round(float(x) / 100, 3) for x in curve]


def _probe_duration(path: Path) -> float:
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=15)
        return float(r.stdout.strip())
    except Exception:
        return 28.0


def _write_status(job_dir: Path, status: str, detail: str = "", progress: int = 0,
                  stage_index: int = 0) -> None:
    (job_dir / "status.json").write_text(json.dumps({
        "status": status, "detail": detail, "progress": progress,
        "stage_index": stage_index, "stage_total": len(STAGES),
        "updated": time.time(),
    }))


def _run_job(rid: str, video: Path, caption: str, duration: float) -> None:
    """Background worker: run the analysis, publish user-facing progress per step."""
    job_dir = JOBS_DIR / rid
    t0 = time.time()

    def say(progress: int, detail: str, stage_index: int) -> None:
        _write_status(job_dir, "running", detail, progress, stage_index)

    say(2, "Reading your video…", 0)
    demo = False
    curve = None
    source = "demo"
    try:
        ok, why = ttcc.environment_ready()
        if not ok:
            demo = True
        else:
            # user-facing heartbeat while the analysis runs (one long step)
            done_evt = threading.Event()
            expected = 75.0 + 2.0 * max(duration, 8)  # generous seconds estimate
            def heartbeat() -> None:
                while not done_evt.wait(10):
                    el = time.time() - t0
                    pct = min(88, 6 + int(82 * el / expected))
                    detail = ("Analysing your reel second by second…"
                              if el < 90 else f"Still analysing… {int(el)}s elapsed")
                    # advance through the analysis stages with progress
                    stage = 2 + min(4, int((pct - 6) / 82 * 5))
                    say(pct, detail, stage)
            hb = threading.Thread(target=heartbeat, daemon=True)
            hb.start()
            try:
                with INFER_LOCK:
                    curve = ttcc.run_inference(video, job_dir, rid, duration)
                source = "ttcc-local"
            except Exception:
                traceback.print_exc()
                demo = True
            finally:
                done_evt.set()
                hb.join()
    except Exception:
        traceback.print_exc()
        demo = True

    if curve is None:
        curve = _demo_curve(max(int(math.ceil(duration)), 1), time.time() % 97)

    say(93, "Scoring the watch-through…", 7)
    try:
        report = analyse(curve, duration, video=str(video), caption=caption, demo=demo)
    except Exception:
        traceback.print_exc()
        _write_status(job_dir, "failed", "Something went wrong while scoring. Please try again.", 100, 7)
        return
    report["stages"] = STAGES
    report["source"] = source
    report["job_id"] = rid
    (job_dir / "report.json").write_text(json.dumps(report, indent=1))
    _write_status(job_dir, "done", "", 100, len(STAGES) - 1)


@app.get("/health")
def health():
    ok, why = ttcc.environment_ready()
    return {"status": "ok", "service": "project-phoenix", "version": 2,
            "model_env": "ready" if ok else f"unavailable: {why}"}


@app.post("/api/analyze")
async def analyze(
    video: Optional[UploadFile] = File(default=None),
    video_path: str = Form(default=""),
    caption: str = Form(default=""),
    sample: str = Form(default=""),
):
    """Queue an analysis. Returns the job id immediately; poll /api/report/{id}."""
    rid = f"job_{uuid.uuid4().hex[:8].upper()}"
    job_dir = JOBS_DIR / rid
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_status(job_dir, "queued")

    saved = ""
    if video is not None and video.filename:
        dest = job_dir / "video.mp4"
        with dest.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        saved = str(dest)
    elif video_path and Path(video_path).is_file():
        saved = video_path
    elif sample:
        matches = [p for p in SAMPLES_DIR.rglob("*.mp4") if sample.lower() in p.stem.lower()]
        if matches:
            saved = str(matches[0])
    if not saved:  # last resort: first sample reel
        fallback = sorted(SAMPLES_DIR.rglob("*.mp4"))
        if fallback:
            saved = str(fallback[0])
    (job_dir / "video_path.txt").write_text(saved or "")

    duration = _probe_duration(Path(saved)) if saved else 28.0
    thread = threading.Thread(target=_run_job, args=(rid, Path(saved or "demo.mp4"),
                                                     caption, duration), daemon=True)
    thread.start()
    return JSONResponse({"job_id": rid, "status": "queued",
                         "model_env": "ready" if ttcc.environment_ready()[0] else "unavailable"})


@app.get("/api/report/{rid}")
def get_report(rid: str):
    job_dir = JOBS_DIR / rid
    report = job_dir / "report.json"
    if report.is_file():
        return JSONResponse(json.loads(report.read_text()))
    status = job_dir / "status.json"
    if status.is_file():
        s = json.loads(status.read_text())
        return JSONResponse({
            "job_id": rid, "status": s["status"], "detail": s.get("detail", ""),
            "progress": s.get("progress", 0),
            "stage_index": s.get("stage_index", 0),
            "stage_total": s.get("stage_total", len(STAGES)),
            "stages": STAGES,
        })
    raise HTTPException(status_code=404, detail=f"no job {rid}")


@app.get("/api/media/{rid}")
def media(rid: str):
    """Stream the analysed video so the report page can play it inline."""
    job_dir = JOBS_DIR / rid
    path = Path((job_dir / "video_path.txt").read_text()) if (job_dir / "video_path.txt").is_file() else Path("")
    if not path.is_file():
        path = job_dir / "video.mp4"
    if not path.is_file() and (job_dir / "report.json").is_file():
        path = Path(json.loads((job_dir / "report.json").read_text()).get("video", ""))
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no video for {rid}")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/reports")
def list_reports():
    out = []
    for d in sorted(JOBS_DIR.iterdir(), reverse=True):
        rj = d / "report.json"
        if rj.is_file():
            r = json.loads(rj.read_text())
            out.append({
                "job_id": r.get("job_id", d.name),
                "video": Path(r.get("video", "")).name,
                "duration_s": r.get("duration_s"),
                "virality": r.get("scores", {}).get("virality"),
                "virality_tag": r.get("scores", {}).get("virality_tag"),
                "demo": r.get("demo", False),
            })
    return out
