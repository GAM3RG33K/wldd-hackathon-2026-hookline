"""Hookline TRIBE Space: plain REST (POST video, GET report).

  POST /analyze  (multipart video) -> {job_id, status}
  GET  /report/{job_id}            -> {status} or full report.json
  GET  /report/{job_id}.png / .md -> chart / markdown
  GET  /health                     -> {ok, device, model_loaded}

Same report.json schema as tribe/hookline_report.py, plus
neural_source="tribev2-full (<device>)". Scores stay
heuristic, within-video normalised, uncalibrated.
"""
import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import hookline_report as HL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK = Path("/tmp/hookline_jobs")
WORK.mkdir(parents=True, exist_ok=True)
MODEL = None
GPU_LOCK = asyncio.Lock()
JOBS: dict = {}

app = FastAPI(title="Hookline TRIBE Space")


def _patch_fast_transcript():
    """In-process faster-whisper transcription (same patch as tribe/fast_transcript_patch.py).

    tribev2's default path shells out to `uvx whisperx`, which is brittle on
    Spaces. Patch once at startup; on marker mismatch log and keep default.
    Never crashes startup.
    """
    try:
        import tribev2.eventstransforms as et
        p = Path(et.__file__)
        s = p.read_text()
        if "_FW_MODEL" in s:
            print("[space] fast-transcript patch already present", flush=True)
            return
        marker = '        device = "cuda" if torch.cuda.is_available() else "cpu"\n        compute_type'
        if marker not in s:
            print("[space] WARN: eventstransforms marker not found, keeping whisperx path", flush=True)
            return
        fast = """        try:
            global _FW_MODEL
            from faster_whisper import WhisperModel
            if "_FW_MODEL" not in globals():
                _FW_MODEL = WhisperModel(os.environ.get("TRIBE_WHISPER_MODEL", "small"),
                                         device="cuda" if torch.cuda.is_available() else "cpu",
                                         compute_type="float16" if torch.cuda.is_available() else "int8")
            import subprocess as _sp, numpy as _np
            _raw = _sp.run(['ffmpeg','-loglevel','error','-i',str(wav_filename),'-f','f32le','-ac','1','-ar','16000','-'], capture_output=True, check=True).stdout
            _audio = _np.frombuffer(_raw, dtype=_np.float32)
            segs, _ = _FW_MODEL.transcribe(_audio, language=language_codes[language], word_timestamps=True)
            words = []
            for i, seg in enumerate(segs):
                for w in (seg.words or []):
                    words.append({"text": w.word.replace('"', "").strip(), "start": w.start,
                                  "duration": w.end - w.start, "sequence_id": i,
                                  "sentence": seg.text.replace('"', "")})
            return pd.DataFrame(words)
        except ImportError:
            pass
"""
        s = s.replace(marker, fast + marker, 1)
        if "\nimport os" not in s:
            s = "import os\n" + s
        p.write_text(s)
        print("[space] fast-transcript patch applied", flush=True)
    except Exception as e:  # noqa: BLE001 - transcription fallback must not kill boot
        print(f"[space] WARN: transcript patch skipped ({e})", flush=True)


@app.on_event("startup")
def _startup():
    _patch_fast_transcript()


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    from tribev2 import TribeModel

    t0 = time.time()
    overrides = {f"data.{m}.device": DEVICE for m in
                 ("text_feature", "audio_feature", "video_feature.image", "image_feature.image")}
    MODEL = TribeModel.from_pretrained("facebook/tribev2", device=DEVICE, config_update=overrides)
    print(f"[space] model ready on {DEVICE} in {time.time() - t0:.0f}s", flush=True)
    return MODEL


async def run_job(job_id: str, video: Path, source_name: str):
    job = WORK / job_id
    try:
        async with GPU_LOCK:  # one video on the GPU at a time
            model = load_model()
            t1 = time.time()
            events = model.get_events_dataframe(video_path=str(video))
            preds, segments = model.predict(events=events)
            preds = np.asarray(preds)
            print(f"[space] {job_id} infer {time.time() - t1:.0f}s shape={preds.shape}", flush=True)
            times = np.array([float(getattr(s, "start", i)) for i, s in enumerate(segments)])
            times = times - times.min() if len(times) else times
            np.save(job / "preds.npy", preds)
            np.save(job / "times.npy", times)
            tsv = video.with_suffix(".tsv")
            (job / "meta.json").write_text(json.dumps({
                "video": str(video), "source": source_name,
                "transcript": str(tsv) if tsv.exists() else None,
                "n_steps": int(preds.shape[0]), "space_device": DEVICE,
            }))
            report, d = HL.analyse(job, video)
            report["neural_source"] = f"tribev2-full ({DEVICE})"
            (job / "report.json").write_text(json.dumps(report, indent=1))
            HL.write_md(report, job / "report.md")
            HL.write_png(report, d, job / "report.png")
        JOBS[job_id] = {"status": "done"}
    except Exception as e:  # noqa: BLE001 - surfaced to the poller
        import traceback
        traceback.print_exc()
        print(f"[space] {job_id} failed: {e}", flush=True)
        JOBS[job_id] = {"status": "failed", "error": str(e)}


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model_loaded": MODEL is not None,
            "method": "heuristic, within-video normalised, uncalibrated",
            "jobs": len(JOBS)}


@app.post("/analyze")
async def analyze(background: BackgroundTasks, video: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    job = WORK / job_id
    job.mkdir(parents=True)
    ext = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = job / f"input{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    JOBS[job_id] = {"status": "queued"}
    background.add_task(run_job, job_id, dest, video.filename or dest.name)
    return {"job_id": job_id, "status": "queued"}


@app.get("/report/{job_id}")
def get_report(job_id: str):
    st = JOBS.get(job_id)
    p = WORK / job_id / "report.json"
    if p.exists():
        return json.loads(p.read_text())
    if st is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"job_id": job_id, **st}


@app.get("/report/{job_id}.md")
def get_md(job_id: str):
    p = WORK / job_id / "report.md"
    return FileResponse(p, media_type="text/markdown") if p.exists() \
        else JSONResponse({"error": "not ready"}, status_code=404)


@app.get("/report/{job_id}.png")
def get_png(job_id: str):
    p = WORK / job_id / "report.png"
    return FileResponse(p, media_type="image/png") if p.exists() \
        else JSONResponse({"error": "not ready"}, status_code=404)


@app.get("/")
def index():
    return {"service": "hookline-tribe", "docs": "/docs",
            "flow": "POST /analyze (video) -> GET /report/{job_id}"}
