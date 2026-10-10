"""Hookline TRIBE Space: plain REST (POST video, GET report).

  POST /analyze  (multipart video) -> {job_id, status}
  GET  /report/{job_id}            -> progress (stage, step/steps, elapsed_s, eta_s,
                                      timeout_s, log tail, mem) or full report.json
  GET  /report/{job_id}.png / .md -> chart / markdown
  POST /jobs/{job_id}/cancel       -> stop at the next checkpoint (model stays loaded)
  POST /jobs/{job_id}/restart      -> stop if running, re-run the same upload as a new job
  GET  /jobs                       -> recent jobs, newest first
  GET  /jobs/{job_id}              -> progress, timings, chunks and log (also after done)
  GET  /health                     -> {ok, device, model_loaded}

Same report.json schema as tribe/hookline_report.py, plus
neural_source="tribev2-full (<device>)". Scores stay
heuristic, within-video normalised, uncalibrated.
"""
import asyncio
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
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
GPU_LOCK = threading.Lock()  # held in a worker thread, never on the event loop
JOBS: dict = {}
BOOT: dict = {}  # boot import check result, surfaced on /health (run logs are unreliable)

# Pipeline steps with a rough cost model: seconds = fixed + per_video_second * duration.
# Priors from the 28s sneaker run on t4-small (job df75072d): events 111s, predict
# text 270s + audio 8s + video >800s. SCALE learns actual/estimate per step.
STEPS = [
    ("load_model", "Loading TRIBE weights", 120.0, 0.0),
    ("events", "Transcribing speech + extracting audio/visual events", 15.0, 3.5),
    ("predict", "Running TRIBE brain encoder", 30.0, 40.0),
    ("report", "Scoring + writing report", 5.0, 0.2),
]
AFTER_VIDEO_S = 30.0  # loaders + inference after the last video chunk (rough)
SCALE = {k: 1.0 for k, *_ in STEPS}
LOG_TAIL = 60
HEARTBEAT_S = 20
# Audio/video are split into chunks of at most this many seconds before feature
# extraction. tribev2 defaults to 30-60s chunks, which put a whole reel through
# Wav2VecBert attention at once and ran a T4 out of memory on a 28s video.
CHUNK_S = float(os.environ.get("HOOKLINE_CHUNK_S", "10"))
CURRENT = {"job": None}       # job holding the GPU lock (one at a time)
_TLS = threading.local()      # nesting guard for timed extractor.prepare()


class JobCancelled(Exception):
    """Raised at a checkpoint when a stop was requested for the running job."""


def check_cancel(job_id=None):
    """Checkpoint: every step, sub-step and loop item calls this. Threads can't be killed,
    so stopping is cooperative; the worst case waits out one loop item."""
    job_id = job_id or CURRENT["job"]
    if job_id and JOBS.get(job_id, {}).get("cancel"):
        raise JobCancelled(f"stopped by request in {JOBS[job_id].get('stage')}")

app = FastAPI(title="Hookline TRIBE Space")


def _mem():
    """Host RAM available, process RSS and GPU memory in GB (for diagnosing stalls)."""
    out = {}
    try:
        info = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
        out["ram_avail_gb"] = round(int(info["MemAvailable"].split()[0]) / 1e6, 1)
        out["ram_total_gb"] = round(int(info["MemTotal"].split()[0]) / 1e6, 1)
        st = dict(l.split(":", 1) for l in Path("/proc/self/status").read_text().splitlines() if ":" in l)
        out["rss_gb"] = round(int(st["VmRSS"].split()[0]) / 1e6, 1)
    except Exception:  # noqa: BLE001 - not on Linux
        pass
    if torch.cuda.is_available():
        out["gpu_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    return out


def _video_seconds(path: Path) -> float:
    import subprocess
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:  # noqa: BLE001 - fall back to a typical reel length
        return 30.0


def _estimate(key: str, video_s: float) -> float:
    if key == "load_model" and MODEL is not None:
        return 0.0
    _, _, fixed, per_s = next(s for s in STEPS if s[0] == key)
    return (fixed + per_s * video_s) * SCALE[key]


def jlog(job_id: str, msg: str):
    """One log line to stdout (Space logs) and to the job's tail (served by the API)."""
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(f"[space] {job_id} {msg}", flush=True)
    job = JOBS.get(job_id)
    if job is not None:
        job.setdefault("log", []).append(line)
        del job["log"][:-LOG_TAIL]


def set_step(job_id: str, key: str):
    check_cancel(job_id)
    job = JOBS[job_id]
    now = time.time()
    prev = job.get("stage")
    if prev and prev in job["plan"] and job["plan"][prev] > 0:
        took = now - job["stage_started"]
        job.setdefault("took", {})[prev] = round(took)
        learned = 0.5 * SCALE[prev] + 0.5 * SCALE[prev] * took / job["plan"][prev]
        SCALE[prev] = min(4.0, max(0.25, learned))
    idx = [s[0] for s in STEPS].index(key)
    job.update(status="running", stage=key, stage_label=STEPS[idx][1], step=idx + 1,
               steps=len(STEPS), stage_started=now)
    eta = progress(job_id).get("eta_s")
    jlog(job_id, f"step {idx + 1}/{len(STEPS)} {key}: {STEPS[idx][1]} "
                 f"(est {job['plan'][key]:.0f}s, total eta {'unknown' if eta is None else f'{eta}s'})")


def _eta(job: dict, now: float):
    """Seconds left for the running job -> (eta or None, basis, overrun_s).

    Live progress wins over the plan: during video features the ETA is
    (current chunk left) + (chunks left x avg chunk time); other loops use their rate.
    Past the plan with no live signal the ETA is unknown (None), never a fake floor."""
    keys = [s[0] for s in STEPS]
    stage = job["stage"]
    later = sum(job["plan"][k] for k in keys[keys.index(stage) + 1:])
    in_stage = now - job["stage_started"]
    plan_left = job["plan"][stage] - in_stage
    loops = list(job.get("loops", {}).values())
    vc = job.get("video_chunk") or {}
    vloop = next((l for l in loops if l["name"].startswith("video chunk")), None)
    if vloop and vc.get("n"):
        done = job.get("chunks_done", [])
        if done:
            per_chunk = sum(c["took_s"] for c in done) / len(done)
        elif vloop.get("per_item_s") and vloop.get("total"):
            per_chunk = vloop["per_item_s"] * vloop["total"]
        else:
            per_chunk = None
        if per_chunk:
            cur_left = vloop["eta_s"] if vloop.get("eta_s") is not None else per_chunk
            left = cur_left + max(0, vc["n"] - vc["i"]) * per_chunk + AFTER_VIDEO_S
            return left + later, f"video chunk {vc['i']}/{vc['n']}", None
    live = [l["eta_s"] for l in loops if l.get("eta_s") is not None]
    if live:
        return max(max(live), plan_left) + later, "loop rate", None
    if plan_left > 0:
        return plan_left + later, "plan", None
    return None, "over plan, no live signal", -plan_left


def progress(job_id: str) -> dict:
    """Public progress view: current step, elapsed, ETA, timeout, recent log lines."""
    job = JOBS[job_id]
    now = time.time()
    out = {k: job[k] for k in ("status", "stage", "stage_label", "step", "steps", "video_s",
                               "error", "trace", "log", "took") if k in job}
    if "started" in job:
        out["elapsed_s"] = round(now - job["started"])
    if job.get("status") == "running":
        in_stage = now - job["stage_started"]
        eta, basis, overrun = _eta(job, now)
        out["stage_elapsed_s"] = round(in_stage)
        out["eta_s"] = None if eta is None else round(eta)
        out["eta_basis"] = basis
        if overrun:
            out["overrun_s"] = round(overrun)
        out["overdue"] = out["elapsed_s"] > job["timeout_s"]
    out["timeout_s"] = job.get("timeout_s")
    out["plan"] = [{"key": k, "label": label, "est_s": round(job.get("plan", {}).get(k, _estimate(k, job["video_s"]))),
                    "took_s": job.get("took", {}).get(k)} for k, label, *_ in STEPS]
    sub = job.get("substep")
    if sub:
        out["substep"] = {"name": sub["name"], "elapsed_s": round(now - sub["started"])}
    out["substeps"] = job.get("substeps", [])
    out["loops"] = list(job.get("loops", {}).values())
    out["chunks"] = job.get("chunks", [])
    out["video_chunk"] = job.get("video_chunk")
    out["chunks_done"] = job.get("chunks_done", [])
    out["heartbeat"] = job.get("heartbeat")
    out["mem"] = _mem()
    return out


@contextmanager
def timed(step: str, name: str):
    """Time one sub-step of the running job; published via /report progress and logs."""
    check_cancel()
    job_id = CURRENT["job"]
    t0 = time.time()
    if job_id:
        JOBS[job_id]["substep"] = {"name": f"{step}/{name}", "started": t0}
        jlog(job_id, f"  > {step}/{name}")
    try:
        yield
    finally:
        took = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if job_id:
            j = JOBS[job_id]
            j.setdefault("substeps", []).append({"step": step, "name": name, "took_s": round(took, 1)})
            j["substep"] = None
            jlog(job_id, f"  < {step}/{name} {took:.1f}s, mem {_mem()}")


def _timed_audio_text_events(events, audio_only: bool = False):
    """Copy of tribev2.demo_utils.get_audio_and_text_events with per-transform timing
    and CHUNK_S chunking (OOM fix)."""
    import tribev2.demo_utils as du
    transforms = [
        ("extract_audio", du.ExtractAudioFromVideo()),
        ("chunk_audio", du.ChunkEvents(event_type_to_chunk="Audio", max_duration=CHUNK_S, min_duration=CHUNK_S / 2)),
        ("chunk_video", du.ChunkEvents(event_type_to_chunk="Video", max_duration=CHUNK_S, min_duration=CHUNK_S / 2)),
    ]
    if not audio_only:
        transforms += [
            ("transcribe", du.ExtractWordsFromAudio()),
            ("add_text", du.AddText()),
            ("sentences", du.AddSentenceToWords(max_unmatched_ratio=0.05)),
            ("word_context", du.AddContextToWords(sentence_only=False, max_context_len=1024, split_field="")),
            ("remove_missing", du.RemoveMissing()),
        ]
    events = du.standardize_events(events)
    for name, t in transforms:
        with timed("events", name):
            events = t(events)
        if name == "chunk_video":
            _publish_chunks(events)
    return du.standardize_events(events)


# Friendly names for the tqdm loops inside tribev2/neuralset.
LOOP_NAMES = {
    "Computing video latents": "video chunks (image model)",  # one item per chunk
    "Encoding video": "video chunk",  # V-JEPA path: one loop per chunk, one item per frame step
    "Computing word embeddings": "text batches",
    "Extracting words from audio": "audio files transcribed",
    None: "inference batches",
}


def _make_progress_tqdm(base):
    """tqdm subclass that publishes each loop's n/total, rate and ETA on the running job."""

    class ProgressTqdm(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._hl_job, self._hl_t0, self._hl_n, self._hl_last = CURRENT["job"], time.time(), 0, 0.0
            desc = getattr(self, "desc", None) or None  # disabled bars skip tqdm's attribute setup
            self._hl_name = LOOP_NAMES.get(desc, desc or "loop")
            self._hl_chunk = None
            if desc == "Encoding video":  # neuralset loops chunks without a bar; count them here
                self._hl_chunk = _next_video_chunk(self._hl_job)
                if self._hl_chunk:
                    self._hl_name = self._hl_chunk["label"]
            self._hl_publish(force=True)

        def __iter__(self):  # count items ourselves: works even when the bar is disabled
            for i, item in enumerate(super().__iter__(), 1):
                yield item
                self._hl_n = i
                self._hl_publish()
                check_cancel(self._hl_job)

        def update(self, n=1):  # manual-update loops; tqdm's __iter__ also lands here
            r = super().update(n)
            self._hl_n = max(self._hl_n, self.n)
            self._hl_publish()
            check_cancel(self._hl_job)
            return r

        def _hl_publish(self, force=False):
            try:
                self._hl_publish_unsafe(force)
            except Exception:  # noqa: BLE001 - progress must never break inference
                pass

        def _hl_publish_unsafe(self, force):
            job = JOBS.get(self._hl_job)
            now = time.time()
            if job is None or (not force and now - self._hl_last < 1.0):
                return
            self._hl_last = now
            el = now - self._hl_t0
            rate = self._hl_n / el if el > 0 and self._hl_n else None
            total = getattr(self, "total", None)
            job.setdefault("loops", {})[id(self)] = {
                "name": self._hl_name, "n": self._hl_n, "total": total, "elapsed_s": round(el, 1),
                "per_item_s": round(1 / rate, 2) if rate else None,
                "eta_s": round((total - self._hl_n) / rate) if rate and total else None}
            if force:
                jlog(self._hl_job, f"    loop {self._hl_name}: 0/{total if total is not None else '?'}")

        def close(self):
            try:
                job = JOBS.get(getattr(self, "_hl_job", None))
                if job is not None and job.get("loops", {}).pop(id(self), None) is not None:
                    el = time.time() - self._hl_t0
                    avg = f", {el / self._hl_n:.2f}s each" if self._hl_n else ""
                    jlog(self._hl_job, f"    loop {self._hl_name}: {self._hl_n}/"
                                       f"{getattr(self, 'total', None)} in {el:.1f}s{avg}")
                    if self._hl_chunk:
                        job.setdefault("chunks_done", []).append(
                            {**self._hl_chunk, "took_s": round(el, 1), "frames": self._hl_n})
            except Exception:  # noqa: BLE001 - progress must never break inference
                pass
            super().close()

    return ProgressTqdm


def _next_video_chunk(job_id):
    """Advance the running job's video-chunk counter; returns {i, n, span, label}."""
    job = JOBS.get(job_id)
    if job is None:
        return None
    spans = [c for c in job.get("chunks", []) if c["type"] == "Video"]
    i = job.get("video_chunk_i", 0) + 1
    job["video_chunk_i"] = i
    n = len(spans) or None
    span = spans[i - 1] if spans and i <= len(spans) else None
    span_txt = f" {span['start']:.0f}-{span['start'] + span['duration']:.0f}s" if span else ""
    label = f"video chunk {i}/{n if n else '?'}{span_txt}"
    job["video_chunk"] = {"i": i, "n": n, "span": span}
    jlog(job_id, f"  {label}: encoding frames")
    return {"i": i, "n": n, "span": span, "label": label}


def _install_progress_loops():
    """Swap every module-level reference to tqdm.tqdm for the publishing subclass."""
    import sys
    import tqdm as tq
    orig = tq.std.tqdm
    sub = _make_progress_tqdm(orig)
    n = 0
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "tqdm", None) is orig:
            try:
                setattr(mod, "tqdm", sub)
                n += 1
            except Exception:  # noqa: BLE001 - read-only module, skip
                pass
    print(f"[space] progress loops on {n} modules", flush=True)


def _publish_chunks(events):
    """Record how Audio/Video were split so the API/TUI can show per-chunk progress."""
    job_id = CURRENT["job"]
    if not job_id:
        return
    chunks = [{"type": r.type, "start": round(float(r.start), 1), "duration": round(float(r.duration), 1)}
              for r in events.itertuples() if r.type in ("Audio", "Video")]
    JOBS[job_id]["chunks"] = chunks
    for kind in ("Audio", "Video"):
        spans = [f"{c['start']:.0f}-{c['start'] + c['duration']:.0f}s" for c in chunks if c["type"] == kind]
        if spans:
            jlog(job_id, f"  {kind.lower()} split into {len(spans)} chunks: {', '.join(spans)}")


def _install_timing_hooks():
    """Patch tribev2/neuralset so each event transform and feature extractor is timed."""
    import tribev2.demo_utils as du
    from neuralset.extractors.base import BaseExtractor
    du.get_audio_and_text_events = _timed_audio_text_events

    def wrap(cls):
        orig = cls.__dict__["prepare"]

        def prepare(self, *a, **k):
            if getattr(_TLS, "depth", 0):  # inner super().prepare: time only the outer call
                return orig(self, *a, **k)
            _TLS.depth = 1
            try:
                with timed("predict", f"features:{type(self).__name__}"):
                    return orig(self, *a, **k)
            finally:
                _TLS.depth = 0
        type.__setattr__(cls, "prepare", prepare)

    seen, todo, n = set(), [BaseExtractor], 0
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        todo.extend(cls.__subclasses__())
        if "prepare" in cls.__dict__:
            try:
                wrap(cls)
                n += 1
            except Exception as e:  # noqa: BLE001 - timing must never break inference
                print(f"[space] WARN: cannot time {cls.__name__}.prepare ({e})", flush=True)
    print(f"[space] timing hooks on {n} extractor classes, chunk {CHUNK_S}s", flush=True)


def timing_summary(job_id: str) -> str:
    subs = sorted(JOBS[job_id].get("substeps", []), key=lambda x: -x["took_s"])
    total = sum(x["took_s"] for x in subs) or 1
    return ", ".join(f"{x['step']}/{x['name']} {x['took_s']:.0f}s ({100 * x['took_s'] / total:.0f}%)"
                     for x in subs[:8])


def _heartbeat(job_id: str, stop: threading.Event):
    while not stop.wait(HEARTBEAT_S):
        p = progress(job_id)
        parts = [f"heartbeat {p.get('stage')} {p.get('stage_elapsed_s')}s in step"]
        sub = p.get("substep")
        if sub:
            parts.append(f"{sub['name']} {sub['elapsed_s']}s")
        for loop in p.get("loops", []):
            parts.append(f"{loop['name']} {loop['n']}/{loop['total']}")
        eta = p.get("eta_s")
        eta_txt = f"eta {eta}s ({p.get('eta_basis')})" if eta is not None else \
            f"eta unknown, step over plan by {p.get('overrun_s')}s"
        parts.append(f"elapsed {p.get('elapsed_s')}s, {eta_txt}, mem {p['mem']}")
        line = ", ".join(parts)
        print(f"[space] {job_id} {line}", flush=True)
        JOBS[job_id]["heartbeat"] = f"{time.strftime('%H:%M:%S')} {line}"  # latest only, keeps log tail useful


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
                _FW_MODEL = WhisperModel(os.environ.get("TRIBE_WHISPER_MODEL", "base"),
                                         device="cuda" if torch.cuda.is_available() else "cpu",
                                         compute_type="int8_float16" if torch.cuda.is_available() else "int8")
            import subprocess as _sp, numpy as _np
            _raw = _sp.run(['ffmpeg','-loglevel','error','-i',str(wav_filename),'-f','f32le','-ac','1','-ar','16000','-'], capture_output=True, check=True).stdout
            _audio = _np.frombuffer(_raw, dtype=_np.float32)
            import time as _time
            _t0 = _time.time()
            segs, _ = _FW_MODEL.transcribe(_audio, language=language_codes[language], word_timestamps=True,
                                           beam_size=1, vad_filter=True)
            segs = list(segs)
            print(f"[space] whisper {os.environ.get('TRIBE_WHISPER_MODEL', 'base')}: "
                  f"{len(_audio) / 16000:.1f}s audio in {_time.time() - _t0:.1f}s", flush=True)
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
    try:
        # Import torchvision fully on the main thread so worker-thread imports
        # never see a partially initialised module.
        import torch as _t, torchaudio as _ta, torchvision as _tv, transformers as _tr
        import torchvision.ops, torchvision.transforms, torchvision.io  # noqa: F401
        from transformers import AutoModel  # noqa: F401 - fails if transformers/torch skew
        BOOT.update(ok=True, torch=_t.__version__, ta=_ta.__version__, tv=_tv.__version__,
                    transformers=_tr.__version__, cuda=_t.cuda.is_available(),
                    tv_ops=bool(_tv.extension._has_ops()))
        print(f"[space] boot imports OK: {BOOT}", flush=True)
    except Exception as e:  # noqa: BLE001 - surfaced on /health, fail fast
        import traceback
        BOOT.update(ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])
        print(f"[space] BOOT IMPORT FAILED: {e}", flush=True)


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    from tribev2 import TribeModel

    t0 = time.time()
    overrides = {f"data.{m}.device": DEVICE for m in
                 ("text_feature", "audio_feature", "video_feature.image", "image_feature.image")}
    MODEL = TribeModel.from_pretrained("facebook/tribev2", device=DEVICE, config_update=overrides)
    for hook in (_install_timing_hooks, _install_progress_loops):
        try:
            hook()
        except Exception as e:  # noqa: BLE001 - progress must never break inference
            print(f"[space] WARN: {hook.__name__} skipped ({e})", flush=True)
    print(f"[space] model ready on {DEVICE} in {time.time() - t0:.0f}s", flush=True)
    return MODEL


def _run_job_sync(job_id: str, video: Path, source_name: str):
    """Blocking GPU work. Runs in a worker thread so /health + polls stay live."""
    job = WORK / job_id
    with GPU_LOCK:  # one video on the GPU at a time
        check_cancel(job_id)  # stopped while waiting in the queue
        j = JOBS[job_id]
        j["plan"] = {k: _estimate(k, j["video_s"]) for k, *_ in STEPS}
        j["timeout_s"] = round(2 * sum(j["plan"].values()) + 120)
        j["started"] = time.time()
        jlog(job_id, f"started {source_name} ({j['video_s']:.1f}s video), "
                     f"estimate {sum(j['plan'].values()):.0f}s, timeout {j['timeout_s']}s, mem {_mem()}")
        CURRENT["job"] = job_id
        stop = threading.Event()
        threading.Thread(target=_heartbeat, args=(job_id, stop), daemon=True).start()
        try:
            set_step(job_id, "load_model")
            model = load_model()
            set_step(job_id, "events")
            events = model.get_events_dataframe(video_path=str(video))
            try:
                counts = events.type.value_counts().to_dict()
            except Exception:  # noqa: BLE001 - logging only
                counts = {}
            jlog(job_id, f"events ready: {len(events)} rows {counts}, mem {_mem()}")
            set_step(job_id, "predict")
            t1 = time.time()
            n_before = len(j.get("substeps", []))
            preds, segments = model.predict(events=events)
            preds = np.asarray(preds)
            took = time.time() - t1
            feats = sum(x["took_s"] for x in j.get("substeps", [])[n_before:])
            j.setdefault("substeps", []).append({"step": "predict", "name": "loaders+inference",
                                                 "took_s": round(took - feats, 1)})
            jlog(job_id, f"infer {took:.0f}s (features {feats:.0f}s, loaders+inference {took - feats:.0f}s) "
                         f"shape={preds.shape}")
            set_step(job_id, "report")
            with timed("report", "score+write"):
                _write_outputs(job_id, job, video, source_name, preds, segments)
        finally:
            stop.set()
            CURRENT["job"] = None
            _TLS.depth = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # a stopped job leaves activations behind


def _write_outputs(job_id, job, video, source_name, preds, segments):
    times = np.array([float(getattr(s, "start", i)) for i, s in enumerate(segments)])
    times = times - times.min() if len(times) else times
    np.save(job / "preds.npy", preds)
    np.save(job / "times.npy", times)
    tsv = video.with_suffix(".tsv")
    (job / "meta.json").write_text(json.dumps({
        "video": str(video), "source": source_name,
        "transcript": str(tsv) if tsv.exists() else None,
        "n_steps": int(preds.shape[0]), "space_device": DEVICE,
        "chunk_s": CHUNK_S, "whisper_model": os.environ.get("TRIBE_WHISPER_MODEL", "base"),
        "timings": JOBS[job_id].get("substeps", []),
    }))
    report, d = HL.analyse(job, video)
    report["neural_source"] = f"tribev2-full ({DEVICE})"
    (job / "report.json").write_text(json.dumps(report, indent=1))
    HL.write_md(report, job / "report.md")
    HL.write_png(report, d, job / "report.png")


async def run_job(job_id: str, video: Path, source_name: str):
    try:
        await asyncio.to_thread(_run_job_sync, job_id, video, source_name)
        j = JOBS[job_id]
        j.setdefault("took", {})["report"] = round(time.time() - j["stage_started"])
        j.update(status="done", stage="done", stage_label="Report ready", step=len(STEPS))
        jlog(job_id, f"done in {time.time() - j['started']:.0f}s, steps {j.get('took', {})}")
        jlog(job_id, f"slowest: {timing_summary(job_id)}")
    except JobCancelled as e:
        JOBS[job_id].update(status="cancelled", stage_label="Stopped by user", substep=None, loops={})
        jlog(job_id, f"cancelled: {e}; timings so far: {timing_summary(job_id)}")
    except Exception as e:  # noqa: BLE001 - surfaced to the poller
        import traceback
        traceback.print_exc()
        JOBS[job_id].update(status="failed", error=str(e), trace=traceback.format_exc()[-4000:])
        jlog(job_id, f"failed in step {JOBS[job_id].get('stage')}: {e}")
        jlog(job_id, f"timings so far: {timing_summary(job_id)}")


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model_loaded": MODEL is not None,
            "method": "heuristic, within-video normalised, uncalibrated",
            "jobs": len(JOBS), "busy": GPU_LOCK.locked(), "mem": _mem(),
            "scale": {k: round(v, 2) for k, v in SCALE.items()}, "boot": BOOT}


@app.post("/analyze")
async def analyze(background: BackgroundTasks, video: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    job = WORK / job_id
    job.mkdir(parents=True)
    ext = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = job / f"input{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return _enqueue(background, job_id, dest, video.filename or dest.name)


def _enqueue(background: BackgroundTasks, job_id: str, dest: Path, source: str, restarted_from=None):
    video_s = _video_seconds(dest)
    est = sum(_estimate(k, video_s) for k, *_ in STEPS)
    JOBS[job_id] = {"status": "queued", "stage": "queued", "stage_label": "Waiting for the GPU",
                    "step": 0, "steps": len(STEPS), "video_s": round(video_s, 1),
                    "input": str(dest), "source": source, "created": time.time(),
                    "restarted_from": restarted_from}
    jlog(job_id, f"queued {source} ({video_s:.1f}s video, est {est:.0f}s"
                 f"{', restart of ' + restarted_from if restarted_from else ''}"
                 f"{', GPU busy' if GPU_LOCK.locked() else ''})")
    background.add_task(run_job, job_id, dest, source)
    return {"job_id": job_id, "status": "queued", "video_s": round(video_s, 1), "estimate_s": round(est),
            "restarted_from": restarted_from}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    j = JOBS.get(job_id)
    if j is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    if j["status"] not in ("queued", "running"):
        return JSONResponse({"error": f"job is {j['status']}, nothing to stop"}, status_code=409)
    j["cancel"] = True
    j["stage_label"] = f"Stopping after current item ({j.get('stage_label', '')})"
    jlog(job_id, "stop requested; exits at the next checkpoint")
    return {"job_id": job_id, "status": "cancelling"}


@app.post("/jobs/{job_id}/restart")
def restart_job(job_id: str, background: BackgroundTasks):
    """Stop the job if it is still going, then queue the same upload again (no re-upload).
    The new job waits for the GPU lock, so it starts as soon as the old one exits."""
    j = JOBS.get(job_id)
    if j is None or not Path(j.get("input", "")).exists():
        return JSONResponse({"error": "unknown job or input gone"}, status_code=404)
    if j["status"] in ("queued", "running"):
        j["cancel"] = True
        jlog(job_id, "stop requested for restart")
    new_id = uuid.uuid4().hex[:8]
    (WORK / new_id).mkdir(parents=True)
    return _enqueue(background, new_id, Path(j["input"]), j.get("source", "video"), restarted_from=job_id)


@app.get("/jobs/{job_id}")
def job_detail(job_id: str):
    """Progress + timings + log for any job, including finished ones (/report returns the report)."""
    if job_id not in JOBS:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"job_id": job_id, **progress(job_id)}


@app.get("/jobs")
def list_jobs():
    now = time.time()
    rows = [{"job_id": k, "status": v.get("status"), "stage": v.get("stage"), "source": v.get("source"),
             "video_s": v.get("video_s"), "restarted_from": v.get("restarted_from"),
             "age_s": round(now - v.get("created", now))} for k, v in JOBS.items()]
    return sorted(rows, key=lambda r: r["age_s"])


@app.get("/report/{job_id}")
def get_report(job_id: str):
    st = JOBS.get(job_id)
    p = WORK / job_id / "report.json"
    if p.exists():
        return json.loads(p.read_text())
    if st is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"job_id": job_id, **progress(job_id)}


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
