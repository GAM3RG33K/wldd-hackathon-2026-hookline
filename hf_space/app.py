"""Hookline TRIBE Space: plain REST (POST video, GET report).

  POST /analyze  (multipart video) -> {job_id, status}
       ?video=1&audio=1&text=0      modalities to run (defaults: video+audio on,
                                    transcription/text off; TRIBE zero-fills the rest)
  GET  /report/{job_id}            -> progress (stage, step/steps, elapsed_s, eta_s,
                                      timeout_s, log tail, mem) or full report.json
  GET  /report/{job_id}.png / .md -> chart / markdown
  POST /jobs/{job_id}/cancel       -> stop at the next checkpoint (model stays loaded)
  POST /jobs/{job_id}/restart      -> stop if running, re-run the same upload as a new job
  GET  /jobs                       -> recent jobs, newest first
  GET  /jobs/{job_id}              -> progress, timings, chunks and log (also after done)
  GET  /jobs/{job_id}/raw          -> zip: preds (float16), times, words.tsv, meta.json (for local re-scoring)
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
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import hookline_report as HL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK = Path("/tmp/hookline_jobs")
WORK.mkdir(parents=True, exist_ok=True)
MODEL = None
GPU_LOCK = threading.Lock()  # held in a worker thread, never on the event loop
# transformers builds models on the "meta" device through process-wide state, so two threads loading
# models at once (boot warm-up + an early job) can leave TRIBE's weights on meta ("Cannot copy out of
# meta tensor"). Every model load in this process goes through this lock.
LOAD_LOCK = threading.RLock()
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
# Per-modality share of the per-video-second cost (fp16 run 794fe136: video ~8s, text ~2.5s,
# audio ~0.3s per video second; transcription ~3.5s per video second in events).
MODALITY_COST = {"video": 8.0, "text": 2.5, "audio": 0.3}
TRANSCRIBE_COST = 3.5
DEFAULT_MODALITIES = {"video": True, "audio": True, "text": True}
# Transcript mode: auto = the speech gate decides per reel (humming/music/song reels are
# scored on what is seen and heard, not on a transcript); on/off force it.
TEXT_MODES = {"auto", "on", "off"}
DEFAULT_TEXT_MODE = os.environ.get("HOOKLINE_TEXT", "auto")
AST_REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"  # AudioSet classifier for the speech gate
# Below SPEECH_SKIP_RATIO speech the transcript is skipped outright; above it, Whisper runs and
# transcript confidence decides (FC023: 14% speech over music, but ~17 clear words).
SPEECH_MIN_RATIO = float(os.environ.get("HOOKLINE_SPEECH_MIN", "0.05"))  # share of seconds with speech
GATE_P = 0.4  # per-second class probability threshold
# AudioSet labels grouped into what matters for a reel's audio.
# AudioSet sounds that signal an emotion (per second, from the same AST pass as the speech gate).
EMO_SOUNDS = {
    "amusement": ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Baby laughter", "Snicker"],
    "awe": ["Cheering", "Applause", "Whoop"],
    "surprise": ["Gasp"],
    "tension": ["Screaming"],
    "sadness": ["Crying, sobbing", "Whimper", "Sigh"],
}
EMOTION_ON = os.environ.get("HOOKLINE_EMOTION", "1") != "0"
AUDIO_GROUPS = {
    "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
               "Conversation", "Narration, monologue"],
    "vocals": ["Song", "Singing", "Male singing", "Female singing", "Child singing", "Vocal music", "Rapping",
               "Choir", "A capella"],
    "music": ["Music"],
    "bed": ["Background music", "Theme music", "Soundtrack music"],
    "indian": ["Music of Bollywood", "Carnatic music", "Music of Asia"],
    "humming": ["Humming", "Chant", "Mantra"],
    "laughter": ["Laughter"],
    "silence": ["Silence"],
}
_AST = {}
SCALE = {k: 1.0 for k, *_ in STEPS}
LOG_TAIL = 60
HEARTBEAT_S = 20
# Audio/video are split into chunks of at most this many seconds before feature
# extraction. tribev2 defaults to 30-60s chunks, which put a whole reel through
# Wav2VecBert attention at once and ran a T4 out of memory on a 28s video.
CHUNK_S = float(os.environ.get("HOOKLINE_CHUNK_S", "10"))
# Speed: fp16 weights + autocast (T4 fp16 tensor cores are ~8x fp32), and keep feature
# models on the GPU between jobs while allocated memory stays under KEEP_GB.
PRECISION = os.environ.get("HOOKLINE_PRECISION", "fp16")  # fp16 | fp32
# 4 GB keeps V-JEPA2 (~2 GB fp16) + Wav2VecBert (~1.2 GB); Llama (~6.4 GB) is freed after use.
# 9 GB (Llama + w2v resident) ran V-JEPA out of memory on a T4 (job 794fe136).
KEEP_GB = float(os.environ.get("HOOKLINE_KEEP_GB", "4"))   # 0 = always free (tribev2 default)
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


def _estimate(key: str, video_s: float, mods=None) -> float:
    if key == "load_model" and MODEL is not None:
        return 0.0
    mods = mods or DEFAULT_MODALITIES
    _, _, fixed, per_s = next(s for s in STEPS if s[0] == key)
    if key == "events":
        per_s = TRANSCRIBE_COST if mods.get("text") else 0.1
    elif key == "predict":
        per_s = sum(c for m, c in MODALITY_COST.items() if mods.get(m))
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
    out = {k: job[k] for k in ("status", "stage", "stage_label", "step", "steps", "video_s", "modalities", "lang", "asr", "text_mode",
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
    out["plan"] = [{"key": k, "label": label,
                    "est_s": round(job.get("plan", {}).get(k, _estimate(k, job["video_s"], job.get("modalities")))),
                    "took_s": job.get("took", {}).get(k)} for k, label, *_ in STEPS]
    if job.get("speech_gate"):
        out["speech_gate"] = {k: v for k, v in job["speech_gate"].items() if not k.endswith("_per_second")}
    sub = job.get("substep")
    if sub:
        out["substep"] = {"name": sub["name"], "elapsed_s": round(now - sub["started"])}
    out["substeps"] = job.get("substeps", [])
    out["loops"] = list(job.get("loops", {}).values())
    out["chunks"] = job.get("chunks", [])
    out["video_chunk"] = job.get("video_chunk")
    out["chunks_done"] = job.get("chunks_done", [])
    out["heartbeat"] = job.get("heartbeat")
    out["transcript"] = job.get("transcript")
    out["model_loads"] = job.get("model_loads", [])
    am = job.get("audio_match")
    if am:
        out["audio_match"] = {"song": am.get("song"), "reuse": am.get("reuse", [])}
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


def _load_audio(path, sr=16000):
    import subprocess
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vn", "-f", "f32le",
                          "-ac", "1", "-ar", str(sr), "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def speech_gate(job_id: str, video) -> dict:
    """Per-second AudioSet labels (2s windows, 1s hop): how much of the reel is speech vs
    singing / humming / music / laughter. In auto mode, too little speech turns the
    transcript off so song/humming reels are scored on visuals + sound."""
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    j = JOBS[job_id]
    if "model" not in _AST:
        with LOAD_LOCK:
            t0 = time.time()
            _AST["fe"] = ASTFeatureExtractor.from_pretrained(AST_REPO)
            m = ASTForAudioClassification.from_pretrained(AST_REPO, torch_dtype=torch.float32).eval()
            _AST["model"] = m.to(DEVICE)
            jlog(job_id, f"  loaded AST AudioSet classifier in {time.time() - t0:.1f}s")
    fe, model = _AST["fe"], _AST["model"]
    labels = model.config.label2id
    sr = 16000
    audio = _load_audio(video, sr)
    n = int(np.ceil(len(audio) / sr)) if len(audio) else 0
    names = sorted({c for grp in (*AUDIO_GROUPS.values(), *EMO_SOUNDS.values()) for c in grp if c in labels})
    probs = np.zeros((n, len(names)), dtype=np.float32)
    for b0 in range(0, n, 16):
        chunks = [audio[i * sr:(i + 2) * sr] for i in range(b0, min(n, b0 + 16))]
        chunks = [c if len(c) >= sr // 2 else np.pad(c, (0, sr // 2 - len(c))) for c in chunks]
        x = fe(chunks, sampling_rate=sr, return_tensors="pt")["input_values"].to(DEVICE)
        with torch.inference_mode(), _autocast():
            p = torch.sigmoid(model(x).logits.float()).cpu().numpy()
        probs[b0:b0 + len(chunks)] = p[:, [labels[c] for c in names]]
    col = {c: probs[:, i] for i, c in enumerate(names)}
    zero = np.zeros(n)
    grp = {g: np.max([col[c] for c in members if c in col], axis=0) if any(c in col for c in members) else zero
           for g, members in AUDIO_GROUPS.items()}
    speech_sec = (grp["speech"] > GATE_P) & (grp["speech"] >= grp["vocals"])
    vocal_sec = (grp["vocals"] > GATE_P) & (grp["vocals"] > grp["speech"])
    frac = lambda m: round(float(m.mean()), 2) if n else 0.0
    ratio = {"speech": frac(speech_sec), "vocals": frac(vocal_sec),
             **{g: frac(grp[g] > GATE_P) for g in ("music", "bed", "indian", "humming", "laughter", "silence")}}
    atype = audio_type(ratio)
    mode = j.get("text_mode", DEFAULT_TEXT_MODE)
    keep = atype in ("speech", "speech_over_music")
    why = {"song": f"audio is a song (vocals {ratio['vocals']:.0%}, speech {ratio['speech']:.0%}); lyrics are not "
                   "the creator's words",
           "instrumental": f"instrumental music, no speech ({ratio['music']:.0%} music)",
           "humming": f"humming/chant, no speech ({ratio['humming']:.0%})",
           "other": f"no clear speech ({ratio['speech']:.0%})",
           "speech": f"speech in {ratio['speech']:.0%} of the reel",
           "speech_over_music": f"speech in {ratio['speech']:.0%} of the reel over a music bed"}[atype]
    if mode == "auto":
        decision = "transcript on" if keep else "transcript off"
    else:
        decision = f"transcript {mode} (forced)"
    sounds = {e: [round(float(v), 3) for v in np.max([col[c] for c in members if c in col], axis=0)]
              for e, members in EMO_SOUNDS.items() if any(c in col for c in members)}
    gate = {"mode": mode, "decision": decision, "reason": why, "audio_type": atype, "ratios": ratio, "seconds": n,
            "sounds_per_second": sounds,
            "speech_per_second": [round(float(v), 2) for v in grp["speech"]],
            "vocals_per_second": [round(float(v), 2) for v in grp["vocals"]]}
    j["speech_gate"] = gate
    if mode == "auto" and not keep:
        j["modalities"]["text"] = False
    jlog(job_id, f"  audio: {atype}; {decision} ({why}); ratios {ratio}")
    return gate


def audio_type(r: dict) -> str:
    """Classify the reel's audio from per-second AudioSet shares."""
    if r["vocals"] >= 0.4 and r["speech"] < 0.2:
        return "song"
    if r["speech"] >= SPEECH_MIN_RATIO:
        return "speech_over_music" if r["music"] >= 0.3 or r["bed"] >= 0.3 else "speech"
    if r["humming"] >= 0.2:
        return "humming"
    if r["music"] >= 0.4:
        return "instrumental"
    return "other"


def song_lookup(job_id: str, video, source: str):
    """Which song is it, is it on the trending list, and have other analysed reels used it?"""
    import audio_match as AM
    j = JOBS[job_id]
    atype = (j.get("speech_gate") or {}).get("audio_type", "unknown")
    res = AM.identify(video, job_id, source, atype, log=lambda m: jlog(job_id, f"  {m}"))
    j["audio_match"] = res
    song = res.get("song")
    if song:
        jlog(job_id, f"  song match: {song['title']} - {song['artist']} (trending list: {song['list']}, "
                     f"similarity {song['similarity']}, at {song['offset_s']}s)")
    else:
        jlog(job_id, f"  song match: none in trending catalog ({AM.STATE['status']})")
    if res.get("reuse"):
        jlog(job_id, f"  same audio as {len(res['reuse'])} earlier reel(s): "
                     f"{', '.join(r['source'] for r in res['reuse'][:5])}")
    return res


def _post_transcript_check(job_id: str, events):
    """Auto mode: drop a transcript that is too thin or low-confidence to trust."""
    j = JOBS.get(job_id, {})
    if j.get("text_mode", DEFAULT_TEXT_MODE) != "auto":
        return events, False
    tr = j.get("transcript") or {}
    words = int((events.type == "Word").sum())
    lp = tr.get("avg_logprob")
    bad = words < 3 or (lp is not None and lp < -1.0)
    if bad:
        events = events[events.type != "Word"]
        j["modalities"]["text"] = False
        j["transcript_words"] = []
        g = j.setdefault("speech_gate", {})
        g["decision"] = "transcript off"
        g["reason"] = (f"transcript not trustworthy ({words} words, avg logprob "
                       f"{lp if lp is None else round(lp, 2)})")
        jlog(job_id, f"  speech gate: {g['decision']} after transcription ({g['reason']})")
    return events, bad


def _timed_audio_text_events(events, audio_only: bool = False):
    """Copy of tribev2.demo_utils.get_audio_and_text_events with per-transform timing
    and CHUNK_S chunking (OOM fix)."""
    import tribev2.demo_utils as du
    job = JOBS.get(CURRENT["job"] or "", {})
    if not job.get("modalities", DEFAULT_MODALITIES).get("text", True):
        audio_only = True  # transcription off: no words, no Llama text features
        jlog(CURRENT["job"], "  transcription off: skipping transcribe/add_text/sentences/context")
    # Transcribe BEFORE chunking. tribev2 chunks first, and ExtractWordsFromAudio then copies
    # the whole-file transcript into every Audio chunk, shifted by the chunk start: with n
    # chunks every word appears n times, n-1 of them at wrong times (upstream bug for clips
    # over 60s; with our 10s chunks it hit every reel: 2086 "words" for a 72s reel).
    transforms = [("extract_audio", du.ExtractAudioFromVideo())]
    if not audio_only:
        transforms += [
            ("transcribe", du.ExtractWordsFromAudio()),
            ("add_text", du.AddText()),
            ("sentences", du.AddSentenceToWords(max_unmatched_ratio=0.05)),
            ("word_context", du.AddContextToWords(sentence_only=False, max_context_len=1024, split_field="")),
            ("remove_missing", du.RemoveMissing()),
        ]
    transforms += [
        ("chunk_audio", du.ChunkEvents(event_type_to_chunk="Audio", max_duration=CHUNK_S, min_duration=CHUNK_S / 2)),
        ("chunk_video", du.ChunkEvents(event_type_to_chunk="Video", max_duration=CHUNK_S, min_duration=CHUNK_S / 2)),
    ]
    events = du.standardize_events(events)
    text_steps = {"add_text", "sentences", "word_context", "remove_missing"}
    skip_text = False
    for name, t in transforms:
        if skip_text and name in text_steps:
            continue
        with timed("events", name):
            events = t(events)
        if name == "transcribe" and CURRENT["job"]:
            events, skip_text = _post_transcript_check(CURRENT["job"], events)
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


def transcript_done(model: str, device: str, audio_s: float, took: float, n_words: int = -1,
                    lang: str = "", detected: str = "", sample: str = ""):
    """Called by the in-process transcription (proves the fast path ran)."""
    job_id = CURRENT["job"]
    msg = (f"whisper {model} on {device} (lang {lang}, detected {detected}): {audio_s:.1f}s audio -> "
           f"{n_words} words in {took:.1f}s; sample: {sample[:120]!r}")
    print(f"[space] {job_id} {msg}", flush=True)
    if job_id:
        jlog(job_id, f"  {msg}")
        JOBS[job_id]["transcript"] = {"path": "faster-whisper", "model": model, "device": device,
                                      "audio_s": round(audio_s, 1), "words": n_words, "took_s": round(took, 1),
                                      "lang": lang, "detected": detected, "sample": sample}


def _install_speed_hooks():
    """fp16 model loading (timed) + keep feature models resident on the GPU."""
    import transformers
    import tribev2.main as tm

    orig_fp = transformers.PreTrainedModel.from_pretrained.__func__

    def from_pretrained(cls, *a, **k):
        if PRECISION == "fp16" and torch.cuda.is_available() and "torch_dtype" not in k and "dtype" not in k:
            k["torch_dtype"] = torch.float16
        t0 = time.time()
        m = orig_fp(cls, *a, **k)
        name = a[0] if a else k.get("pretrained_model_name_or_path", cls.__name__)
        msg = f"loaded {name} ({k.get('torch_dtype', 'default dtype')}) in {time.time() - t0:.1f}s"
        print(f"[space] {msg}", flush=True)
        if CURRENT["job"]:
            jlog(CURRENT["job"], f"  {msg}")
            JOBS[CURRENT["job"]].setdefault("model_loads", []).append(
                {"model": str(name), "took_s": round(time.time() - t0, 1)})
        return m

    transformers.PreTrainedModel.from_pretrained = classmethod(from_pretrained)

    orig_free = tm._free_extractor_model

    def maybe_free(extractor):
        gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        name = type(extractor).__name__
        if KEEP_GB > 0 and gb <= KEEP_GB:
            if CURRENT["job"]:
                jlog(CURRENT["job"], f"  keeping {name} on GPU ({gb:.1f} GB allocated, budget {KEEP_GB:.0f})")
            return
        orig_free(extractor)

    tm._free_extractor_model = maybe_free
    tm._orig_free_extractor_model = orig_free
    print(f"[space] speed hooks: precision {PRECISION}, keep models <= {KEEP_GB} GB", flush=True)


# Which tribev2 extractors (and models) each modality needs.
MODALITY_EXTRACTORS = {"video": ("video_feature", "image_feature"), "audio": ("audio_feature",),
                       "text": ("text_feature",)}
MODALITY_MODELS = {"video": "V-JEPA2 ViT-g", "audio": "Wav2Vec-BERT 2.0", "text": "Whisper + Llama-3.2-3B"}


def release_unneeded(job_id: str, model, mods: dict):
    """Free models kept resident by earlier jobs that this job's modalities don't use.
    Models for enabled modalities load lazily on first use; disabled ones never load."""
    import tribev2.main as tm
    free = getattr(tm, "_orig_free_extractor_model", tm._free_extractor_model)
    freed = []
    for mod, on in mods.items():
        if on:
            continue
        for attr in MODALITY_EXTRACTORS[mod]:
            ext = getattr(model.data, attr, None)
            targets = [ext, getattr(ext, "image", None)] if ext is not None else []
            if any(isinstance(getattr(t, "_model", None), torch.nn.Module) for t in targets if t is not None):
                free(ext)
                freed.append(attr)
    if not mods.get("text"):
        if _FW:
            _FW.clear()  # faster-whisper from an earlier transcription job
            freed.append("whisper")
    if freed and torch.cuda.is_available():
        torch.cuda.empty_cache()
    load = [MODALITY_MODELS[m] for m, on in mods.items() if on]
    skip = [MODALITY_MODELS[m] for m, on in mods.items() if not on]
    gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    jlog(job_id, f"models: using {', '.join(load) or 'none'}; not loading {', '.join(skip) or 'none'}"
                 f"{'; freed resident ' + ', '.join(freed) if freed else ''} (GPU {gb:.1f} GB allocated)")


def free_kept_models(model):
    """Drop every resident feature model (OOM recovery)."""
    import tribev2.main as tm
    free = getattr(tm, "_orig_free_extractor_model", tm._free_extractor_model)
    for m in getattr(model.data, "features_to_use", []):
        ext = getattr(model.data, f"{m}_feature", None)
        if ext is not None:
            free(ext)


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


_FW = {}  # faster-whisper models by name, loaded once per process
REPEAT_TOKEN = __import__("re").compile(r"^(.{1,4})\1{2,}$")  # short unit repeated 3+ times
_ORIG_TRANSCRIPT = None
WHISPER_CODES = dict(english="en", french="fr", spanish="es", dutch="nl", chinese="zh")
# Reels are Hindi, English or Hinglish. Whisper runs multilingual with per-window language
# detection unless a job pins one. tribev2 still labels Word events "english": that label only
# picks the spaCy sentence model (no Hindi pipeline exists), not the transcription language.
DEFAULT_LANG = os.environ.get("HOOKLINE_LANG", "auto")  # auto | hi | en
LANGS = {"auto", "hi", "en"}
# ASR backends. hinglish = Oriserve/Whisper-Hindi2Hinglish-Apex (Apache-2.0, large-v3-turbo
# fine-tune, Roman-script Hinglish), converted once per container to CTranslate2 so
# faster-whisper gives word timestamps. whisper = stock multilingual model (Devanagari for Hindi).
DEFAULT_ASR = os.environ.get("HOOKLINE_ASR", "hinglish")
ASRS = {"hinglish", "whisper"}
HINGLISH_REPO = "Oriserve/Whisper-Hindi2Hinglish-Apex"
HINGLISH_DIR = Path(os.environ.get("HOOKLINE_ASR_DIR", "/tmp/asr")) / "hinglish-apex-ct2"
_CONVERT_LOCK = threading.Lock()


def ensure_hinglish_model() -> str:
    """Convert the Hinglish model to CTranslate2 once (about 1-2 min), return its path."""
    import subprocess
    with _CONVERT_LOCK:
        if (HINGLISH_DIR / "model.bin").exists():
            return str(HINGLISH_DIR)
        HINGLISH_DIR.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cmd = ["ct2-transformers-converter", "--model", HINGLISH_REPO, "--output_dir", str(HINGLISH_DIR),
               "--quantization", "float16", "--copy_files", "tokenizer.json", "preprocessor_config.json",
               "--force"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not (HINGLISH_DIR / "model.bin").exists():
            raise RuntimeError(f"hinglish conversion failed: {(r.stderr or r.stdout)[-400:]}")
        BOOT["hinglish_asr"] = f"ready ({time.time() - t0:.0f}s convert)"
        print(f"[space] {HINGLISH_REPO} converted in {time.time() - t0:.0f}s", flush=True)
        return str(HINGLISH_DIR)


def _warm_hinglish():
    try:
        BOOT["hinglish_asr"] = "converting"
        ensure_hinglish_model()
    except Exception as e:  # noqa: BLE001 - jobs fall back to whisper and log why
        BOOT["hinglish_asr"] = f"FAILED {e}"[:300]


def _fast_transcript(wav_filename, language):
    """In-process faster-whisper, replacing tribev2's `uvx whisperx` subprocess (which
    installs whisperx and downloads an alignment model on every job). Any failure is
    logged with the real error, then falls back to the original so the job finishes."""
    import pandas as pd
    import subprocess
    job_id = CURRENT["job"]
    lang = JOBS.get(job_id or "", {}).get("lang", DEFAULT_LANG)
    asr = JOBS.get(job_id or "", {}).get("asr", DEFAULT_ASR)
    name = os.environ.get("TRIBE_WHISPER_MODEL", "large-v3-turbo")
    if asr == "hinglish":
        try:
            name = ensure_hinglish_model()
        except Exception as e:  # noqa: BLE001 - say why, use stock whisper
            asr = "whisper"
            if job_id:
                jlog(job_id, f"  hinglish ASR unavailable ({str(e)[:200]}); using {name}")
    try:
        from faster_whisper import WhisperModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if name not in _FW:
            t0 = time.time()
            _FW[name] = WhisperModel(name, device=dev, compute_type="float16" if dev == "cuda" else "int8")
            if job_id:
                jlog(job_id, f"  loaded faster-whisper {name} on {dev} in {time.time() - t0:.1f}s")
        t0 = time.time()
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav_filename), "-f", "f32le",
                              "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
        audio = np.frombuffer(raw, dtype=np.float32)
        def run(vad):
            segs, info = _FW[name].transcribe(
                audio, language=None if lang == "auto" else lang, multilingual=lang == "auto",
                word_timestamps=True, beam_size=5, vad_filter=vad, condition_on_previous_text=False)
            out, dropped, lps = [], {"loop": 0, "no_speech": 0, "repeat": 0}, []
            for i, seg in enumerate(segs):  # generator: decoding happens while iterating
                # Whisper hallucination guards: repetition loops (seen on songs: "nan nan nan ...")
                # and low-confidence segments over music/silence.
                if seg.compression_ratio > 2.4:
                    dropped["loop"] += 1
                    continue
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                    dropped["no_speech"] += 1
                    continue
                seg_words = [w.word.strip().lower() for w in (seg.words or []) if w.word.strip()]
                lps.append(seg.avg_logprob)
                if len(seg_words) >= 6 and len(set(seg_words)) / len(seg_words) < 0.3:
                    dropped["loop"] += 1  # e.g. 43 words, 2 distinct
                    continue
                for w in (seg.words or []):
                    text = w.word.replace('"', "").strip()
                    if not text:
                        continue
                    if REPEAT_TOKEN.match(text.lower()):
                        dropped["repeat"] += 1  # glued loop token like "nannannannan"
                        continue
                    if len(out) >= 2 and out[-1]["text"].lower() == out[-2]["text"].lower() == text.lower():
                        dropped["repeat"] += 1  # third+ identical word in a row
                        continue
                    out.append({"text": text, "start": w.start, "duration": w.end - w.start,
                                "sequence_id": i, "sentence": seg.text.replace('"', "")})
            if job_id and any(dropped.values()):
                jlog(job_id, f"  transcript guard dropped {dropped}")
                JOBS[job_id]["transcript_dropped"] = dropped
            info.hl_avg_logprob = float(np.mean(lps)) if lps else None
            return out, info

        words, info = run(vad=True)
        if not words:  # VAD can drop speech under loud music; try once without it
            if job_id:
                jlog(job_id, "  no speech after VAD; retrying transcription without VAD")
            words, info = run(vad=False)
        detected = f"{info.language} ({info.language_probability:.2f})"
        sample = " ".join(w["text"] for w in words[:15])
        transcript_done(asr if asr == "hinglish" else name, dev, len(audio) / 16000, time.time() - t0,
                        len(words), lang=lang, detected=detected, sample=sample)
        if job_id and JOBS[job_id].get("transcript") is not None:
            JOBS[job_id]["transcript"]["avg_logprob"] = getattr(info, "hl_avg_logprob", None)
        if job_id:
            JOBS[job_id]["transcript_segments"] = _segments(words)
            JOBS[job_id]["transcript_words"] = words
        return pd.DataFrame(words)
    except Exception as e:  # noqa: BLE001 - fall back, but say exactly why
        msg = f"{type(e).__name__}: {e}"
        BOOT["transcript"] = f"fallback to uvx whisperx ({msg[:200]})"
        if job_id:
            jlog(job_id, f"  fast transcription FAILED ({msg[:300]}); falling back to slow uvx whisperx")
            JOBS[job_id]["transcript"] = {"path": "uvx whisperx fallback", "error": msg[:500]}
        return _ORIG_TRANSCRIPT(wav_filename, language)


def _segments(words):
    """Group transcript words into timestamped lines for the report."""
    out = []
    for w in words:
        if out and out[-1]["seq"] == w["sequence_id"]:
            out[-1]["text"] += " " + w["text"]
            out[-1]["end"] = round(w["start"] + w["duration"], 2)
        else:
            out.append({"seq": w["sequence_id"], "start": round(w["start"], 2),
                        "end": round(w["start"] + w["duration"], 2), "text": w["text"]})
    return [{k: v for k, v in seg.items() if k != "seq"} for seg in out]


def _install_transcript_hook():
    global _ORIG_TRANSCRIPT
    import tribev2.eventstransforms as et
    cls = et.ExtractWordsFromAudio
    if _ORIG_TRANSCRIPT is None:
        _ORIG_TRANSCRIPT = cls._get_transcript_from_audio
    type.__setattr__(cls, "_get_transcript_from_audio", staticmethod(_fast_transcript))
    BOOT["transcript"] = f"faster-whisper {os.environ.get('TRIBE_WHISPER_MODEL', 'base')} (in-process)"
    print(f"[space] transcript hook: {BOOT['transcript']}", flush=True)


def _warm_catalog():
    try:
        import audio_match as AM
        AM.build_catalog(log=lambda m: print(m, flush=True))
        BOOT["trending_catalog"] = AM.STATE["status"]
    except Exception as e:  # noqa: BLE001 - surfaced on /health
        BOOT["trending_catalog"] = f"FAILED {type(e).__name__}: {e}"[:300]


def _warm_emotion():
    if not EMOTION_ON:
        return
    try:
        import emotion as EMO
        t0 = time.time()
        with LOAD_LOCK:
            st = EMO.warm(DEVICE)
        ok = all(v == "ok" for v in st.values())
        BOOT["emotion_models"] = (f"ready ({time.time() - t0:.0f}s)" if ok else
                                  f"PARTIAL ({time.time() - t0:.0f}s): {st}")
    except Exception as e:  # noqa: BLE001 - jobs retry lazily and log per channel
        BOOT["emotion_models"] = f"FAILED {type(e).__name__}: {e}"[:300]


@app.on_event("startup")
def _startup():
    threading.Thread(target=_warm_catalog, daemon=True).start()
    threading.Thread(target=_warm_emotion, daemon=True).start()
    if DEFAULT_ASR == "hinglish":
        threading.Thread(target=_warm_hinglish, daemon=True).start()
    try:  # import check only; the model loads on the first transcription
        import faster_whisper, ctranslate2
        BOOT["faster_whisper"] = (f"{faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}, "
                                  f"cuda devices {ctranslate2.get_cuda_device_count()}")
    except Exception as e:  # noqa: BLE001 - surfaced on /health
        BOOT["faster_whisper"] = f"IMPORT FAILED {type(e).__name__}: {e}"
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
    with LOAD_LOCK:  # waits for boot warm-up to finish loading its models
        if MODEL is not None:
            return MODEL
        MODEL = TribeModel.from_pretrained("facebook/tribev2", device=DEVICE, config_update=overrides)
    for hook in (_install_transcript_hook, _install_speed_hooks, _install_timing_hooks, _install_progress_loops):
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
        j["plan"] = {k: _estimate(k, j["video_s"], j.get("modalities")) for k, *_ in STEPS}
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
            release_unneeded(job_id, model, j.get("modalities", DEFAULT_MODALITIES))
            set_step(job_id, "events")
            if j["modalities"].get("audio") or j.get("text_mode") != "off":
                try:
                    with timed("events", "speech_gate"):
                        speech_gate(job_id, video)
                except JobCancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - gate is advisory; keep the requested mode
                    jlog(job_id, f"  speech gate failed ({type(e).__name__}: {e}); keeping transcript mode")
            if j["modalities"].get("audio") or j.get("text_mode") != "off":
                try:
                    with timed("events", "song_match"):
                        song_lookup(job_id, video, source_name)
                except JobCancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - context only
                    jlog(job_id, f"  song match failed ({type(e).__name__}: {e})")
            events = model.get_events_dataframe(video_path=str(video))
            mods = j.get("modalities", DEFAULT_MODALITIES)
            drop = [t for m, t in (("video", "Video"), ("audio", "Audio")) if not mods.get(m)]
            if drop:
                events = events[~events.type.isin(drop)]
                jlog(job_id, f"modalities off: dropped {drop} events (TRIBE zero-fills them)")
            try:
                counts = events.type.value_counts().to_dict()
            except Exception:  # noqa: BLE001 - logging only
                counts = {}
            jlog(job_id, f"events ready: {len(events)} rows {counts}, mem {_mem()}")
            if EMOTION_ON:
                try:
                    import emotion as EMO
                    with timed("events", "emotion"):
                        gate = j.get("speech_gate") or {}
                        audio = _load_audio(video)
                        n = int(np.ceil(len(audio) / 16000)) or int(np.ceil(j["video_s"]))
                        sp = gate.get("speech_per_second") or [0.0] * n
                        segs = j.get("transcript_segments") if j["modalities"].get("text") else []
                        with LOAD_LOCK:  # its channels may still load models lazily
                            j["emotion"] = EMO.emotion_cues(video, audio, 16000, sp, segs or [], DEVICE,
                                                            log=lambda m: jlog(job_id, f"  {m}"),
                                                            caption=j.get("caption", ""),
                                                            transcript=" ".join(s["text"] for s in
                                                                                (j.get("transcript_segments") or [])))
                except JobCancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - emotion is additive, never fatal
                    jlog(job_id, f"  emotion cues failed ({type(e).__name__}: {e})")
            set_step(job_id, "predict")
            t1 = time.time()
            n_before = len(j.get("substeps", []))
            preds, segments = _predict(job_id, model, events)
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


def _autocast():
    if PRECISION == "fp16" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _predict(job_id: str, model, events):
    """model.predict under fp16 autocast. On CUDA OOM with resident models, free them,
    stop keeping models for this process, and retry once."""
    global KEEP_GB
    ctx = _autocast()
    try:
        with ctx:
            return model.predict(events=events)
    except Exception as e:  # noqa: BLE001 - only OOM is retried
        if "out of memory" not in str(e).lower() or KEEP_GB <= 0:
            raise
        jlog(job_id, f"CUDA OOM with resident models ({e.__class__.__name__}); freeing them, "
                     "disabling keep, retrying predict once")
        KEEP_GB = 0
        free_kept_models(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        JOBS[job_id].pop("video_chunk_i", None)
        JOBS[job_id].pop("chunks_done", None)
        ctx = _autocast()
        with ctx:
            return model.predict(events=events)


def _space_extra(job_id):
    """Everything the local re-scorer needs besides preds/times/words (travels in meta.json)."""
    jj = JOBS[job_id]
    am = jj.get("audio_match") or {}
    return {"neural_source": f"tribev2-full ({DEVICE})",
            "modalities": jj.get("modalities", DEFAULT_MODALITIES),
            "speech_gate": jj.get("speech_gate"),
            "transcript_info": jj.get("transcript"), "transcript_segments": jj.get("transcript_segments"),
            "transcript_dropped": jj.get("transcript_dropped"),
            "audio_match": {"song": am.get("song"), "reuse": am.get("reuse", [])},
            "text_mode": jj.get("text_mode"), "lang": jj.get("lang"), "asr": jj.get("asr"),
            "emotion": jj.get("emotion"), "caption": jj.get("caption", "")}


def _write_outputs(job_id, job, video, source_name, preds, segments):
    import pandas as pd
    times = np.array([float(getattr(s, "start", i)) for i, s in enumerate(segments)])
    times = times - times.min() if len(times) else times
    np.save(job / "preds.npy", preds)
    np.save(job / "times.npy", times)
    extra = _space_extra(job_id)
    words = JOBS[job_id].get("transcript_words") or []
    words_tsv = job / "words.tsv"  # the words the model actually used (post guard / gate)
    if extra["modalities"].get("text") and words:
        pd.DataFrame(words).to_csv(words_tsv, sep="\t", index=False)
    (job / "meta.json").write_text(json.dumps({
        "video": str(video), "source": source_name,
        "transcript": str(words_tsv) if words_tsv.exists() else None,
        "n_steps": int(preds.shape[0]), "space_device": DEVICE, "space_job": job_id,
        "chunk_s": CHUNK_S, "whisper_model": os.environ.get("TRIBE_WHISPER_MODEL", "base"),
        "precision": PRECISION, "keep_gb": KEEP_GB,
        "modalities": extra["modalities"],
        "timings": JOBS[job_id].get("substeps", []), "model_loads": JOBS[job_id].get("model_loads", []),
        "space": extra,
    }, default=str))
    report, d = HL.analyse(job, video)
    HL.finalize(report, extra)
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
            "scale": {k: round(v, 2) for k, v in SCALE.items()},
            "speed": {"precision": PRECISION, "keep_gb": KEEP_GB, "chunk_s": CHUNK_S,
                      "whisper": os.environ.get("TRIBE_WHISPER_MODEL", "base")}, "boot": BOOT}


@app.post("/analyze")
async def analyze(background: BackgroundTasks, video: UploadFile = File(...),
                  video_on: bool = Query(True, alias="video"), audio_on: bool = Query(True, alias="audio"),
                  text: str = Query(DEFAULT_TEXT_MODE), lang: str = Query(DEFAULT_LANG),
                  asr: str = Query(DEFAULT_ASR), caption: str = Form("")):
    if asr not in ASRS:
        return JSONResponse({"error": f"asr must be one of {sorted(ASRS)}"}, status_code=422)
    if lang not in LANGS:
        return JSONResponse({"error": f"lang must be one of {sorted(LANGS)}"}, status_code=422)
    text_mode = {"1": "on", "true": "on", "0": "off", "false": "off"}.get(text.lower(), text.lower())
    if text_mode not in TEXT_MODES:
        return JSONResponse({"error": f"text must be one of {sorted(TEXT_MODES)}"}, status_code=422)
    mods = {"video": video_on, "audio": audio_on, "text": text_mode != "off"}
    if not any(mods.values()):
        return JSONResponse({"error": "enable at least one of video, audio, text"}, status_code=422)
    job_id = uuid.uuid4().hex[:8]
    job = WORK / job_id
    job.mkdir(parents=True)
    ext = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = job / f"input{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return _enqueue(background, job_id, dest, video.filename or dest.name, modalities=mods, lang=lang, asr=asr,
                    text_mode=text_mode, caption=caption.strip()[:2200])  # Instagram's caption limit


def _enqueue(background: BackgroundTasks, job_id: str, dest: Path, source: str, restarted_from=None,
             modalities=None, lang=None, asr=None, text_mode=None, caption=""):
    lang = lang or DEFAULT_LANG
    text_mode = text_mode or DEFAULT_TEXT_MODE
    asr = asr or DEFAULT_ASR
    modalities = modalities or dict(DEFAULT_MODALITIES)
    video_s = _video_seconds(dest)
    est = sum(_estimate(k, video_s, modalities) for k, *_ in STEPS)
    JOBS[job_id] = {"status": "queued", "stage": "queued", "stage_label": "Waiting for the GPU",
                    "step": 0, "steps": len(STEPS), "video_s": round(video_s, 1),
                    "input": str(dest), "source": source, "created": time.time(),
                    "restarted_from": restarted_from, "modalities": modalities, "lang": lang, "asr": asr,
                    "text_mode": text_mode, "caption": caption or ""}
    on = "+".join(m for m, v in modalities.items() if v)
    jlog(job_id, f"queued {source} ({video_s:.1f}s video, modalities {on}, est {est:.0f}s"
                 f"{', restart of ' + restarted_from if restarted_from else ''}"
                 f"{', caption ' + str(len(caption)) + ' chars' if caption else ''}"
                 f"{', GPU busy' if GPU_LOCK.locked() else ''})")
    background.add_task(run_job, job_id, dest, source)
    return {"job_id": job_id, "status": "queued", "video_s": round(video_s, 1), "estimate_s": round(est),
            "restarted_from": restarted_from, "modalities": modalities, "lang": lang, "asr": asr,
            "text_mode": text_mode, "caption": bool(caption)}


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
    return _enqueue(background, new_id, Path(j["input"]), j.get("source", "video"), restarted_from=job_id,
                    modalities=dict(j.get("modalities") or DEFAULT_MODALITIES,
                                    text=j.get("text_mode", DEFAULT_TEXT_MODE) != "off"),
                    lang=j.get("lang"), asr=j.get("asr"), text_mode=j.get("text_mode"), caption=j.get("caption", ""))


@app.get("/jobs/{job_id}/raw")
def job_raw(job_id: str):
    """Raw model output for local report generation: preds (float16), times, words, meta.
    Lives in /tmp: gone after a Space restart, so clients should fetch it right after 'done'."""
    import io
    import zipfile
    from fastapi.responses import Response
    jd = WORK / job_id
    if not (jd / "preds.npy").exists():
        return JSONResponse({"error": "no raw output for this job (not done, or Space restarted)"}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        b = io.BytesIO()
        np.save(b, np.load(jd / "preds.npy").astype(np.float16))
        z.writestr("preds.npy", b.getvalue())
        z.write(jd / "times.npy", "times.npy")
        z.write(jd / "meta.json", "meta.json")
        if (jd / "words.tsv").exists():
            z.write(jd / "words.tsv", "words.tsv")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename={job_id}-raw.zip"})


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


# Asset routes first: FastAPI matches in order, and /report/{job_id} would otherwise
# capture '<id>.png' as a job id and answer 'unknown job'.
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


@app.get("/report/{job_id}")
def get_report(job_id: str):
    st = JOBS.get(job_id)
    p = WORK / job_id / "report.json"
    if p.exists():
        return json.loads(p.read_text())
    if st is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"job_id": job_id, **progress(job_id)}


@app.get("/")
def index():
    return {"service": "hookline-tribe", "docs": "/docs",
            "flow": "POST /analyze (video) -> GET /report/{job_id}"}
