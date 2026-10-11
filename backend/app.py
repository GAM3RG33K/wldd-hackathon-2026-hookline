#!/usr/bin/env python
"""Hookline prototype backend (FastAPI front door, Colab = GPU backend).

Demo in 15 mins:
  uvicorn backend.app:app --port 8000
  open http://localhost:8000  -> upload a reel -> get report in seconds.

Pipeline per job (PREVIEW mode): upload -> probe duration -> synthetic
preview preds sized to duration -> hookline_report.analyse() on REAL video
signals (cuts/motion/audio) -> report.json/md/png. Schema matches
tribe/hookline_report.py. Full TRIBE neural preds come from
tribe/tribe_colab.ipynb (T4); drop them into backend/out/<job>/ to upgrade
a job from preview to full (see POST /jobs/colab).
"""
import hashlib
import json
import math
import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
TRIBE = ROOT / "tribe"
sys.path.insert(0, str(TRIBE))
import hookline_report as HL  # noqa: E402
from backend import theme as T  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
DEMO_SRC = TRIBE / "out" / "demo_sneaker" / "report.json"

NEURAL_SOURCE = "synthetic-preview"
METHOD_NOTE = "heuristic, within-video normalised, uncalibrated"

app = FastAPI(title="Hookline prototype")


def preview_preds(T: int, seed: str):
    import numpy as np

    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
    V = 20484
    preds = rng.normal(0, 1, (T, V)).astype(np.float64)
    if T >= 3:
        preds[1:3] += 2.0
    mid = T // 2
    preds[max(mid - 1, 0):min(mid + 2, T)] += 2.5
    dip = T // 3
    preds[max(dip - 1, 0):min(dip + 1, T)] -= 2.0
    return preds


def run_preview(job: Path, video: Path, source_name: str) -> dict:
    import numpy as np

    dur = HL.probe_duration(video)
    T = max(int(math.ceil(dur)), 1)
    preds = preview_preds(T, job.name)
    np.save(job / "preds.npy", preds)
    np.save(job / "times.npy", np.arange(T, dtype=float))
    (job / "meta.json").write_text(json.dumps({
        "video": str(video), "source": source_name,
        "transcript": None, "n_steps": T, "seconds_total": round(dur, 1),
    }))
    report, d = HL.analyse(job, video)
    report["preview"] = True
    report["neural_source"] = NEURAL_SOURCE
    import suggest as SG
    report["suggestions"] = SG.suggest(report)
    report["source_id"] = Path(source_name).stem  # list reports by the uploaded file's name
    report["local_video"] = str(video.resolve())
    (job / "report.json").write_text(json.dumps(report, indent=1))
    HL.write_md(report, job / "report.md")
    HL.write_png(report, d, job / "report.png")
    return report


INDEX = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Hookline — pre-post attention preview</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem}
.card{border:1px solid #ddd;border-radius:12px;padding:1rem;margin:1rem 0}.muted{color:#666;font-size:.9em}
button{background:#111;color:#fff;border:0;border-radius:8px;padding:.6rem 1.2rem;cursor:pointer}</style>
<h1>Hookline prototype</h1>
<p>Upload a reel → attention preview in seconds. PREVIEW mode: real video/audio signals, synthetic-preview brain stand-in. A full analysis upgrades the same report.</p>
<div class=card><h3>1. Try instant demo</h3><p><a href="/demo"><button>Open sneaker demo report</button></a>
<a href="/reports"><button>All reports (live)</button></a></p>
<p class=muted>Precomputed 28s 9:16 reel: hook 61.4, drops 0:09–0:13 + 0:26–0:29.</p></div>
<div class=card><h3>2. Upload your reel</h3>
<form id=f><input type=file name=video accept="video/*" required> <button>Analyze</button></form>
<pre id=o class=muted></pre>
<script>f.onsubmit=async e=>{e.preventDefault();o.textContent='uploading…';
const r=await fetch('/analyze',{method:'POST',body:new FormData(f)});const j=await r.json();
o.textContent=JSON.stringify(j,null,1);if(j.job_id)location.href='/report/'+j.job_id+'/view';}</script></div>
<div class=card><h3>3. Full neural (Colab T4)</h3>
<p class=muted>Run <code>tribe/tribe_colab.ipynb</code> on your video, drop <code>preds.npy/times.npy/meta.json</code> into <code>backend/out/&lt;job&gt;/</code>, then re-open the report. Schema identical.</p></div>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


@app.get("/health")
def health():
    import os
    try:
        from backend.hf_client import SPACE_ID
    except ImportError:
        from hf_client import SPACE_ID
    return {"ok": True, "neural_source": NEURAL_SOURCE,
            "method": METHOD_NOTE, "colab": "tribe/tribe_colab.ipynb (retired, see hf_space/)",
            "hf_space": SPACE_ID, "hf_configured": bool(os.environ.get("HF_TOKEN")),
            "demo_ready": DEMO_SRC.exists()}


@app.get("/demo")
def demo():
    if not DEMO_SRC.exists():
        return JSONResponse({"error": "demo not built yet"}, status_code=404)
    r = json.loads(DEMO_SRC.read_text())
    r["preview"] = True
    r["neural_source"] = NEURAL_SOURCE
    return r


@app.post("/analyze")
async def analyze(video: UploadFile = File(...), mode: str = "preview"):
    job_id = uuid.uuid4().hex[:8]
    job = OUT / job_id
    job.mkdir(parents=True)
    ext = Path(video.filename or "upload.mp4").suffix or ".mp4"
    dest = job / f"input{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    if mode == "hf":
        try:
            try:
                from backend.hf_client import analyze_via_space
            except ImportError:
                from hf_client import analyze_via_space  # ran from backend/
            report, png_tmp, md_tmp = analyze_via_space(str(dest))
            report["preview"] = False
            shutil.copy(png_tmp, job / "report.png")
            shutil.copy(md_tmp, job / "report.md")
            (job / "report.json").write_text(json.dumps(report, indent=1))
        except Exception as e:
            return JSONResponse({"error": f"space call failed: {e}"}, status_code=502)
        return {"job_id": job_id, "preview": False,
                "neural_source": report.get("neural_source", "tribev2-full"), "report": report}
    try:
        report = run_preview(job, dest, video.filename or dest.name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    try:  # one report per video: replace older previews of the same upload name
        import rescore as RS
        RS.supersede(job)
    except Exception:  # noqa: BLE001 - never fail the upload over cleanup
        pass
    return {"job_id": job_id, "preview": True,
            "neural_source": NEURAL_SOURCE, "report": report}


@app.get("/report/{job_id}.md")
def get_md(job_id: str):
    p = OUT / job_id / "report.md"
    rj = OUT / job_id / "report.json"
    if rj.exists():  # regenerate so it follows the current display rules (MVP0: no suggestions)
        HL.write_md(json.loads(rj.read_text()), p)
    if not p.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return FileResponse(p, media_type="text/markdown")


@app.get("/report/{job_id}.png")
def get_png(job_id: str):
    p = OUT / job_id / "report.png"
    if not p.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return FileResponse(p, media_type="image/png")


@app.get("/report/{job_id}")
def get_report(job_id: str):
    p = OUT / job_id / "report.json"
    if not p.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return json.loads(p.read_text())


def _is_png(p: Path) -> bool:
    return p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _fmt_t(sec) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _chart_svg(r: dict) -> str:
    """Per-second attention + survival as a full-width SVG, drawn from report.json."""
    import html
    att = r.get("series", {}).get("attention") or []
    surv = r.get("series", {}).get("survival_pct") or []
    cuts = r.get("series", {}).get("cuts_s") or []
    if not att:
        return "<p class=muted>No per-second series in this report.</p>"
    T = len(att)
    m = r.get("moments", {})
    # x in seconds (0..T), y in percent (0 top = 100%); stretched to the box, strokes stay crisp
    pts = lambda ys: " ".join(f"{i + 0.5:.2f},{100 - max(0, min(100, y)):.2f}" for i, y in enumerate(ys))
    parts = [f'<rect x="0" y="0" width="{min(3, T)}" height="100" class="hook"/>']
    for a, b in m.get("drop_zones_s", []):
        parts.append(f'<rect x="{a}" y="0" width="{max(0.5, b - a)}" height="100" class="drop"/>')
    sw = m.get("strongest_window_s")
    if sw:
        parts.append(f'<rect x="{sw[0]}" y="0" width="{sw[1] - sw[0]}" height="100" class="strong"/>')
    for g in (25, 50, 75):
        parts.append(f'<line x1="0" x2="{T}" y1="{g}" y2="{g}" class="grid"/>')
    for c in cuts:
        parts.append(f'<line x1="{c}" x2="{c}" y1="0" y2="100" class="cut"/>')
    if surv:
        parts.append(f'<polyline points="{pts(surv)}" class="surv"/>')
    parts.append(f'<polyline points="{pts(att)}" class="att"/>')
    pk = m.get("peak_attention_at_s")
    if pk is not None:
        parts.append(f'<line x1="{pk + 0.5}" x2="{pk + 0.5}" y1="0" y2="100" class="peak"/>')
    step = max(1, round(T / 10 / 5) * 5) if T > 20 else (2 if T > 8 else 1)
    ticks = "".join(f'<span style="left:{100 * t / T:.2f}%">{_fmt_t(t)}</span>' for t in range(0, T + 1, step))
    peak_lbl = f'<span class="pk" style="left:{100 * (pk + 0.5) / T:.2f}%">peak {_fmt_t(pk)}</span>' if pk is not None else ""
    dots = ""
    em = r.get("emotion")
    strong_at = {t: m["emotion"] for m in ((em or {}).get("strong") or {}).get("moments", [])
                 for t in range(m["start"], m["end"])}
    if em:  # emotion dots sit on the attention curve: colour = emotion, size = arousal
        s_ = em["series"]
        for i, (d, a_) in enumerate(zip(s_["dominant"], s_["arousal"])):
            if d in ("neutral", "calm") or i >= T:
                continue
            size = 6 + 10 * float(a_)
            y = 100 - max(0, min(100, att[i]))
            strong_d = strong_at.get(i)
            dots += (f'<i class="edot{" strong" if strong_d else ""}" data-s="{i}" '
                     f'data-label="{_fmt_t(i)} · {strong_d + " (strong)" if strong_d else d}" '
                     f'title="{_fmt_t(i)} {d} (arousal {a_:.2f})" style="left:{100 * (i + 0.5) / T:.2f}%;'
                     f'top:{y:.2f}%;width:{size:.0f}px;height:{size:.0f}px;background:{EMO_COLORS.get(d, "#888")}"></i>')
    title = html.escape(f"Attention per second, {T}s")
    return f"""<figure class="chart" aria-label="{title}">
  <div class="plot">
    <div class="yax"><span style="top:0">100</span><span style="top:50%">50</span><span style="top:100%">0</span></div>
    <svg viewBox="0 0 {T} 100" preserveAspectRatio="none" role="img"><title>{title}</title>{''.join(parts)}</svg>
    <div class="peaklbl">{peak_lbl}</div>
    <div class="edots">{dots}</div>
    <div class="etip" id="etip" hidden></div>
    <div class="playhead" id="playhead" hidden></div>
  </div>
  <div class="xax">{ticks}</div>
  <figcaption>
    <span><i class="sw att"></i>attention (avg second &asymp; 50)</span>
    <span><i class="sw surv"></i>est. viewers remaining %</span>
    <span><i class="sw hook"></i>hook window 0-3s</span>
    <span><i class="sw drop"></i>drop zone</span>
    <span><i class="sw strong"></i>strongest window</span>
    <span><i class="sw cut"></i>scene cut</span>
  </figcaption>
</figure>"""


VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}


def report_name(r: dict, job: str):
    """Name a report is listed under: its video file's name. None = not listed (video gone,
    or no real file name, e.g. old uploads stored as input.mp4 or the built-in demo)."""
    video = r.get("local_video")
    if job == "demo" or not video or not Path(video).is_file():
        return None
    name = r.get("source_id") or Path(video).stem
    return None if name in ("input", job) else name


@app.post("/import/{space_job}")
def import_space_job(space_job: str, video: str, source_id: str = ""):
    """Fetch a finished Space job's raw output and build its report locally (the Space only
    does GPU work; scoring lives here so formula changes apply instantly)."""
    import rescore as RS
    if not Path(video).is_file():
        return JSONResponse({"error": f"local video not found: {video}"}, status_code=400)
    try:
        job_dir = RS.fetch_raw(space_job)
        report = RS.rescore(job_dir, video, source_id or None)
        report["replaced"] = RS.supersede(job_dir)  # one report per video
    except Exception as e:  # noqa: BLE001 - surfaced to the caller
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
    return {"job_id": f"s-{space_job}", "report": report}


_VERIFYING = set()


@app.post("/report/{job_id}/verify")
def verify_report(job_id: str, redirect: bool = False):
    """Model-check the top editing suggestions (renders each edit, one full GPU run per edit)."""
    import threading
    from fastapi.responses import RedirectResponse
    if job_id in _VERIFYING:
        return JSONResponse({"status": "already running"}, status_code=409)
    p = OUT / job_id / "report.json"
    if not p.exists() or not (OUT / job_id / "raw" / "meta.json").exists():
        return JSONResponse({"error": "needs a full-analysis report"}, status_code=404)

    def run():
        import verify as VF
        _VERIFYING.add(job_id)
        try:
            VF.verify(OUT / job_id)
        except Exception as e:  # noqa: BLE001 - shown on the report page
            r = json.loads(p.read_text())
            r["verify_status"] = f"failed: {type(e).__name__}: {e}"[:300]
            p.write_text(json.dumps(r, indent=1))
        finally:
            _VERIFYING.discard(job_id)

    threading.Thread(target=run, daemon=True).start()
    if redirect:
        return RedirectResponse(f"/report/{job_id}/view", status_code=303)
    return {"job_id": job_id, "status": "started"}


@app.get("/report/{job_id}/variant/{name}")
def report_variant(job_id: str, name: str):
    v = OUT / job_id / "variants" / name / "edited.mp4"
    if not v.exists() or ".." in name:
        return JSONResponse({"error": "no such variant"}, status_code=404)
    return FileResponse(v, media_type="video/mp4")


@app.post("/report/{job_id}/regenerate")
def regenerate(job_id: str, redirect: bool = False):
    """Rebuild a report from its stored raw model output with the current formulas."""
    import rescore as RS
    from fastapi.responses import RedirectResponse
    if not (OUT / job_id / "raw" / "meta.json").exists():
        return JSONResponse({"error": "no raw output stored for this report"}, status_code=404)
    try:
        report = RS.rescore(OUT / job_id)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    if redirect:
        return RedirectResponse(f"/report/{job_id}/view", status_code=303)
    return {"job_id": job_id, "report": report}


def _local_video(job_id: str, r: dict):
    """The reel on this machine: the path saved with the report, else the preview job's upload."""
    demo = [TRIBE / "samples" / "sneaker.mp4"] if job_id == "demo" else []  # demo report's source reel
    for cand in (r.get("local_video"), *(OUT / job_id).glob("input.*"), *demo):
        if cand and Path(cand).is_file() and Path(cand).suffix.lower() in VIDEO_EXT:
            return Path(cand)
    return None


@app.get("/report/{job_id}/video")
def report_video(job_id: str):
    p = OUT / job_id / "report.json"
    if not p.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    v = _local_video(job_id, json.loads(p.read_text()))
    if v is None:
        return JSONResponse({"error": "local video not found"}, status_code=404)
    return FileResponse(v, media_type="video/mp4")


@app.get("/report/{job_id}/poster.jpg")
def report_poster(job_id: str):
    """One frame of the reel (at 1 s) for decks and previews; cached next to the report."""
    import subprocess
    p = OUT / job_id / "report.json"
    if not p.exists():
        return JSONResponse({"error": "unknown job"}, status_code=404)
    out = OUT / job_id / "poster.jpg"
    if not out.exists():
        v = _local_video(job_id, json.loads(p.read_text()))
        if v is None:
            return JSONResponse({"error": "local video not found"}, status_code=404)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(v), "-frames:v", "1",
                        "-vf", "scale=540:-2", str(out)], check=False)
    if not out.exists():
        return JSONResponse({"error": "could not extract a frame"}, status_code=500)
    return FileResponse(out, media_type="image/jpeg")


EMO_COLORS = {'amusement': '#f5b301', 'excitement': '#f97316', 'awe': '#7c3aed', 'admiration': '#a78bfa', 'adoration': '#f9a8d4', 'satisfaction': '#2dd4bf', 'craving': '#d97706', 'surprise': '#ec4899', 'interest': '#06b6d4', 'nostalgia': '#c4b5fd', 'romance': '#e11d48', 'hope': '#84cc16', 'empathic_pain': '#fb7185', 'sadness': '#3b82f6', 'fear': '#ef4444', 'anger': '#b91c1c', 'disgust': '#65a30d', 'awkwardness': '#a3a3a3', 'moved': '#22c55e', 'laughter': '#f5b301', 'outrage': '#b91c1c', 'sorrow': '#3b82f6'}


def _emotion_html(r: dict) -> str:
    import html
    em = r.get("emotion")
    if not em:
        return ""
    s = em["series"]
    T = len(s["arousal"]) or 1
    cells = "".join(f'<i title="{_fmt_t(t)} {d}" style="flex:1;background:{EMO_COLORS.get(d, "#555")};'
                    f'opacity:{0.15 + 0.85 * a:.2f}"></i>' for t, (d, a) in enumerate(zip(s["dominant"], s["arousal"])))
    sm = em["summary"]
    dom = ", ".join(f'{x["emotion"]} {x["share"]:.0%}' for x in sm.get("dominant_emotions", [])) or "no clear emotion"
    if sm.get("no_clear_share") is not None:
        dom += f' · no clear emotion {sm["no_clear_share"]:.0%}'
    if sm.get("confidence"):
        dom += (f' · confidence {sm["confidence"]} ({sm.get("channels_agreeing", 0):.1f} channels agree on average)'
                if sm["confidence"] != "none" else "")
    peaks = ", ".join(f'{_fmt_t(p["at_s"])} {p["emotion"]}' for p in sm.get("peak_moments", [])) or "none"
    shown = set(s["dominant"]) - {"neutral", "calm"}
    legend = "".join(f'<span><i class=sw style="background:{c}"></i>{e.replace("_", " ")}</span>'
                     for e, c in EMO_COLORS.items() if e in shown)
    br = em.get("brain_salience_r")
    st = em.get("strong") or {}
    if st:
        parts = [f'{e} {d}s' for e, d in (st.get("durations_s") or {}).items() if d]
        laugh = st.get("audible_laughter_s")
        laugh_txt = "" if laugh is None else f' · audible laughter in the audio: {laugh}s' if laugh else ' · no audible laughter in the audio'
        strong_line = (f'<p><b>Strong emotions</b> (predicted peaks the viewer feels: amusement, awe, outrage, sorrow): '
                       f'{html.escape(", ".join(parts)) if parts else "none detected"} · '
                       f'{st.get("seconds", 0)}s of {T}s ({st.get("share", 0):.0%}) → '
                       f'<b>emotional strength {st.get("score", 0):.0f}/100</b>{laugh_txt}'
                       f'{" · moments " + html.escape(", ".join(_fmt_t(m["start"]) + "-" + _fmt_t(m["end"]) + " " + m["emotion"] for m in st.get("moments", [])[:5])) if st.get("moments") else ""}</p>')
    else:
        strong_line = ""
    return f"""<div class=emo><div class=band title="emotion per second (colour = emotion, strength = arousal)">{cells}</div>
{strong_line}
<p><b>Emotion</b> (predicted, evoked for an average viewer): {html.escape(dom)} ·
sequence {html.escape(" → ".join(sm.get("sequence", [])) or "-")} · arc {html.escape(sm.get("arc", "-"))} ·
high-arousal share {sm.get("high_arousal_share", 0):.0%} · emotional peaks {html.escape(peaks)}</p>
<p class=muted>channels: {", ".join(em.get("channels", []))}{f" · brain salience agreement r={br:+.2f}" if br is not None else ""}
· {html.escape(em.get("note", ""))}</p><figcaption>{legend}</figcaption></div>"""


def _music_html(r: dict) -> str:
    import html
    mu = (r.get("music") or {}).get("summary")
    mood = (r.get("emotion") or {}).get("music_mood")
    if not mu and not mood:
        return ""
    bits = []
    if mu:
        bits.append(f'{mu["tempo_bpm"]:.0f} BPM')
        bits.append(mu["mode_label"])
        if mu.get("buildups"):
            bits.append("build-ups " + ", ".join(f'{_fmt_t(x["start"])}-{_fmt_t(x["end"])} (+{x["rise_db"]:.0f} dB)'
                                                 for x in mu["buildups"][:3]))
        if mu.get("drops"):
            bits.append("drops at " + ", ".join(_fmt_t(x["at"]) for x in mu["drops"][:4]))
        if mu.get("beat_cut_sync") is not None:
            bits.append(f'cuts on the beat {mu["beat_cut_sync"]:.0%}')
    if mood:
        bits.append("mood " + " → ".join(mood.get("sequence", [])[:6]) + " (" +
                    ", ".join(f"{k} {v:.0%}" for k, v in mood.get("shares", {}).items()) + ")")
    return f'<p><b>Music</b>: {html.escape(" · ".join(bits))}</p>'


def _caption_html(r: dict) -> str:
    """The creator's caption (only when given): verdict, evidence, emotion it adds, effect on viral potential."""
    import html
    import caption as CAP
    cap = r.get("caption")
    if not cap:
        return ""
    ev = cap.get("evidence") or {}
    ok = cap.get("relevance", 0) > 0
    rows = []
    if ev.get("frames_z") is not None:
        rows.append(f'<li>Frames: fits this reel better than {ev.get("frames_beats_other_reels", 0):.0%} of other reels '
                    f'(z {ev["frames_z"]:.1f}; a generic caption fits every reel about equally)</li>')
    if ev.get("transcript_z") is not None:
        rows.append(f'<li>Speech: closer to the transcript than {ev.get("transcript_beats_controls", 0):.0%} of unrelated '
                    f'captions (z {ev["transcript_z"]:.1f})</li>')
    emo = (cap.get("emotion") or {}).get("top") or []
    if ok and emo:
        rows.append("<li>Adds emotional context: " + html.escape(", ".join(f"{e} {p:.0%}" for e, p in emo)) + "</li>")
    effect = (f'<b>+{cap.get("virality_points", 0):.1f}</b> viral potential (relevance {cap["relevance"]:.2f})'
              if ok else "no effect: ignored")
    return (f'<section class=caption><h2>Caption <span class=muted>({html.escape(cap.get("verdict", ""))})</span></h2>'
            f'<blockquote>{html.escape(cap["text"])}</blockquote>'
            f'<p>Effect: {effect}. A caption counts from z {CAP.Z_LOW} on frames or speech; full effect at z {CAP.Z_HIGH}.</p>'
            f'<ul class=muted>{"".join(rows)}</ul></section>')


def _virality_html(r: dict) -> str:
    """Why this viral potential: each factor's score, points, and pull vs a typical reel (diverging bar)."""
    import html
    s = r.get("scores") or {}
    b = s.get("virality_breakdown") or []
    if not b:
        return ""
    if s.get("virality_summary") is None:  # report scored before the summary existed
        s["virality_summary"] = HL.virality_summary(s)
    span = max(10.0, max(abs(x["vs_typical"]) for x in b))
    top = max(b, key=lambda x: abs(x["vs_typical"]))
    rows = []
    for x in b:
        d = x["vs_typical"]
        w = 50 * abs(d) / span  # % of the bar track; the centre line is a typical reel
        bar = (f'<span class=vbar><i class="{"up" if d >= 0 else "down"}" '
               f'style="{"left:50%" if d >= 0 else f"left:{50 - w:.1f}%"};width:{w:.1f}%"></i></span>')
        pts = "" if x.get("points") is None else f'{x["points"]:.1f}'
        rows.append(f'<tr{" class=top" if x is top else ""}><td>{html.escape(x["label"])}</td>'
                    f'<td class=num>{x["value"]:.0f}</td><td class=num>×{x["weight"]}</td><td class=num>{pts}</td>'
                    f'<td>{bar}</td><td class="num {"pos" if d >= 0 else "neg"}">{d:+.1f}</td>'
                    f'<td class=muted>{html.escape(x["explain"])}</td></tr>')
    base = s.get("virality_base")
    lifts = [(x["label"].lower(), x["points"]) for x in b if x["key"] in ("emotion_lift", "caption") and x.get("points")]
    total = (f'{base:.1f} from attention' + "".join(f" + {p:.1f} {lbl}" for lbl, p in lifts) +
             f' = <b>{s["virality"]:.0f}</b>' if base is not None else f'<b>{s["virality"]:.0f}</b>')
    return (f'<section class=why><h2>Why viral potential {s["virality"]:.0f} '
            f'<span class=muted>({html.escape(s.get("virality_tag", s.get("virality_band", "")))})</span></h2>'
            f'<p>{html.escape(s.get("virality_summary") or "")}</p>'
            f'<div class=wrap><table><tr><th>Factor</th><th>Score /100</th><th>Weight</th><th>Points</th>'
            f'<th>vs a typical reel (factor at 50)</th><th></th><th>What it measures here</th></tr>{"".join(rows)}</table></div>'
            f'<p class=muted>Total: {total}. Uncalibrated heuristic: weights are set by hand until fitted on real view '
            f'counts; it judges the video only, never the account.</p></section>')


def _suggestions_html(r: dict, job_id: str) -> str:
    import html
    sugs = r.get("suggestions") or []
    if not sugs:
        return ""
    status = r.get("verify_status")
    running = bool(status and status.startswith("running"))
    can_verify = (OUT / job_id / "raw" / "meta.json").exists()
    items = []
    for s in sugs:
        v = s.get("verification")
        if v and v.get("verdict") in ("improved", "worse", "no clear change"):
            color = {"improved": "var(--hook)", "worse": "var(--drop)"}.get(v["verdict"], "var(--muted)")
            name = Path(v.get("edited_video", "")).parent.name
            badge = (f'<span class=badge style="border-color:{color};color:{color}">model-checked: {v["verdict"]}'
                     f' · engagement {v["engagement_change_pct"]:+.1f}%'
                     f'{" · first 3s " + format(v["first_3s_change_pct"], "+.1f") + "%" if s["edit"] == "cold_open" else ""}'
                     f'</span> <details class=variant><summary>watch the edited version</summary>'
                     f'<video controls preload=none src="/report/{job_id}/variant/{name}"></video>'
                     f'{"<p class=muted>Simulation: " + html.escape(v["approximate"]) + "</p>" if v.get("approximate") else ""}'
                     f'</details>')
        elif v:
            badge = f'<span class="badge muted">{html.escape(v.get("verdict", ""))}: {html.escape(v.get("reason", ""))}</span>'
        else:
            badge = '<span class="badge muted">not model-checked yet</span>'
        items.append(f'<li><b>{html.escape(s["at"])}</b> {html.escape(s["suggestion"])} {badge}'
                     f'<br><span class=muted>Why: {html.escape(s["why"])}. Est. viewers lost here: '
                     f'{s["impact"]["viewers_lost_pct"]:.1f}% (model estimate).</span></li>')
    btn = ""
    if can_verify and not running:
        btn = (f'<form method=post action="/report/{job_id}/verify?redirect=true" style="display:inline">'
               f'<button>Verify top 3 edits (~2-3 min each)</button></form>')
    st = f'<span class=muted>{html.escape(status)}</span>' if status else ""
    return (f'<h2>Editing suggestions <span class=muted style="font-weight:400;font-size:1rem">'
            f'ranked by viewers lost</span></h2>{btn} {st}<ol class=sugs>{"".join(items)}</ol>')


def _transcript_html(r: dict) -> str:
    import html
    tr = r.get("transcript")
    if not isinstance(tr, dict) or ("segments" not in tr and "status" not in tr):
        return ""
    head = (f'<h2>Transcript {"used for judging" if tr.get("used_by_model") else "(not used for judging)"}</h2>'
            f'<p class=muted>{html.escape(str(tr.get("status", "")))}'
            f'{" · " + html.escape(str(tr.get("model"))) if tr.get("model") else ""}'
            f'{" · detected " + html.escape(str(tr.get("detected"))) if tr.get("detected") else ""}'
            f'{" · " + str(tr.get("words")) + " words" if tr.get("words") is not None else ""}'
            f'{" · guard dropped " + html.escape(str(tr.get("dropped_by_guard"))) if tr.get("dropped_by_guard") else ""}'
            f'</p>')
    segs = tr.get("segments") or []
    if not segs:
        return head
    rows = "".join(f'<li><a class=ts href="#" data-t="{s["start"]}">{_fmt_t(s["start"])}</a> '
                   f'{html.escape(s["text"])}</li>' for s in segs)
    return head + f'<ul class=transcript>{rows}</ul>'


def analysed_at(report_json: Path) -> float:
    """When the reel was analysed: the model output's time, not report.json's (re-scoring rewrites that)."""
    d = report_json.parent
    for f in (d / "raw" / "preds.npy", d / "preds.npy", report_json):  # preds are never rewritten
        if f.exists():
            return f.stat().st_mtime
    return 0.0


REPORT_SORTS = {"viral": "viral first", "latest": "latest first", "oldest": "oldest first"}


@app.post("/launch-ui")
def launch_ui(request: Request):
    """Open the Hookline user interface (TUI) in a new Terminal window. Local demo only: refuses
    anything but requests from this machine, and only ever runs ./run-demo.sh."""
    import subprocess
    if (request.client.host if request.client else "") not in ("127.0.0.1", "::1", "localhost"):
        return JSONResponse({"error": "Only available on the presenting machine."}, status_code=403)
    if sys.platform != "darwin":
        return JSONResponse({"error": "Launching a terminal is only set up for macOS. Run ./run-demo.sh instead."})
    cmd = f"cd {json.dumps(str(ROOT))} && ./run-demo.sh"
    r = subprocess.run(["osascript", "-e", f'tell application "Terminal" to do script {json.dumps(cmd)}',
                        "-e", 'tell application "Terminal" to activate'], capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        return JSONResponse({"error": f"Could not open Terminal: {r.stderr.strip()[:160]}. Run ./run-demo.sh instead."})
    return {"message": "The user interface is opening in a new Terminal window."}


@app.get("/pitch", response_class=HTMLResponse)
def pitch():
    """Pitch deck rendered from pitch/deck.md with live report blocks (edit the file, refresh)."""
    import importlib
    P = importlib.reload(importlib.import_module("backend.pitch"))
    return HTMLResponse(P.render(sys.modules[__name__]))


@app.get("/reports", response_class=HTMLResponse)
def list_reports(sort: str = "viral"):
    """All saved reports: viral first (default), latest first or oldest first; refreshes itself so
    batch results appear as they land."""
    import html
    import time as _time
    sort = "latest" if sort == "recent" else sort if sort in REPORT_SORTS else "viral"
    rows = []
    for p in OUT.glob("*/report.json"):
        try:
            r = json.loads(p.read_text())
            s = r["scores"]
        except Exception:
            continue
        job = p.parent.name
        name = report_name(r, job)
        if name is None:
            continue
        full = r.get("preview") is False or "full" in str(r.get("neural_source", ""))
        if not full:  # the list shows full analyses only; quick previews stay on disk, reachable by URL
            continue
        audio = (r.get("audio") or {}).get("type") or "-"
        song = (r.get("audio") or {}).get("song")
        at = analysed_at(p)
        v = s["virality"]
        tag = s.get("virality_tag", s.get("virality_band", ""))
        cap = r.get("caption")
        rows.append((v, at, full, f'<tr><td class=reel><a href="/report/{job}/view">{html.escape(name)}</a>'
                    f'<span class=sub>{"Full analysis" if full else "Quick preview"}'
                    f'{(" + caption matched" if cap.get("relevance", 0) > 0 else " + caption ignored") if cap else ""}</span></td>'
                    f'<td><div class=vp><b class=num>{v:.0f}</b><span class=meter><i style="width:{max(0, min(100, v)):.0f}%"></i></span>'
                    f'<span class="chip {T.tier_class(tag)}">{html.escape(tag)}</span></div></td>'
                    f'<td class=num>{s["hook_pct"]:.0f}</td><td class=num>{s["hold_rate_pct"]:.0f}%</td><td class=num>{s["finish_pct"]:.0f}%</td>'
                    f'<td class=muted>{html.escape(audio.replace("_", " "))}{" · &#9834; " + html.escape(song["title"]) if song else ""}</td>'
                    f'<td class="num muted">{r.get("duration_s", 0):.0f} s</td>'
                    f'<td class="num muted mono">{_time.strftime("%b %d %H:%M", _time.localtime(at))}</td></tr>'))
    key = {"viral": lambda x: (-x[0], -x[1]), "latest": lambda x: -x[1], "oldest": lambda x: x[1]}[sort]
    rows = [x[3] for x in sorted(rows, key=key)]
    options = "".join(f'<option value={k}{" selected" if k == sort else ""}>{label}</option>' for k, label in REPORT_SORTS.items())
    empty = '<tr><td colspan=7 class=empty>No reports yet. Analyze a reel in the TUI (Enter or a) and it appears here.</td></tr>'
    return f"""<!doctype html><html lang=en><head>{T.page_head("Hookline reports")}
<script>/* live list: reload every 5s, but never while the sort menu is open */
setInterval(()=>{{if(document.activeElement?.id!=="sort")location.reload()}},5000)</script>
<style>{T.TOKENS}
/* Layout: one column, max 1180px; header bar, summary line, one table panel. */
.wrapper{{max-width:1180px;margin:0 auto;padding-block:28px 48px;padding-inline:clamp(16px,4vw,32px);display:flex;flex-direction:column;gap:20px}}
.bar{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}}
.brand{{font:700 1.05rem var(--display);display:flex;align-items:center;gap:8px}}
.brand i{{width:9px;height:9px;border-radius:50%;background:var(--accent)}}
.nav{{display:flex;gap:16px;font-size:.92rem}}
h1{{font-size:clamp(1.7rem,3vw,2.4rem)}}
.summary{{display:flex;flex-wrap:wrap;gap:10px 24px;align-items:center;color:var(--muted);font-size:.95rem}}
.summary b{{color:var(--ink);font-family:var(--mono);font-weight:500}}
label.sort{{display:flex;gap:8px;align-items:center;margin-left:auto;font-size:.9rem;color:var(--muted)}}
select{{font:500 .92rem var(--sans);color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:6px 10px}}
.panel{{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.94rem}}
th,td{{text-align:left;padding:12px 16px;border-bottom:1px solid var(--line);vertical-align:middle}}
tbody tr:last-child td{{border-bottom:0}} tbody tr:hover{{background:var(--sunken)}}
th{{font:500 .72rem/1.2 var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted);background:var(--sunken);white-space:nowrap}}
th.num,td.num{{text-align:right;white-space:nowrap}}
td.reel a{{font-weight:600;text-decoration:none;color:var(--ink)}} td.reel a:hover{{color:var(--accent)}}
td.reel .sub{{display:block;font-size:.8rem;color:var(--muted)}}
.vp{{display:flex;align-items:center;gap:10px;white-space:nowrap}}
.vp b{{font:600 1.15rem var(--mono);min-width:2ch;text-align:right}}
.meter{{width:72px;height:6px;border-radius:3px;background:var(--sunken);overflow:hidden}} .meter i{{display:block;height:100%;background:var(--accent)}}
.empty{{text-align:center;color:var(--muted);padding:40px 16px}}
.note{{font-size:.85rem;color:var(--muted)}}
</style></head><body><div class=wrapper>
<div class=bar><span class=brand><i></i>Hookline</span><nav class=nav><a href="/pitch">Pitch deck</a><a href="/">Upload a video</a></nav></div>
<h1>Reports</h1>
<div class=summary><span><b>{len(rows)}</b> reels, full analysis</span><span>refreshes every 5 s</span>
<label class=sort for=sort>Sort <select id=sort onchange="location.search='?sort='+this.value">{options}</select></label></div>
<div class=panel><table><thead><tr><th>Reel</th><th>Viral potential</th><th class=num>Hook</th><th class=num>Hold</th><th class=num>Finish</th>
<th>Audio</th><th class=num>Length</th><th class=num>Analysed</th></tr></thead><tbody>{''.join(rows) or empty}</tbody></table></div>
<p class=note>Viral potential, hook, hold and finish are model estimates: within-video, uncalibrated. Views and follower counts are never inputs.</p>
</div></body></html>"""


@app.get("/report/{job_id}/view", response_class=HTMLResponse)
def view_report(job_id: str):
    import html
    p = OUT / job_id / "report.json"
    if not p.exists():
        return HTMLResponse("unknown job", status_code=404)
    r = json.loads(p.read_text())
    s = r["scores"]
    full = r.get("preview") is False or "full" in str(r.get("neural_source", ""))
    mode = ("Full analysis" + (" + caption" if r.get("caption") else "")) if full else "Quick preview"
    md = OUT / job_id / "report.md"
    if not md.exists() or md.read_text(errors="ignore").startswith('{"error"'):
        HL.write_md(r, md)  # Space jobs may come back without markdown; it only needs report.json
    png = OUT / job_id / "report.png"
    png_html = (f'<details><summary>Full chart (PNG)</summary><img src="/report/{job_id}.png" alt="chart"></details>'
                if _is_png(png) and HL.SHOW_SUGGESTIONS else "")
    gate = r.get("speech_gate")
    audio = r.get("audio") or {}
    gate_html = ""
    if gate:
        kinds = {"song": "Song", "speech_over_music": "Speech over music", "speech": "Speech",
                 "instrumental": "Instrumental music", "humming": "Humming / chant", "other": "No clear speech"}
        rt = gate.get("ratios", {})
        mix = " · ".join(f"{k} {v:.0%}" for k, v in rt.items() if k != "silence" and v)
        song = audio.get("song")
        song_html = (f' · <b class=song>&#9834; {html.escape(song["title"])} - {html.escape(song["artist"])}</b> '
                     f'<span class=muted>(on trending list: {html.escape(song.get("list", ""))}, '
                     f'match {song.get("similarity", 0):.2f})</span>' if song else
                     ' · <span class=muted>no match in trending catalog</span>')
        reuse = audio.get("reused_by") or []
        reuse_html = f" · same audio in {len(reuse)} other analysed reel(s)" if reuse else ""
        gate_html = (f'<p><b>Audio: {kinds.get(gate.get("audio_type"), "not classified (older report)")}</b>{song_html}{reuse_html}</p>'
                     f'<p class=muted>{html.escape(mix or "no audio detected")} · '
                     f'<b>{html.escape(gate.get("decision", ""))}</b> ({html.escape(gate.get("reason", ""))}) · '
                     f'song/trend info is context only, not a score input</p>')
    vid = _local_video(job_id, r)
    video_html = (f'<div class=player><video id=reel controls preload=metadata src="/report/{job_id}/video"></video>'
                  f'<p class=muted>Local file: <code>{html.escape(str(vid))}</code></p></div>' if vid else
                  '<p class=muted>Local video not found on this machine.</p>')
    mods = r.get("modalities")
    mods_txt = (" · modalities: " + ", ".join(k for k, v in mods.items() if v)) if mods else ""
    recs = "".join(f"<li><b>{html.escape(x['at'])}</b> {html.escape(x['issue'])}. {html.escape(x['fix'])}</li>"
                   for x in r["recommendations"])
    tile = lambda name, val, sub="": (f'<div class="tile"><div class="k">{name}</div><div class="v">{val}</div>'
                                      f'<div class="s">{sub}</div></div>')
    return f"""<!doctype html><html lang=en><head>{T.page_head(f"Hookline report · {html.escape(report_name(r, job_id) or job_id)}")}
{'<meta http-equiv=refresh content=10>' if str(r.get('verify_status', '')).startswith('running') else ''}
<style>{T.TOKENS}
/* Layout: one reading column, max 1180px; tiles, then video + chart, then explanation sections. */
body{{max-width:1180px;margin:0 auto;padding-block:28px 56px;padding-inline:clamp(16px,4vw,32px)}}
.song{{color:var(--strong)}}
h1{{font-size:clamp(1.6rem,3vw,2.3rem)}} h1 span{{color:var(--muted);font-weight:500;font-family:var(--sans);font-size:.6em}}
h2{{font-size:1.25rem;margin:2rem 0 .6rem}} h2 span{{font-family:var(--sans);font-weight:400;font-size:.8em}}
p{{margin:.5em 0}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--gap);margin:1.2rem 0}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:14px 16px;text-align:left}}
.tile .k{{font:500 .72rem/1.2 var(--mono);letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}}
.tile .v{{font:700 clamp(1.8rem,3.4vw,2.5rem)/1.15 var(--display)}} .tile .s{{color:var(--muted);font-size:.9rem}}
section.why table,table{{font-variant-numeric:tabular-nums}}
section.why,section.caption,ul.transcript{{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:16px 20px}}
ol.sugs{{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:16px 20px 16px 44px}}
section.why h2,section.caption h2{{margin-top:0}}
.chart{{margin:1.5rem 0;width:100%}}
.plot{{position:relative;height:30vh;min-height:180px;margin-left:36px;border-left:1px solid var(--line);border-bottom:1px solid var(--line)}}
.plot svg{{position:absolute;inset:0;width:100%;height:100%;overflow:visible}}
.yax span{{position:absolute;left:-34px;transform:translateY(-50%);font-size:.75rem;color:var(--muted)}}
.xax{{position:relative;height:1.4rem;margin-left:36px;font-size:.75rem;color:var(--muted)}}
.xax span{{position:absolute;transform:translateX(-50%);top:4px}}
.peaklbl .pk{{position:absolute;top:-1.2rem;transform:translateX(-50%);font-size:.75rem;color:var(--strong)}}
svg .att{{fill:none;stroke:var(--att);stroke-width:2.5;vector-effect:non-scaling-stroke}}
svg .surv{{fill:none;stroke:var(--surv);stroke-width:2;stroke-dasharray:6 4;vector-effect:non-scaling-stroke}}
svg .hook{{fill:var(--hook);opacity:.12}} svg .drop{{fill:var(--drop);opacity:.12}} svg .strong{{fill:var(--strong);opacity:.10}}
svg .grid{{stroke:var(--line);vector-effect:non-scaling-stroke}}
svg .cut{{stroke:var(--muted);stroke-dasharray:2 3;vector-effect:non-scaling-stroke;opacity:.6}}
svg .peak{{stroke:var(--strong);stroke-width:1.5;vector-effect:non-scaling-stroke}}
figcaption{{display:flex;flex-wrap:wrap;gap:14px;font-size:.8rem;color:var(--muted);margin-top:.4rem}}
.sw{{display:inline-block;width:14px;height:10px;margin-right:5px;border-radius:2px;vertical-align:middle}}
.sw.att{{background:var(--att)}} .sw.surv{{background:var(--surv)}} .sw.hook{{background:var(--hook);opacity:.4}}
.sw.drop{{background:var(--drop);opacity:.4}} .sw.strong{{background:var(--strong);opacity:.35}}
.sw.cut{{border-left:2px dashed var(--muted);border-radius:0;width:0}}
details img{{max-width:100%;border:1px solid var(--line);border-radius:12px;margin-top:.5rem}}
.media{{display:grid;grid-template-columns:minmax(180px,260px) 1fr;gap:20px;align-items:start}}
.player video{{width:100%;max-height:60vh;border-radius:12px;background:#000}}
.player code{{font-size:.7rem;word-break:break-all}}
.playhead{{position:absolute;top:0;bottom:0;width:2px;background:var(--fg);opacity:.8;pointer-events:none}}
.edots .edot{{position:absolute;transform:translate(-50%,-50%);border-radius:50%;border:1.5px solid var(--bg);
  box-shadow:0 0 0 1px rgba(0,0,0,.25);cursor:default}}
.edots .edot.strong{{box-shadow:0 0 0 2px var(--bg),0 0 0 4px var(--fg)}}
.edots .edot.on{{transform:translate(-50%,-50%) scale(1.6);z-index:2}}
.etip{{position:absolute;transform:translate(-50%,calc(-100% - 14px));padding:3px 8px;border-radius:6px;font-size:.75rem;
  white-space:nowrap;background:var(--fg);color:var(--bg);pointer-events:none;z-index:3}}
.emo .band{{display:flex;height:18px;margin:.6rem 0 0 36px;border-radius:4px;overflow:hidden}}
.emo .band i{{display:block}} .emo p{{margin:.4rem 0 0 36px;font-size:.9rem}} .emo figcaption{{margin-left:36px}}
ol.sugs li{{margin:.6rem 0}} .badge{{display:inline-block;font-size:.75rem;border:1px solid var(--line);border-radius:10px;padding:1px 8px;margin-left:4px}}
details.variant video{{max-height:45vh;border-radius:8px;margin-top:.4rem}}
ul.transcript{{list-style:none;padding:0;columns:2;column-gap:24px}} ul.transcript li{{break-inside:avoid;margin:.2rem 0}}
a.ts{{font-variant-numeric:tabular-nums;margin-right:.4rem}}
@media (max-width:800px){{.media{{grid-template-columns:1fr}} ul.transcript{{columns:1}}}}
@media (max-width:640px){{.tiles{{grid-template-columns:repeat(2,1fr)}}}}
.why table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
.why th,.why td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:middle}}
.why th{{color:var(--muted);font-weight:500;font-size:.85rem}} .why .num{{text-align:right;white-space:nowrap}}
.why tr.top td:first-child{{font-weight:700}} .why .pos{{color:var(--hook)}} .why .neg{{color:var(--drop)}}
.why .wrap{{overflow-x:auto}}
.caption blockquote{{margin:.4rem 0;padding:.6rem 1rem;border-left:3px solid var(--strong);background:color-mix(in srgb,var(--line) 35%,transparent);border-radius:6px;white-space:pre-wrap}}
.vbar{{position:relative;display:block;width:160px;height:10px;background:var(--line);border-radius:5px}}
.vbar::after{{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--muted)}}
.vbar i{{position:absolute;top:0;bottom:0;border-radius:5px}} .vbar i.up{{background:var(--hook)}} .vbar i.down{{background:var(--drop)}}
</style>
<h1>Hookline report <span>{job_id} ({mode})</span></h1>
<p class=muted>{html.escape(Path(r.get('video', '')).name)} · {r.get('duration_s', '?')}s{mods_txt}</p>
{gate_html}
{_music_html(r)}
<div class="tiles">
{tile('Hook', f"{s['hook_pct']:.0f}", s['hook_band'])}
{tile('Hold', f"{s['hold_rate_pct']:.0f}%", 'stay to the end')}
{tile('Finish', f"{s['finish_pct']:.0f}%", 'watch-through')}
{tile('Viral potential', f"{s['virality']:.0f}", s.get('virality_tag', s['virality_band']))}
</div>
<div class=media>{video_html}<div class=chartcol>{_chart_svg(r)}{_emotion_html(r)}</div></div>
{_caption_html(r)}
{_virality_html(r)}
{_suggestions_html(r, job_id)}
{_transcript_html(r)}
{'<h2>What to change</h2><ul>' + recs + '</ul>' if HL.SHOW_SUGGESTIONS else ''}
{png_html}
<p class=muted>Viral potential: {html.escape(s.get('virality_method', 'legacy formula'))}.
Avg attention {s.get('avg_attention', '?')} (~50 by design: attention is scored within the video).</p>
<p><i>{html.escape(r['method'])}</i> · {"Full analysis by the Virality Predictor" if full else "Quick preview (synthetic brain stand-in)"}</p>
<p><a href="/report/{job_id}">JSON</a> · <a href="/report/{job_id}.md">Markdown</a>
{f'<form method=post action="/report/{job_id}/regenerate?redirect=true" style="display:inline"> · <button>Regenerate from raw model output</button></form>' if (OUT / job_id / "raw" / "meta.json").exists() else ''}
{'<span class=muted> · built locally from the raw analysis</span>' if r.get("regenerated_locally") else ''}</p>
<script>
(() => {{
  const v = document.getElementById("reel"), ph = document.getElementById("playhead");
  const T = {len((r.get("series") or {}).get("attention") or []) or 1};
  const tip = document.getElementById("etip");
  const dots = {{}};
  document.querySelectorAll(".edot").forEach(d => dots[d.dataset.s] = d);
  let lit = null;
  const sync = () => {{
    ph.hidden = false; ph.style.left = Math.min(100, 100 * v.currentTime / T) + "%";
    // emotion label when the playhead reaches a dot
    const d = dots[Math.floor(v.currentTime)];
    if (lit && lit !== d) {{ lit.classList.remove("on"); lit = null; }}
    if (d && tip) {{
      d.classList.add("on"); lit = d;
      tip.textContent = d.dataset.label; tip.style.left = d.style.left; tip.style.top = d.style.top;
      tip.style.background = d.style.background; tip.hidden = false;
    }} else if (tip) {{ tip.hidden = true; }}
  }};
  // playback and scrubbing (a paused seek fires 'seeked', not 'timeupdate')
  if (v && ph) ["timeupdate", "seeked"].forEach(ev => v.addEventListener(ev, sync));
  document.querySelectorAll("a.ts").forEach(a => a.addEventListener("click", e => {{
    e.preventDefault(); if (v) {{ v.currentTime = +a.dataset.t; v.play(); }}
  }}));
}})();
</script>"""


@app.post("/jobs/colab")
def colab_instructions(job_id: str = ""):
    return {"how": "open tribe/tribe_colab.ipynb (T4) → upload backend/out/<job>/input.* "
                   "→ run --device cuda → copy preds.npy/times.npy/meta.json back → "
                   "POST /analyze already produced preview; full report re-runs hookline_report.analyse().",
            "job": job_id or "<job_id>"}
