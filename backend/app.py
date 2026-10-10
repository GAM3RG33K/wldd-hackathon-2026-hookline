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

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
TRIBE = ROOT / "tribe"
sys.path.insert(0, str(TRIBE))
import hookline_report as HL  # noqa: E402

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
<p>Upload a reel → attention preview in seconds. PREVIEW mode: real video/audio signals, synthetic-preview brain stand-in. Full TRIBE run via Colab upgrades the same report schema.</p>
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
    if em:  # emotion dots sit on the attention curve: colour = emotion, size = arousal
        s_ = em["series"]
        for i, (d, a_) in enumerate(zip(s_["dominant"], s_["arousal"])):
            if d == "neutral" or i >= T:
                continue
            size = 6 + 10 * float(a_)
            y = 100 - max(0, min(100, att[i]))
            dots += (f'<i class=edot title="{_fmt_t(i)} {d} (arousal {a_:.2f})" style="left:{100 * (i + 0.5) / T:.2f}%;'
                     f'top:{y:.2f}%;width:{size:.0f}px;height:{size:.0f}px;background:{EMO_COLORS.get(d, "#888")}"></i>')
    title = html.escape(f"Attention per second, {T}s")
    return f"""<figure class="chart" aria-label="{title}">
  <div class="plot">
    <div class="yax"><span style="top:0">100</span><span style="top:50%">50</span><span style="top:100%">0</span></div>
    <svg viewBox="0 0 {T} 100" preserveAspectRatio="none" role="img"><title>{title}</title>{''.join(parts)}</svg>
    <div class="peaklbl">{peak_lbl}</div>
    <div class="edots">{dots}</div>
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
        return JSONResponse({"error": "needs a full TRIBE report built from raw output"}, status_code=404)

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


EMO_COLORS = {"amusement": "#f5b301", "awe": "#7c3aed", "surprise": "#ec4899", "curiosity": "#06b6d4",
              "tension": "#ef4444", "outrage": "#b91c1c", "warmth": "#22c55e", "sadness": "#3b82f6", "neutral": "#555"}


def _emotion_html(r: dict) -> str:
    import html
    em = r.get("emotion")
    if not em:
        return ""
    s = em["series"]
    T = len(s["arousal"]) or 1
    cells = "".join(f'<i title="{_fmt_t(t)} {d} (arousal {a:.2f})" style="flex:1;background:{EMO_COLORS.get(d, "#555")};'
                    f'opacity:{0.15 + 0.85 * a:.2f}"></i>' for t, (d, a) in enumerate(zip(s["dominant"], s["arousal"])))
    sm = em["summary"]
    dom = ", ".join(f'{x["emotion"]} {x["share"]:.0%}' for x in sm.get("dominant_emotions", [])) or "mostly neutral"
    peaks = ", ".join(f'{_fmt_t(p["at_s"])} {p["emotion"]}' for p in sm.get("peak_moments", [])) or "none"
    legend = "".join(f'<span><i class=sw style="background:{c}"></i>{e}</span>' for e, c in EMO_COLORS.items()
                     if e != "neutral")
    br = em.get("brain_salience_r")
    return f"""<div class=emo><div class=band title="emotion per second (colour = emotion, strength = arousal)">{cells}</div>
<p><b>Emotion</b> (predicted, evoked for an average viewer): {html.escape(dom)} ·
sequence {html.escape(" → ".join(sm.get("sequence", [])) or "-")} · arc {html.escape(sm.get("arc", "-"))} ·
high-arousal share {sm.get("high_arousal_share", 0):.0%} · emotional peaks {html.escape(peaks)}</p>
<p class=muted>channels: {", ".join(em.get("channels", []))}{f" · brain salience agreement r={br:+.2f}" if br is not None else ""}
· {html.escape(em.get("note", ""))}</p><figcaption>{legend}</figcaption></div>"""


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
               f'<button>Verify top 3 edits (~2-3 min GPU each)</button></form>')
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


@app.get("/reports", response_class=HTMLResponse)
def list_reports(sort: str = "viral"):
    """All saved reports, highest viral potential first (or ?sort=recent); refreshes itself so
    batch results appear as they land."""
    import html
    rows = []
    for p in sorted(OUT.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
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
        audio = (r.get("audio") or {}).get("type") or "-"
        song = (r.get("audio") or {}).get("song")
        rows.append((s["virality"], p.stat().st_mtime, f'<tr><td><a href="/report/{job}/view">{html.escape(name)}</a></td>'
                    f'<td>{"full TRIBE" if full else "preview"}</td>'
                    f'<td><b>{s["virality"]:.0f}</b> {html.escape(s.get("virality_tag", s.get("virality_band", "")))}</td>'
                    f'<td>{s["hook_pct"]:.0f}</td><td>{s["hold_rate_pct"]:.0f}%</td><td>{s["finish_pct"]:.0f}%</td>'
                    f'<td>{html.escape(audio)}{" · &#9834; " + html.escape(song["title"]) if song else ""}</td>'
                    f'<td class=muted>{r.get("duration_s", "?")}s</td>'
                    f'<td class=muted>{__import__("time").strftime("%H:%M", __import__("time").localtime(p.stat().st_mtime))}</td></tr>'))
    rows.sort(key=(lambda x: -x[1]) if sort == "recent" else (lambda x: (-x[0], -x[1])))
    rows = [r[2] for r in rows]
    by_viral = sort != "recent"
    sort_links = (f'Sort: {"<b>viral potential</b>" if by_viral else "<a href=/reports>viral potential</a>"} · '
                  f'{"<b>newest</b>" if not by_viral else "<a href=/reports?sort=recent>newest</a>"}')
    return f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content=5><title>Hookline reports</title>
<style>:root{{--bg:#fff;--fg:#111;--muted:#666;--line:#e3e3e3;--accent:#2563eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#999;--line:#2a2a2a;--accent:#60a5fa}}}}
body{{font-family:system-ui;background:var(--bg);color:var(--fg);max-width:1200px;margin:2rem auto;padding:0 16px}}
a{{color:var(--accent)}} .muted{{color:var(--muted)}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}} th{{color:var(--muted);font-weight:500}}
.wrap{{overflow-x:auto}}</style>
<h1>Hookline reports <span class=muted style=font-weight:400>({len(rows)}, refreshes every 5s)</span></h1>
<p class=muted>{sort_links} · Viral potential, hook, hold and finish are model estimates. <a href="/">Upload a video</a></p>
<div class=wrap><table><tr><th>Reel</th><th>Run</th><th>Viral potential</th><th>Hook</th><th>Hold</th><th>Finish</th>
<th>Audio</th><th>Length</th><th>Saved</th></tr>{''.join(rows) or '<tr><td colspan=9 class=muted>No reports yet.</td></tr>'}</table></div>"""


@app.get("/report/{job_id}/view", response_class=HTMLResponse)
def view_report(job_id: str):
    import html
    p = OUT / job_id / "report.json"
    if not p.exists():
        return HTMLResponse("unknown job", status_code=404)
    r = json.loads(p.read_text())
    s = r["scores"]
    full = r.get("preview") is False or "full" in str(r.get("neural_source", ""))
    mode = "full TRIBE" if full else "preview"
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
    return f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Hookline report {job_id}</title>
{'<meta http-equiv=refresh content=10>' if str(r.get('verify_status', '')).startswith('running') else ''}
<style>
:root{{--bg:#fff;--fg:#111;--muted:#666;--line:#e3e3e3;--att:#2563eb;--surv:#9ca3af;--hook:#16a34a;--drop:#dc2626;--strong:#7c3aed}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#999;--line:#2a2a2a;--att:#60a5fa;--surv:#8b8f97}}}}
body{{font-family:system-ui;background:var(--bg);color:var(--fg);max-width:1200px;margin:2rem auto;padding:0 16px}}
a{{color:var(--att)}} .song{{color:var(--strong)}}
h1 span{{color:var(--muted);font-weight:500}} .muted{{color:var(--muted)}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1rem 0}}
.tile{{border:1px solid var(--line);border-radius:12px;padding:12px;text-align:center}}
.tile .k{{color:var(--muted);font-size:.85rem}} .tile .v{{font-size:2rem;font-weight:700}} .tile .s{{color:var(--muted)}}
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
.emo .band{{display:flex;height:18px;margin:.6rem 0 0 36px;border-radius:4px;overflow:hidden}}
.emo .band i{{display:block}} .emo p{{margin:.4rem 0 0 36px;font-size:.9rem}} .emo figcaption{{margin-left:36px}}
ol.sugs li{{margin:.6rem 0}} .badge{{display:inline-block;font-size:.75rem;border:1px solid var(--line);border-radius:10px;padding:1px 8px;margin-left:4px}}
details.variant video{{max-height:45vh;border-radius:8px;margin-top:.4rem}}
ul.transcript{{list-style:none;padding:0;columns:2;column-gap:24px}} ul.transcript li{{break-inside:avoid;margin:.2rem 0}}
a.ts{{font-variant-numeric:tabular-nums;margin-right:.4rem}}
@media (max-width:800px){{.media{{grid-template-columns:1fr}} ul.transcript{{columns:1}}}}
@media (max-width:640px){{.tiles{{grid-template-columns:repeat(2,1fr)}}}}
</style>
<h1>Hookline report <span>{job_id} ({mode})</span></h1>
<p class=muted>{html.escape(Path(r.get('video', '')).name)} · {r.get('duration_s', '?')}s{mods_txt}</p>
{gate_html}
<div class="tiles">
{tile('Hook', f"{s['hook_pct']:.0f}", s['hook_band'])}
{tile('Hold', f"{s['hold_rate_pct']:.0f}%", 'stay to the end')}
{tile('Finish', f"{s['finish_pct']:.0f}%", 'watch-through')}
{tile('Viral potential', f"{s['virality']:.0f}", s.get('virality_tag', s['virality_band']))}
</div>
<div class=media>{video_html}<div class=chartcol>{_chart_svg(r)}{_emotion_html(r)}</div></div>
{_suggestions_html(r, job_id)}
{_transcript_html(r)}
{'<h2>What to change</h2><ul>' + recs + '</ul>' if HL.SHOW_SUGGESTIONS else ''}
{png_html}
<p class=muted>Viral potential: {html.escape(s.get('virality_method', 'legacy formula'))}.
Avg attention {s.get('avg_attention', '?')} (~50 by design: attention is scored within the video).</p>
<p><i>{html.escape(r['method'])}</i> · neural_source={html.escape(str(r.get('neural_source', '?')))}</p>
<p><a href="/report/{job_id}">JSON</a> · <a href="/report/{job_id}.md">Markdown</a>
{f'<form method=post action="/report/{job_id}/regenerate?redirect=true" style="display:inline"> · <button>Regenerate from raw model output</button></form>' if (OUT / job_id / "raw" / "meta.json").exists() else ''}
{'<span class=muted> · built locally from raw TRIBE output</span>' if r.get("regenerated_locally") else ''}</p>
<script>
(() => {{
  const v = document.getElementById("reel"), ph = document.getElementById("playhead");
  const T = {len((r.get("series") or {}).get("attention") or []) or 1};
  if (v && ph) v.addEventListener("timeupdate", () => {{
    ph.hidden = false; ph.style.left = Math.min(100, 100 * v.currentTime / T) + "%";
  }});
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
