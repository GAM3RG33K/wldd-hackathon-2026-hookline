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
<div class=card><h3>1. Try instant demo</h3><p><a href="/demo"><button>Open sneaker demo report</button></a></p>
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
    title = html.escape(f"Attention per second, {T}s")
    return f"""<figure class="chart" aria-label="{title}">
  <div class="plot">
    <div class="yax"><span style="top:0">100</span><span style="top:50%">50</span><span style="top:100%">0</span></div>
    <svg viewBox="0 0 {T} 100" preserveAspectRatio="none" role="img"><title>{title}</title>{''.join(parts)}</svg>
    <div class="peaklbl">{peak_lbl}</div>
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
                if _is_png(png) else "")
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
<style>
:root{{--bg:#fff;--fg:#111;--muted:#666;--line:#e3e3e3;--att:#2563eb;--surv:#ea8a0c;--hook:#16a34a;--drop:#dc2626;--strong:#7c3aed}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--muted:#999;--line:#2a2a2a;--att:#60a5fa;--surv:#f5a524}}}}
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
<div class=media>{video_html}<div class=chartcol>{_chart_svg(r)}</div></div>
{_transcript_html(r)}
{'<h2>What to change</h2><ul>' + recs + '</ul>' if HL.SHOW_SUGGESTIONS else ''}
{png_html}
<p class=muted>Viral potential: {html.escape(s.get('virality_method', 'legacy formula'))}.
Avg attention {s.get('avg_attention', '?')} (~50 by design: attention is scored within the video).</p>
<p><i>{html.escape(r['method'])}</i> · neural_source={html.escape(str(r.get('neural_source', '?')))}</p>
<p><a href="/report/{job_id}">JSON</a> · <a href="/report/{job_id}.md">Markdown</a></p>
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
