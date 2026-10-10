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


@app.get("/report/{job_id}/view", response_class=HTMLResponse)
def view_report(job_id: str):
    p = OUT / job_id / "report.json"
    if not p.exists():
        return HTMLResponse("unknown job", status_code=404)
    r = json.loads(p.read_text())
    recs = "".join(f"<li><b>{x['at']}</b> {x['issue']}. {x['fix']}</li>" for x in r["recommendations"])
    return f"""<!doctype html><meta charset=utf-8><title>Hookline report {job_id}</title>
<style>body{{font-family:system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}}</style>
<h1>Hookline report <span style=color:#666>{job_id} (preview)</span></h1>
<p>Hook <b>{r['scores']['hook_pct']}%</b> ({r['scores']['hook_band']}) · Hold
<b>{r['scores']['hold_rate_pct']}%</b> · Finish <b>{r['scores']['finish_pct']}%</b> ·
Virality <b>{r['scores']['virality']}</b> ({r['scores']['virality_band']})</p>
<img src="/report/{job_id}.png" style="max-width:100%;border:1px solid #ddd;border-radius:12px">
<h2>What to change</h2><ul>{recs}</ul>
<p><i>{r['method']}</i> · neural_source={r.get('neural_source', '?')}</p>
<p><a href="/report/{job_id}">JSON</a> · <a href="/report/{job_id}.md">Markdown</a></p>"""


@app.post("/jobs/colab")
def colab_instructions(job_id: str = ""):
    return {"how": "open tribe/tribe_colab.ipynb (T4) → upload backend/out/<job>/input.* "
                   "→ run --device cuda → copy preds.npy/times.npy/meta.json back → "
                   "POST /analyze already produced preview; full report re-runs hookline_report.analyse().",
            "job": job_id or "<job_id>"}
