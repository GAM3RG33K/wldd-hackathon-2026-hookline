#!/usr/bin/env python
"""Hookline terminal UI — browse samples, analyze, review reports.

Stdlib only. Talks to the local backend (localhost:8000) and, when
configured, the HF Space.  Run:  python3 tui.py
"""
import csv
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SAMPLES = ROOT / "samples"
BACKEND = "http://localhost:8000"
OUT = ROOT / "backend" / "out"
STRATA = ("face", "faceless", "outliers")

B = "\033[1m"
D = "\033[2m"
R = "\033[0m"
C = "\033[36m"
G = "\033[32m"
Y = "\033[33m"
RD = "\033[31m"


def get(url, token="", timeout=30):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post_file(url, path, token="", extra=""):
    import uuid as _uuid
    boundary = _uuid.uuid4().hex
    data = open(path, "rb").read()
    name = os.path.basename(path)
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"{name}\"\r\n"
            f"Content-Type: video/mp4\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url + extra,
                                 data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def load_corpus():
    corpus = {}
    for s in STRATA:
        meta = SAMPLES / s / "metadata.csv"
        rows = list(csv.DictReader(meta.open())) if meta.exists() else []
        vids = {p.stem: p for p in (SAMPLES / s / "videos").glob("*.mp4")}
        corpus[s] = [dict(r, _path=str(vids[r["video_id"]])) for r in rows if r["video_id"] in vids]
    return corpus


def band_color(b):
    return {"Bad": RD, "Weak": Y, "Average": C, "Strong": G, "Viral-grade": G}.get(b, "")


def show_report(r, job=""):
    s = r["scores"]
    print(f"\n{B}Hookline report {D}{job}{R}")
    for k in ("hook_pct", "hold_rate_pct", "finish_pct", "virality"):
        band = s.get("hook_band" if "hook" in k else "virality_band", "") if "hook" in k or k == "virality" else ""
        print(f"  {k:15s} {B}{s[k]}{R} {band_color(band)}{band}{R}")
    m = r["moments"]
    fmt = lambda x: f"{x // 60}:{x % 60:02d}"
    print(f"  peak {fmt(m['peak_attention_at_s'])} · strongest {fmt(m['strongest_window_s'][0])}-{fmt(m['strongest_window_s'][1])} "
          f"· weakest {fmt(m['weakest_window_s'][0])}-{fmt(m['weakest_window_s'][1])}")
    print(f"  drops: {', '.join(fmt(a) + '-' + fmt(b) for a, b in m['drop_zones_s'])}")
    print(f"\n{B}What to change{R}")
    for x in r["recommendations"]:
        print(f"  {G}{x['at']}{R} {x['issue']}\n     → {x['fix']}")
    print(f"{D}{r.get('method', '')} · {r.get('neural_source', '')}{R}")


def backend_up():
    try:
        return get(BACKEND + "/health", timeout=5)
    except Exception:
        return None


def ensure_backend():
    h = backend_up()
    if h:
        return h
    print(f"{Y}Local backend is down.{R}")
    if input("Start it? [Y/n] ").strip().lower() not in ("", "y"):
        return None
    subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--port", "8000"],
                     cwd=ROOT, stdout=open("/tmp/hookline_uv.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(20):
        time.sleep(1)
        h = backend_up()
        if h:
            return h
    print(f"{RD}Backend failed to start, see /tmp/hookline_uv.log{R}")
    return None


def pick_video(corpus):
    print(f"\n{B}Pick a reel{R}  [1] face  [2] faceless  [3] outliers  [4] any  [b] back")
    c = input("> ").strip()
    pool = {"1": "face", "2": "faceless", "3": "outliers"}.get(c)
    if c == "b":
        return None
    items = corpus[pool] if pool else [v for s in STRATA for v in corpus[s]]
    v = random.choice(items)
    print(f"\n🎲 {B}{v['video_id']}{R} ({v.get('page_type', pool or 'mixed')}) · "
          f"{v['views']} views {D}(validation context only){R}\n   {v['_path']}")
    return v


def analyze_flow(path, mode):
    h = ensure_backend()
    if not h:
        return
    print(f"{D}Uploading + analyzing ({mode})…{R}")
    try:
        res = post_file(BACKEND + "/analyze", path, extra=f"?mode={mode}")
    except Exception as e:
        print(f"{RD}Analyze failed: {e}{R}")
        return
    if "report" not in res:
        print(f"{RD}{res.get('error', res)}{R}")
        return
    job = res["job_id"]
    show_report(res["report"], job)
    print(f"\n{D}view: {BACKEND}/report/{job}/view · png: {BACKEND}/report/{job}.png{R}")


def review_flow():
    jobs = sorted((p for p in OUT.glob("*/report.json") if p.stat().st_size > 100),
                  key=lambda p: p.stat().st_mtime, reverse=True)[:15]
    if not jobs:
        print(f"{Y}No saved reports yet (backend/out/).{R}")
        return
    print(f"\n{B}Saved reports{R}")
    for i, p in enumerate(jobs):
        try:
            r = json.loads(p.read_text())
            s = r["scores"]
            print(f"  [{i}] {p.parent.name}  hook {s['hook_pct']} · virality {s['virality']} "
                  f"{D}{time.strftime('%m-%d %H:%M', time.localtime(p.stat().st_mtime))}{R}")
        except Exception:
            print(f"  [{i}] {p.parent.name}  (unreadable)")
    c = input("Open # or [b]ack > ").strip()
    if c == "b" or not c.isdigit() or int(c) >= len(jobs):
        return
    p = jobs[int(c)]
    show_report(json.loads(p.read_text()), p.parent.name)
    print(f"\n{D}view: {BACKEND}/report/{p.parent.name}/view{R}")


SPACE_BASE = os.environ.get("HOOKLINE_SPACE_URL", "https://wlddhq-hookline-tribe.hf.space")


def space_token():
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        print(f"{Y}HF_TOKEN not set — run: export HF_TOKEN=$(hf auth token){R}")
    return tok


def space_health():
    tok = space_token()
    if not tok:
        return
    try:
        print(json.dumps(get(SPACE_BASE + "/health", token=tok, timeout=30), indent=1))
    except Exception as e:
        print(f"{RD}Space unreachable: {e}{R}")


def own_video_flow():
    """Upload your own video file straight to the Space. Nothing stored locally."""
    raw = input("Video path (drag-drop ok) > ").strip().strip("'\"")
    if not raw:
        return None
    p = Path(os.path.expanduser(raw))
    if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".m4v", ".webm"):
        print(f"{RD}Not a video file: {raw}{R}")
        return None
    return {"video_id": p.stem, "_path": str(p), "_own": True,
            "views": "n/a", "page_type": "upload"}


def space_direct_flow(path):
    """POST video to the Space, poll, show report. Local disk untouched."""
    tok = space_token()
    if not tok:
        return
    try:
        size = Path(path).stat().st_size / 1e6
        print(f"{D}Uploading {size:.1f}MB to Space (not stored locally)…{R}")
        res = post_file(SPACE_BASE + "/analyze", path, token=tok)
        job = res["job_id"]
    except Exception as e:
        print(f"{RD}Space upload failed: {e}{R}")
        return
    print(f"{D}Job {job} queued — first run warms the model, a reel takes minutes. Polling…{R}")
    try:
        for _ in range(60):
            time.sleep(30)
            r = get(f"{SPACE_BASE}/report/{job}", token=tok, timeout=60)
            if "scores" in r:
                show_report(r, f"space:{job}")
                print(f"\n{D}assets (token-gated): {SPACE_BASE}/report/{job}.png "
                      f"{SPACE_BASE}/report/{job}.md{R}")
                return
            if r.get("status") == "failed":
                print(f"{RD}Space job failed: {r.get('error')}{R}")
                return
            print(f"{D}…{r.get('status', '?')}{R}", end=" ", flush=True)
        print(f"\n{Y}Still running after ~30min — check later: job {job}{R}")
    except Exception as e:
        print(f"\n{RD}Poll error: {e} (job {job} may still be running){R}")


def main():
    corpus = load_corpus()
    total = sum(len(v) for v in corpus.values())
    print(f"{B}{C}HOOKLINE{R} {D}video in → attention preview · {total} sample reels "
          f"({', '.join(f'{s}:{len(corpus[s])}' for s in STRATA)}){R}")
    current = None
    while True:
        h = backend_up()
        print(f"\n{D}backend: {G + 'up' if h else RD + 'down'}{R} {D}| "
              f"space: {'token set' if os.environ.get('HF_TOKEN') else 'no token'}{R}")
        print("[1] 🎲 random reel   [2] your own video file   [3] analyze preview (local)   "
              "[4] analyze via Space   [5] review saved   [6] health   [q] quit")
        c = input("> ").strip().lower()
        if c == "q":
            break
        elif c == "1":
            current = pick_video(corpus) or current
        elif c == "2":
            v = own_video_flow()
            if v:
                current = v
                space_direct_flow(v["_path"])
        elif c == "3":
            if not current:
                current = pick_video(corpus)
            if current:
                analyze_flow(current["_path"], "preview")
        elif c == "4":
            if not current:
                current = pick_video(corpus)
            if current:
                if current.get("_own"):
                    space_direct_flow(current["_path"])
                else:
                    analyze_flow(current["_path"], "hf")
        elif c == "5":
            review_flow()
        elif c == "6":
            print(f"{D}--- backend ---{R}");
            print(json.dumps(h or {"ok": False}, indent=1)[:400])
            print(f"{D}--- space ---{R}");
            space_health()
        if current:
            print(f"{D}current: {current['video_id']}{R}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
