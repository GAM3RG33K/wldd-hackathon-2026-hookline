"""Re-run every reel that has a saved report on the Space (full TRIBE + all raw channels), then import
through the local backend so the new report replaces the old one. Sequential: one GPU job at a time.
Usage: python tribe/rerun_all.py [--only=name1,name2] [--skip=name1,...]"""
import glob
import json
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tui  # noqa: E402  (http helpers, token)

SPACE, BACKEND = tui.SPACE, tui.BACKEND
TOKEN = tui.hf_token()


def reels():
    seen = {}
    for f in glob.glob(str(ROOT / "backend/out/*/report.json")):
        if "/demo/" in f:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        v = r.get("local_video")
        name = r.get("source_id") or (Path(v).stem if v else None)
        if not v or not Path(v).is_file() or name in (None, "input"):
            continue
        cap = (r.get("caption") or {}).get("text") or ""  # keep the caption it was analysed with
        seen.setdefault(name.lower(), (name, v, r.get("duration_s") or 0, cap))
    return sorted(seen.values(), key=lambda x: x[2])  # shortest first: results land early


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(name, video, caption=""):
    q = "video=1&audio=1&text=auto&lang=auto"
    for _ in range(3):
        try:
            res = tui.post_video(f"{SPACE}/analyze?{q}", video, token=TOKEN, timeout=900,
                                 fields={"caption": caption} if caption else None)
            break
        except Exception as e:  # noqa: BLE001
            log(f"  upload failed ({e}); retry in 30s")
            time.sleep(30)
    else:
        return "upload failed"
    sid = res["job_id"]
    t0 = time.time()
    while True:
        time.sleep(5)
        try:
            p = tui.http_json(f"{SPACE}/jobs/{sid}", TOKEN, timeout=20)
        except Exception:
            if time.time() - t0 > 3600:
                return f"lost job {sid}"
            continue
        if p.get("status") in ("done", "failed", "cancelled"):
            break
    if p["status"] != "done":
        return f"{p['status']} in {p.get('stage')}: {str(p.get('error', ''))[:120]}"
    qs = urllib.parse.urlencode({"video": str(Path(video).resolve()), "source_id": name})
    r = tui.http_json(f"{BACKEND}/import/{sid}?{qs}", timeout=600, data=b"")
    if "report" not in r:
        return f"import failed: {r}"
    s = r["report"]["scores"]
    st = (r["report"].get("emotion") or {}).get("strong") or {}
    strong = {k: v for k, v in (st.get("durations_s") or {}).items() if v}
    faces = sum(1 for x in ((((json.load(open(ROOT / "backend/out" / r["job_id"] / "raw/meta.json"))
                                .get("space") or {}).get("emotion") or {}).get("face") or {}).get("probs") or []) if x)
    return (f"ok {r['job_id']} in {time.time() - t0:.0f}s · viral {s['virality']:.0f} {s.get('virality_tag')} · "
            f"strong {strong} · face frames {faces} · replaced {r['report'].get('replaced')}")


def main():
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    skip = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--skip=")), [])
    todo = [x for x in reels() if (not only or x[0] in only) and x[0] not in skip]
    log(f"batch start: {len(todo)} reels, {sum(x[2] for x in todo):.0f}s of video")
    for i, (name, video, dur, caption) in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {name} ({dur:.0f}s){' + caption' if caption else ''}")
        try:
            log("  " + run(name, video, caption))
        except Exception as e:  # noqa: BLE001 - keep the batch going
            log(f"  error {type(e).__name__}: {e}")
    log("batch complete")


if __name__ == "__main__":
    main()
