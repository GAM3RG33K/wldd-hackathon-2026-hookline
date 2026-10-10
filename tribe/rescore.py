#!/usr/bin/env python
"""Build Hookline reports locally from raw TRIBE output.

The GPU Space only produces raw output (preds, times, transcript words, gate/audio metadata).
Reports are generated here, against the local video, so formula changes apply instantly and
every past reel can be re-scored without a GPU.

  python tribe/rescore.py                  # regenerate every report that has raw/ output
  python tribe/rescore.py s-1a2b3c4d       # one job
  python tribe/rescore.py --fetch <space_job_id> --video samples/face/videos/FC001.mp4
"""
import argparse
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "backend" / "out"
SPACE = os.environ.get("HOOKLINE_SPACE_URL", "https://wlddhq-hookline-tribe.hf.space")
sys.path.insert(0, str(HERE))
import audio_music as AMU  # noqa: E402
import emotion_fusion as EF  # noqa: E402
import hookline_report as HL  # noqa: E402
import suggest as SG  # noqa: E402
import taxonomy as TX  # noqa: E402


def hf_token():
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        return tok
    out = subprocess.run(["hf", "auth", "token"], capture_output=True, text=True).stdout.strip()
    return out.splitlines()[-1] if out else ""


def fetch_raw(space_job, token=None, out=OUT, dest=None):
    """Download the Space job's raw bundle into backend/out/s-<job>/raw/ (or `dest`); keep it forever."""
    req = urllib.request.Request(f"{SPACE}/jobs/{space_job}/raw",
                                 headers={"Authorization": f"Bearer {token or hf_token()}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    raw = Path(dest) if dest else out / f"s-{space_job}" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(raw)
    return raw.parent


def rescore(job_dir, video=None, source_id=None):
    """(Re)build report.json/md/png in job_dir from job_dir/raw + the local video."""
    import numpy as np
    for mod in (TX, AMU, EF, SG, HL):  # pick up formula edits without restarting the backend (dependencies first)
        importlib_reload(mod)
    job_dir = Path(job_dir)
    raw = job_dir / "raw"
    meta = json.loads((raw / "meta.json").read_text())
    old = json.loads((job_dir / "report.json").read_text()) if (job_dir / "report.json").exists() else {}
    video = Path(video or old.get("local_video") or "")
    if not video.is_file():
        raise FileNotFoundError(f"local video not found for {job_dir.name}: {video}")
    # analyse() reads meta.json from the folder it is given: point it at the local copies
    words = raw / "words.tsv"
    meta.update(video=str(video), transcript=str(words) if words.exists() else None)
    (raw / "meta.json").write_text(json.dumps(meta, default=str))
    preds = np.load(raw / "preds.npy")
    if preds.dtype != np.float32:  # stored as float16 to keep bundles small
        np.save(raw / "preds.npy", preds.astype(np.float32))
    report, d = HL.analyse(raw, video)
    HL.finalize(report, meta.get("space") or {})
    report.update(preview=False, local_video=str(video.resolve()),
                  source_id=source_id or old.get("source_id") or video.stem,
                  space_job=meta.get("space_job"), regenerated_locally=True)
    # keep model-check results already paid for in GPU time when the same edit is suggested again
    done = {(x["edit"], x.get("start"), x.get("end")): x.get("verification")
            for x in old.get("suggestions") or [] if x.get("verification")}
    for x in report.get("suggestions") or []:
        x["verification"] = done.get((x["edit"], x.get("start"), x.get("end")))
    if old.get("verify_status"):
        report["verify_status"] = old["verify_status"]
    (job_dir / "report.json").write_text(json.dumps(report, indent=1))
    HL.write_md(report, job_dir / "report.md")
    HL.write_png(report, d, job_dir / "report.png")
    return report


def _video_key(report):
    """Identity of the reel = its file name (without extension), so a web-upload preview and a
    full run of the same file match. Two different files with the same name would replace each other."""
    v = report.get("local_video")
    if not v:
        return None
    name = report.get("source_id") or Path(v).stem
    return None if name in ("input", "") else name.lower()


def _is_full(report):
    return report.get("preview") is False or "full" in str(report.get("neural_source", ""))


def supersede(job_dir, out=OUT, log=print):
    """One report per video: move older reports of the same video to out/.replaced/.
    A full TRIBE report replaces older full reports and previews; a preview only replaces
    older previews (it never deletes a full report). Returns the removed folder names."""
    import shutil
    job_dir = Path(job_dir)
    new = json.loads((job_dir / "report.json").read_text())
    key, full = _video_key(new), _is_full(new)
    if not key:
        return []
    removed = []
    for rp in out.glob("*/report.json"):
        d = rp.parent
        if d.resolve() == job_dir.resolve() or d.name == "demo":
            continue
        try:
            old = json.loads(rp.read_text())
        except Exception:  # noqa: BLE001 - unreadable report: leave it alone
            continue
        if _video_key(old) != key or (_is_full(old) and not full):
            continue
        archive = out / ".replaced"  # out of every listing; recoverable until purged
        archive.mkdir(exist_ok=True)
        if (archive / d.name).exists():
            shutil.rmtree(archive / d.name)
        shutil.move(str(d), str(archive / d.name))
        removed.append(d.name)
    if removed:
        log(f"replaced older report(s) of {Path(key).name}: {', '.join(removed)}")
    return removed


def importlib_reload(mod):
    import importlib
    try:
        importlib.reload(mod)
    except Exception:  # noqa: BLE001 - keep the loaded version if the edit is mid-save
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*", help="job folders under backend/out (default: all with raw/)")
    ap.add_argument("--fetch", help="Space job id to download raw output for, then score")
    ap.add_argument("--video", help="local video for --fetch")
    a = ap.parse_args()
    if a.fetch:
        job_dir = fetch_raw(a.fetch)
        r = rescore(job_dir, a.video)
        print(f"{job_dir.name}: viral potential {r['scores']['virality']} {r['scores'].get('virality_tag')}")
        return
    dirs = [OUT / j for j in a.jobs] if a.jobs else sorted(p.parent.parent for p in OUT.glob("*/raw/meta.json"))
    for d in dirs:
        try:
            r = rescore(d)
            s = r["scores"]
            print(f"{d.name:12s} {r.get('source_id', ''):7s} viral potential {s['virality']:5.1f} "
                  f"{s.get('virality_tag', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"{d.name:12s} FAILED: {e}")


if __name__ == "__main__":
    main()
