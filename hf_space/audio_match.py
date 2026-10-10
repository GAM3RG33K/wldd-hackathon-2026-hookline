"""Song identification for reels: Chromaprint fingerprints matched against
(1) a hand-curated trending-audio catalog (iTunes 30s previews), and
(2) every reel analysed earlier on this Space (audio reuse across reels).

No reel audio leaves the Space: only public previews are downloaded.
Report context only; nothing here feeds the scores.
"""
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CATALOG_FILE = HERE / "trending_audio.json"
CACHE = Path("/tmp/audio_catalog")
INDEX_FILE = Path("/tmp/audio_index.json")
ITEM_S = 0.1238  # seconds per Chromaprint raw fingerprint item
MATCH_SIM = 0.65  # 1 - bit error rate; unrelated audio sits near 0.5
MIN_OVERLAP = 40  # items (~5s) of overlap required for a match
_LOCK = threading.Lock()
STATE = {"catalog": [], "status": "not built"}


def fingerprint(path, length_s=180):
    """Raw Chromaprint fingerprint (list of uint32) via fpcalc; works on video files too."""
    r = subprocess.run(["fpcalc", "-raw", "-json", "-length", str(length_s), str(path)],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"fpcalc failed: {r.stderr.strip()[:200]}")
    return [int(x) & 0xFFFFFFFF for x in json.loads(r.stdout)["fingerprint"]]


def similarity(a, b, min_overlap=MIN_OVERLAP):
    """Best 1 - bit-error-rate over all alignments; returns (sim, offset_s of b inside a)."""
    a = np.asarray(a, dtype=np.uint32)
    b = np.asarray(b, dtype=np.uint32)
    best, best_off = 0.0, 0
    for off in range(-(len(b) - min_overlap), len(a) - min_overlap + 1):
        a0, b0 = max(0, off), max(0, -off)
        n = min(len(a) - a0, len(b) - b0)
        if n < min_overlap:
            continue
        x = np.bitwise_xor(a[a0:a0 + n], b[b0:b0 + n])
        sim = 1.0 - np.unpackbits(x.view(np.uint8)).sum() / (32.0 * n)
        if sim > best:
            best, best_off = sim, off
    return float(best), round(best_off * ITEM_S, 1)


def build_catalog(log=print):
    """Download previews and fingerprint them once per container (cached in /tmp)."""
    with _LOCK:
        if STATE["catalog"]:
            return STATE["catalog"]
        STATE["status"] = "building"
        CACHE.mkdir(parents=True, exist_ok=True)
        meta = json.loads(CATALOG_FILE.read_text())
        out, t0 = [], time.time()
        for t in meta.get("tracks", []):
            fp_file = CACHE / f"{t['itunes_id']}.json"
            try:
                if not fp_file.exists():
                    audio = CACHE / f"{t['itunes_id']}.m4a"
                    urllib.request.urlretrieve(t["preview_url"], audio)
                    fp_file.write_text(json.dumps(fingerprint(audio)))
                out.append({**t, "fp": json.loads(fp_file.read_text())})
            except Exception as e:  # noqa: BLE001 - one bad preview must not kill the catalog
                log(f"[audio] skip {t.get('title')}: {e}")
        STATE["catalog"] = out
        STATE["status"] = f"{len(out)} tracks ({time.time() - t0:.0f}s), updated {meta.get('updated')}"
        return out


def _load_index():
    try:
        return json.loads(INDEX_FILE.read_text())
    except Exception:  # noqa: BLE001
        return []


def identify(video, job_id, source, audio_type, log=print):
    """Fingerprint the reel, match against the catalog and earlier reels, remember it."""
    fp = fingerprint(video)
    result = {"song": None, "reuse": [], "fingerprint_items": len(fp)}
    best = None
    for t in build_catalog(log):
        sim, off = similarity(fp, t["fp"])
        if sim >= MATCH_SIM and (best is None or sim > best["similarity"]):
            best = {"title": t["title"], "artist": t["artist"], "list": t["list"], "region": t["region"],
                    "trending": True, "similarity": round(sim, 3), "offset_s": off}
    result["song"] = best
    with _LOCK:
        index = _load_index()
        for e in index:
            if e["job"] == job_id:
                continue
            sim, off = similarity(fp, e["fp"])
            if sim >= MATCH_SIM:
                result["reuse"].append({"job": e["job"], "source": e["source"], "similarity": round(sim, 3)})
        index.append({"job": job_id, "source": source, "audio_type": audio_type, "fp": fp})
        INDEX_FILE.write_text(json.dumps(index[-500:]))
    return result
