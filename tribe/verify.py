"""Model-check editing suggestions: render each edit, run full TRIBE on it, compare with the original.

Comparison uses ABSOLUTE TRIBE engagement (series.engagement_raw), which is comparable across
videos. The within-video scores (attention ~50, hook/hold) are not: they renormalise per video.
Costs one full GPU run per edit (~2-3 min for a 60s reel), so it runs on demand.
"""
import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np

import edit_render as ER
import rescore as RS

TOP = 3
IMPROVED = 0.02  # +2% relative engagement counts as an improvement


def _post_video(path, token, q):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"video\"; filename=\"{Path(path).name}\"\r\n"
            f"Content-Type: video/mp4\r\n\r\n").encode() + Path(path).read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"{RS.SPACE}/analyze?{q}", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def _job(sid, token):
    req = urllib.request.Request(f"{RS.SPACE}/jobs/{sid}", headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _windows(x, w=3):
    x = np.asarray(x, dtype=float)
    return np.array([x[i:i + w].mean() for i in range(max(1, len(x) - w + 1))])


IN_PLACE = {"zoom", "text", "sound"}  # same timeline before and after: compare second by second


def compare(orig, var, edit, params=None):
    eo = np.asarray(orig["series"].get("engagement_raw") or [], dtype=float)
    ev = np.asarray(var["series"].get("engagement_raw") or [], dtype=float)
    if not len(eo) or not len(ev):
        return {"verdict": "unknown", "reason": "no raw engagement series"}
    scale = float(np.abs(eo).mean()) or 1e-9
    d_mean = (ev.mean() - eo.mean()) / scale
    d_worst = (_windows(ev).min() - _windows(eo).min()) / scale
    d_open = (ev[:3].mean() - eo[:3].mean()) / scale
    res = {"engagement_change_pct": round(100 * d_mean, 1), "worst_3s_change_pct": round(100 * d_worst, 1),
           "first_3s_change_pct": round(100 * d_open, 1),
           "viral_potential": {"before": orig["scores"]["virality"], "after": var["scores"]["virality"]},
           "duration_s": {"before": orig.get("duration_s"), "after": var.get("duration_s")}}
    if edit in IN_PLACE and params and abs(len(ev) - len(eo)) <= 1:
        a, b = int(params["start"]), int(params["end"])
        n = min(len(eo), len(ev))
        zone = slice(a, max(a + 1, min(b, n)))
        rest = np.r_[0:a, min(b, n):n]
        d_zone = (ev[zone].mean() - eo[zone].mean()) / scale
        d_rest = (ev[rest].mean() - eo[rest].mean()) / scale if len(rest) else 0.0
        res.update(edited_zone_change_pct=round(100 * d_zone, 1), rest_change_pct=round(100 * d_rest, 1),
                   judged_on="edited stretch (rest must not drop)")
        key, guard = d_zone, d_rest > -2 * IMPROVED
    elif edit == "cold_open":
        key, guard = d_open, True
        res["judged_on"] = "first 3 seconds"
    else:
        key, guard = d_mean, True
        res["judged_on"] = "whole video"
    res["verdict"] = ("improved" if key > IMPROVED and guard else
                      "worse" if key < -IMPROVED else "no clear change")
    return res


def recompare(job_dir):
    """Re-judge stored edit checks with the current rules (no GPU)."""
    job_dir = Path(job_dir)
    rp = job_dir / "report.json"
    report = json.loads(rp.read_text())
    for s in report.get("suggestions") or []:
        vdir = job_dir / "variants" / f"{s['id']}-{s['edit']}"
        if s.get("verification") and (vdir / "report.json").exists():
            keep = {k: s["verification"].get(k) for k in ("space_job", "edited_video", "approximate")}
            bfile = job_dir / "variants" / "0-baseline" / "report.json"
            ref = json.loads(bfile.read_text()) if bfile.exists() else report
            s["verification"] = {**compare(ref, json.loads((vdir / "report.json").read_text()),
                                           s["edit"], s.get("params")), **keep,
                                 "compared_with": "null edit (re-encoded original)" if bfile.exists() else "original"}
    rp.write_text(json.dumps(report, indent=1))
    return report


def _run_on_space(clip, token, q):
    sid = _post_video(clip, token, q)["job_id"]
    t0 = time.time()
    while True:
        time.sleep(5)
        st = _job(sid, token)
        if st.get("status") in ("done", "failed", "cancelled"):
            return sid, st
        if time.time() - t0 > 30 * 60:
            return sid, {"status": "timeout"}


def baseline(job_dir, report, video, token, q, save):
    """Null edit: the same video re-encoded like every variant. Variants are compared with this,
    not with the original, so encoding differences don't count as edit effects. Run once per reel."""
    bdir = job_dir / "variants" / "0-baseline"
    if (bdir / "report.json").exists():
        return json.loads((bdir / "report.json").read_text())
    save("running 0: null edit (re-encode only) for a fair baseline")
    clip = ER.render(video, "baseline", {}, bdir / "edited.mp4")
    sid, st = _run_on_space(clip, token, q)
    if st.get("status") != "done":
        return None
    RS.fetch_raw(sid, token, dest=bdir / "raw")
    return RS.rescore(bdir, clip, source_id=f"{report.get('source_id')}-baseline")


def verify(job_dir, top=TOP, log=print):
    job_dir = Path(job_dir)
    rp = job_dir / "report.json"
    report = json.loads(rp.read_text())
    video = report.get("local_video")
    token = RS.hf_token()
    todo = [s for s in report.get("suggestions") or [] if s["edit"] in ER.RENDERABLE][:top]
    mods = report.get("modalities") or {}
    q = f"video={int(mods.get('video', True))}&audio={int(mods.get('audio', True))}&text=auto"

    def save(status):
        report["verify_status"] = status
        rp.write_text(json.dumps(report, indent=1))

    base = baseline(job_dir, report, video, token, q, save)
    if base is None:
        log(f"verify {job_dir.name}: baseline run failed; comparing with the original instead")
    ref = base or report
    for i, s in enumerate(todo, 1):
        save(f"running {i}/{len(todo)}: rendering {s['edit']} {s['at']}")
        vdir = job_dir / "variants" / f"{s['id']}-{s['edit']}"
        clip = ER.render(video, s["edit"], s["params"], vdir / "edited.mp4")
        save(f"running {i}/{len(todo)}: full TRIBE on edited video ({s['edit']} {s['at']})")
        sid, st = _run_on_space(clip, token, q)
        if st.get("status") != "done":
            s["verification"] = {"verdict": "not checked", "reason": f"GPU job {st.get('status')}"}
            save(f"running {i}/{len(todo)}: {s['edit']} failed")
            continue
        RS.fetch_raw(sid, token, dest=vdir / "raw")
        var = RS.rescore(vdir, clip, source_id=f"{report.get('source_id')}-edit{s['id']}")
        res = compare(ref, var, s["edit"], s.get("params"))
        res.update(space_job=sid, edited_video=str(clip), approximate=ER.APPROXIMATE.get(s["edit"]),
                   compared_with="null edit (re-encoded original)" if base else "original")
        s["verification"] = res
        log(f"verify {job_dir.name} #{s['id']} {s['edit']}: {res['verdict']} "
            f"(engagement {res.get('engagement_change_pct')}%)")
        save(f"running {i}/{len(todo)}: done {s['edit']}")
    save(f"done ({len(todo)} edits checked, vs {'null edit' if base else 'original'})")
    return report
