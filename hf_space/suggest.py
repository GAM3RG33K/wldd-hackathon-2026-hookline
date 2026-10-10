"""Retention editing suggestions: diagnose each dip / drop-off from evidence, prescribe a concrete
edit (with render parameters so it can be model-checked), rank by estimated viewers lost.

Evidence compares each problem zone with the reel's OWN best stretch (its strongest window), so
advice is specific to this video: "no cut for 4.0s (your best part cuts every 1.2s)" instead of
"add more cuts". Everything here runs on the report alone (re-scorable, no GPU).
"""
import numpy as np

MAX_SUGGESTIONS = 5


def fmt(t):
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


def _arr(series, key, n, default=0.0):
    x = series.get(key)
    if not x:
        return np.full(n, default, dtype=float)
    x = [default if v is None else v for v in x]
    return np.array((x + [default] * n)[:n], dtype=float)


def _stats(series, a, b, n, cuts, emotion):
    seg = slice(a, max(a + 1, b))
    dur = max(1, b - a)
    nz = series.get("network_z") or {}
    out = {
        "dur": dur,
        "cuts": sum(1 for c in cuts if a <= c < b),
        "motion": float(_arr(series, "motion", n)[seg].mean()),
        "loud": float(_arr(series, "loudness_db", n, -45.0)[seg].mean()),
        "words": float(_arr(series, "words_per_s", n)[seg].mean()) if series.get("words_per_s") else None,
        "att": float(_arr(series, "attention", n, 50.0)[seg].mean()),
        "net": {g: float(np.mean((list(v) + [0.0] * n)[:n][a:max(a + 1, b)])) for g, v in nz.items()},
    }
    if emotion:
        ar = emotion["series"]["arousal"]
        out["arousal"] = float(np.mean(ar[a:max(a + 1, b)]))
        doms = [d for d in emotion["series"]["dominant"][a:max(a + 1, b)] if d != "neutral"]
        out["emotion"] = max(set(doms), key=doms.count) if doms else "neutral"
    out["cut_every"] = dur / out["cuts"] if out["cuts"] else None
    return out


def _evidence(z, ref):
    """Human sentence comparing a zone to the reel's best stretch."""
    bits = []
    if z["cuts"] == 0:
        ref_txt = f" (your best part cuts every {ref['cut_every']:.1f}s)" if ref.get("cut_every") else ""
        bits.append(f"no cut for {z['dur']:.0f}s{ref_txt}")
    if ref["motion"] > 1e-6 and z["motion"] < 0.7 * ref["motion"]:
        bits.append(f"movement {100 * z['motion'] / ref['motion']:.0f}% of your best part")
    if z["words"] is not None and z["words"] < 0.3:
        bits.append("no speech")
    if z["loud"] < ref["loud"] - 4:
        bits.append(f"sound {ref['loud'] - z['loud']:.0f} dB quieter than your best part")
    drops = sorted(((z["net"].get(g, 0) - ref["net"].get(g, 0), g) for g in ("visual", "auditory", "language")),
                   key=lambda x: x[0])
    if drops and drops[0][0] < -0.8:
        bits.append(f"{drops[0][1]} network {drops[0][0]:+.1f}σ vs your best part")
    if "arousal" in z and "arousal" in ref and z["arousal"] < ref["arousal"] - 0.1:
        bits.append(f"emotional arousal down ({z['arousal']:.2f} vs {ref['arousal']:.2f}, {z.get('emotion', '?')})")
    return ", ".join(bits) or "attention falls below the rest of the video"


def _caption_for(report, a, b, max_words=9):
    """The words actually spoken during the dip (word timeline), else the overlapping transcript line."""
    words = [w["w"] for w in report.get("words_timeline") or [] if a - 0.5 <= w["t"] < b]
    if words:
        line = " ".join(words[:max_words])
        return line + (" ..." if len(words) > max_words else "")
    best, best_ov = None, 0.0
    for seg in (report.get("transcript") or {}).get("segments") or []:
        ov = min(seg["end"], b) - max(seg["start"], a)
        if ov > best_ov:
            best, best_ov = seg, ov
    return best["text"].strip()[:60] if best else None


def suggest(report):
    s = report.get("series") or {}
    att = s.get("attention") or []
    n = len(att)
    if n < 4:
        return []
    surv = _arr(s, "survival_pct", n, 100.0)
    cuts = [c for c in (s.get("cuts_s") or [])]
    m = report.get("moments") or {}
    emotion = report.get("emotion")
    sw = m.get("strongest_window_s") or [int(np.argmax(att)), int(np.argmax(att)) + 3]
    ref = _stats(s, sw[0], min(n, sw[1]), n, cuts, emotion)
    out = []

    # 1) dips and drop-offs, ranked by viewers lost in the zone (survival curve, model estimate)
    for a, b in m.get("drop_zones_s") or []:
        a, b = int(a), int(min(b, n))
        if b - a < 1:
            continue
        lost = float(surv[max(0, a - 1)] - surv[b - 1])
        z = _stats(s, a, b, n, cuts, emotion)
        why = _evidence(z, ref)
        speech = z["words"] is not None and z["words"] >= 0.3
        lang_ok = z["net"].get("language", 0) > z["net"].get("visual", 0)
        if not speech and z["cuts"] == 0 and z["dur"] >= 2 and (z.get("arousal", 0) < 0.4):
            edit, text, params = "trim", f"Cut {fmt(a)}-{fmt(b)} ({b - a}s of dead air).", {"start": a, "end": b}
        elif speech and lang_ok and z["cuts"] == 0:
            cap = _caption_for(report, a, b)
            edit = "text"
            text = (f"Put the line on screen at {fmt(a)}: \"{cap}\"." if cap else
                    f"Add an on-screen text hook at {fmt(a)} (the words carry this part, the picture does not).")
            params = {"start": a, "end": b, "text": cap or "Wait for it..."}
        elif z["cuts"] == 0 or (ref["motion"] > 1e-6 and z["motion"] < 0.7 * ref["motion"]):
            edit, text, params = ("zoom", f"Pattern interrupt at {fmt(a)}: punch-in zoom or b-roll for {fmt(a)}-{fmt(b)}.",
                                  {"start": a, "end": b, "zoom": 1.15})
        elif z["loud"] < ref["loud"] - 4:
            edit, text, params = ("sound", f"Lift the audio at {fmt(a)}: sound hit or music swell into {fmt(b)}.",
                                  {"start": a, "end": b, "gain_db": 6})
        else:
            edit, text, params = ("speed", f"Speed {fmt(a)}-{fmt(b)} up to 1.25x.", {"start": a, "end": b, "rate": 1.25})
        out.append({"at": f"{fmt(a)}-{fmt(b)}", "start": a, "end": b, "issue": "attention dip / drop-off",
                    "edit": edit, "suggestion": text, "why": why,
                    "impact": {"viewers_lost_pct": round(lost, 1)}, "params": params, "verification": None})

    # 2) hook: strongest moment buried late -> cold open teaser
    hook = report.get("scores", {}).get("hook_pct", 100)
    peak = m.get("peak_attention_at_s", 0)
    if hook < 60 and peak > 5:
        a = max(0, min(int(sw[0]), n - 2))
        b = min(n, a + 2)
        out.append({"at": "0:00-0:03", "start": 0, "end": 3, "issue": "weak hook, best moment comes late",
                    "edit": "cold_open", "suggestion": f"Open with {fmt(a)}-{fmt(b)} as a teaser, then start the video.",
                    "why": f"hook {hook:.0f}; your strongest stretch is {fmt(sw[0])}-{fmt(sw[1])}",
                    "impact": {"viewers_lost_pct": round(float(100 - surv[min(2, n - 1)]), 1)},
                    "params": {"start": a, "end": b}, "verification": None})

    # 3) ending: weak last seconds -> end on the last peak
    import hookline_report as HL
    peaks = HL.attention_peaks(att)
    if n > 6 and float(np.mean(att[-3:])) < 40 and peaks and peaks[-1] < n - 3:
        end = min(n, peaks[-1] + 2)
        out.append({"at": f"{fmt(end)}-{fmt(n)}", "start": end, "end": n, "issue": "weak ending",
                    "edit": "end", "suggestion": f"End at {fmt(end)} on the last high point; cut the last {n - end}s.",
                    "why": f"attention in the last 3s averages {np.mean(att[-3:]):.0f}",
                    "impact": {"viewers_lost_pct": round(float(surv[end - 1] - surv[-1]), 1)},
                    "params": {"end": end}, "verification": None})

    # drop dips already covered by a bigger structural edit (cold open fixes 0-3s, end trim removes the tail)
    cold = any(x["edit"] == "cold_open" for x in out)
    end_cut = min((x["start"] for x in out if x["edit"] == "end"), default=None)
    out = [x for x in out if x["edit"] in ("cold_open", "end")
           or not ((cold and x["start"] < 3) or (end_cut is not None and x["start"] >= end_cut))]
    out.sort(key=lambda x: -x["impact"]["viewers_lost_pct"])
    for i, x in enumerate(out[:MAX_SUGGESTIONS], 1):
        x["id"] = i
    return out[:MAX_SUGGESTIONS]
