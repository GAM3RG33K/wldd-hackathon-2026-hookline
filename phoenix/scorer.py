#!/usr/bin/env python3
"""Phoenix v2 scorer — builds a creator report from a TTCC retention curve R(1..60).

TTCC's retention head outputs R(t) = fraction of viewers still watching at second t.
This module turns that curve into:
- hook / hold / finish / virality scores (within-video, uncalibrated heuristics)
- hooks: consecutive retention climbs (what keeps people watching)
- drop-off zones: consecutive retention drops (where viewers leave)
- timestamped editing suggestions against the reel's own best stretch

Method note that must ship with every report: scores are heuristic,
within-video normalised and uncalibrated until fitted on real view counts.
They judge the video only — never the account.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

T_MAX = 60  # retention head emits R(1..60)

VIRALITY_WEIGHTS = {
    "hook": 0.30,
    "drop_free": 0.25,
    "strong_moments": 0.30,
    "ending": 0.15,
}
METHOD = "heuristic, within-video normalised, uncalibrated"


def band(score: float) -> str:
    return ("Bad" if score < 30 else "Weak" if score < 50 else "Average"
            if score < 70 else "Strong" if score < 85 else "Viral-grade")


def virality_tag(score: float) -> str:
    return ("Low reach" if score < 35 else "Average" if score < 55 else
            "Above average" if score < 70 else "Popular" if score < 85 else "Viral potential")


def fmt(t: float) -> str:
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


def segments(curve: list[float], min_len: int = 2, threshold: float = 0.01):
    """Consecutive climb (hook) and drop (drop-off) segments, in seconds.

    Returns (climbs, drops) as lists of [start_s, end_s) over R(1..T).
    """
    diffs = np.diff(np.concatenate([[1.0], np.asarray(curve, dtype=float)]))
    climbs, drops = [], []
    i = 0
    n = len(diffs)
    while i < n:
        if diffs[i] > threshold:
            start = i
            while i < n and diffs[i] > threshold:
                i += 1
            if i - start >= min_len:
                climbs.append([start, i])
        elif diffs[i] < -threshold:
            start = i
            while i < n and diffs[i] < -threshold:
                i += 1
            if i - start >= min_len:
                drops.append([start, i])
        else:
            i += 1
    return climbs, drops


def attention_from_retention(curve: list[float]) -> list[float]:
    """Within-video normalised attention: mean ~50 by design, std band 15."""
    r = np.asarray(curve, dtype=float)
    if r.std() < 1e-9:
        return [50.0] * len(r)
    a = 50.0 + 15.0 * (r - r.mean()) / r.std()
    return np.clip(a, 0, 100).tolist()


def analyse(curve: list[float], duration_s: float, video: str = "",
            caption: str = "", demo: bool = False) -> dict:
    """Build the full report dict from a retention curve R(1..T)."""
    T = max(int(math.ceil(duration_s)), 1)
    curve = [float(x) for x in curve[:T]]
    if not curve:
        curve = [1.0]
    while len(curve) < T:  # hold last value if the head covered fewer seconds
        curve.append(curve[-1])

    att = attention_from_retention(curve)
    climbs, drops = segments(curve)
    drop_zones = [[a, b] for a, b in drops if b - a >= 1]

    hook = float(np.mean(curve[:3])) * 100
    finish = curve[-1] * 100
    hold = min(100.0, finish / max(hook, 1e-6) * 100)
    drop_share = sum(b - a for a, b in drop_zones) / T
    peak_share = sum(1 for a in att if a >= 70) / len(att)
    ending = float(np.mean(att[-3:]))

    rows = [
        {"key": "hook", "label": "Hook (first 3s)", "value": round(hook, 1),
         "weight": VIRALITY_WEIGHTS["hook"],
         "explain": f"Attention in the first 3s scores {round(hook)}/100."},
        {"key": "drop_free", "label": "Few drop-offs", "value": round((1 - drop_share) * 100, 1),
         "weight": VIRALITY_WEIGHTS["drop_free"],
         "explain": f"{round((1 - drop_share) * 100)}% of the reel stays out of drop zones."},
        {"key": "strong_moments", "label": "Strong-attention moments", "value": round(min(100.0, 200 * peak_share), 1),
         "weight": VIRALITY_WEIGHTS["strong_moments"],
         "explain": f"{round(peak_share * 100)}% of seconds at attention 70+."},
        {"key": "ending", "label": "Ending", "value": round(ending, 1),
         "weight": VIRALITY_WEIGHTS["ending"],
         "explain": f"Attention in the last 3s averages {round(ending)}/100."},
    ]
    wsum = sum(VIRALITY_WEIGHTS.values())
    for row in rows:
        row["points"] = round(row["weight"] * row["value"] / wsum, 1)
        row["vs_typical"] = round(row["weight"] * (row["value"] - 50) / wsum, 1)
    virality = round(sum(r["points"] for r in rows) * (100 / wsum) / 100 * wsum / wsum, 1)
    # points are weighted/100-scale; virality = weighted sum of values
    virality = round(sum(r["weight"] * r["value"] for r in rows) / wsum, 1)

    peak_t = int(np.argmax(att))

    # suggestions: timestamped, against the reel's own best stretch
    suggestions = []
    for a, b in drop_zones[:4]:
        lost = round((curve[a] - curve[min(b, len(curve) - 1)]) * 100, 1)
        suggestions.append({
            "at": f"{fmt(a)}–{fmt(b)}",
            "title": f"Cut {fmt(a)}–{fmt(b)} ({b - a}s of weak pull).",
            "why": f"Retention falls from {round(curve[a] * 100)}% to {round(curve[min(b, len(curve) - 1)] * 100)}% here — {lost} points of viewers leave.",
            "est_lost_pct": lost,
        })
    if T > 6 and ending < 40:
        suggestions.append({
            "at": f"{fmt(max(0, T - 3))}–{fmt(T)}",
            "title": "Fix the ending.",
            "why": f"Attention in the last 3s averages {round(ending)}/100. End on the payoff or loop back to the opening.",
            "est_lost_pct": round(50 - ending, 1),
        })

    return {
        "video": video,
        "caption": caption,
        "duration_s": round(duration_s, 2),
        "demo": demo,
        "method": METHOD,
        "scores": {
            "hook_pct": round(hook, 1),
            "hook_band": band(hook),
            "hold_rate_pct": round(hold, 1),
            "finish_pct": round(finish, 1),
            "virality": virality,
            "virality_band": band(virality),
            "virality_tag": virality_tag(virality),
        },
        "virality_rows": rows,
        "series": {
            "retention": [round(x, 3) for x in curve],
            "attention": [round(x, 1) for x in att],
        },
        "moments": {
            "peak_attention_at_s": peak_t,
            "hooks_s": climbs,
            "drop_zones_s": drop_zones,
        },
        "suggestions": suggestions,
    }


def load_curve(out_dir: Path) -> tuple[list[float], str]:
    """Read r_pred from a TTCC raw output dir. Returns (curve, source)."""
    out_dir = Path(out_dir)
    raw = out_dir / "r_pred.json"
    if raw.is_file():
        data = json.loads(raw.read_text())
        return [float(x) for x in data["r_pred"]], "ttcc-local"
    npy = out_dir / "r_pred.npy"
    if npy.is_file():
        arr = np.load(npy)
        if arr.ndim == 2:
            arr = arr[0]
        return [float(x) for x in arr], "ttcc-local"
    raise FileNotFoundError(f"no r_pred.json or r_pred.npy in {out_dir}")
