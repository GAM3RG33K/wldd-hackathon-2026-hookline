#!/usr/bin/env python
"""Turn TRIBE v2 output plus cheap video signals into a Hookline report.

usage: python hookline_report.py OUT_DIR/<video-stem> [--video PATH]

Reads preds.npy / times.npy / meta.json from tribe_analyze.py. Writes
report.json, report.md and report.png into the same folder.

IMPORTANT: scores are heuristic and normalised WITHIN the video (an average
second scores ~50). The weights below are hypotheses, to be fitted against real
performance data (see validate step). They are not measured retention.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CFG = dict(
    w_neural=0.6, w_signal=0.4,        # blend of brain-response and signal-based attention
    hook_seconds=3,                    # the hook window
    hook_k=1.3,                        # steepness of the hook-score squash
    attn_k=1.2,                        # steepness of the per-second attention squash
    drop_thresh=35, drop_min_len=2,    # attention score below this for >=N s is a drop zone
    scene_thresh=0.3,                  # ffmpeg scene-cut sensitivity
    hold_floor=0.93, hold_span=0.065,  # per-second hold prob range (hypothesis)
    silence_db=-45.0,
)
# MVP0 shows the judgement only. Suggestions are still computed into report.json
# ("recommendations") but not displayed until they stop being generic (MVP1).
SHOW_SUGGESTIONS = False

# Destrieux atlas labels per functional group. Attention index = sensory + attention +
# language activity minus default-mode activity (engaged vs mind-wandering).
GROUPS = {
    "visual": ["Pole_occipital", "S_calcarine", "G_cuneus", "G_oc-temp_med-Lingual",
               "G_occipital_middle", "G_occipital_sup", "S_oc_middle_and_Lunatus",
               "S_oc_sup_and_transversal", "G_and_S_occipital_inf", "S_occipital_ant",
               "S_oc-temp_med_and_Lingual"],
    "auditory": ["G_temp_sup-G_T_transv", "G_temp_sup-Lateral", "G_temp_sup-Plan_tempo",
                 "S_temporal_transverse"],
    "language": ["G_front_inf-Opercular", "G_front_inf-Triangul", "G_temporal_middle",
                 "S_temporal_sup"],
    "attention": ["S_intrapariet_and_P_trans", "G_parietal_sup", "S_precentral-sup-part"],
    "default_mode": ["G_precuneus", "G_cingul-Post-dorsal", "G_cingul-Post-ventral",
                     "G_pariet_inf-Angular", "G_and_S_cingul-Ant"],
}


# ---------------------------------------------------------------- helpers
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def zs(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return np.zeros_like(x) if s < 1e-9 else (x - x.mean()) / s


def smooth(x, k=(0.25, 0.5, 0.25)):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x
    return np.convolve(np.pad(x, 1, mode="edge"), k, mode="valid")


def run(cmd, binary=False):
    return subprocess.run(cmd, capture_output=True, text=not binary)


def fmt(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


# Viral-potential tiers (uncalibrated until VIRALITY_MODEL is fitted on the sample corpus).
VIRALITY_TIERS = [(35, "Low reach"), (55, "Average"), (70, "Above average"), (85, "Popular"),
                  (101, "Viral potential")]
# Written by tribe/fit_virality.py (logistic fit: known-viral outliers vs low-view reels).
# Missing file = interim heuristic below.
_VM = Path(__file__).with_name("virality_model.json")
VIRALITY_MODEL = json.loads(_VM.read_text()) if _VM.exists() else None


def virality_tag(score):
    return next(name for cut, name in VIRALITY_TIERS if score < cut)


def band(score):
    return ("Bad" if score < 30 else "Weak" if score < 50 else "Average" if score < 70
            else "Strong" if score < 85 else "Viral-grade")


# ---------------------------------------------------------------- video signals
def probe_duration(video):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)])
    return float(r.stdout.strip())


def per_second(values, times, T, how="mean"):
    out = np.zeros(T)
    cnt = np.zeros(T)
    for v, t in zip(values, times):
        i = int(t)
        if 0 <= i < T:
            out[i] += v
            cnt[i] += 1
    return out if how == "sum" else np.divide(out, np.maximum(cnt, 1))


def audio_signal(video, T):
    r = run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
             "-f", "f32le", "-"], binary=True)
    a = np.frombuffer(r.stdout, dtype=np.float32)
    if len(a) < 16000:
        return np.full(T, -80.0), None
    db = np.full(T, -80.0)
    for s in range(T):
        seg = a[s * 16000:(s + 1) * 16000]
        if len(seg):
            db[s] = 20 * math.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-6)
    loud = np.where(np.abs(a) > 0.01)[0]
    onset = float(loud[0] / 16000) if len(loud) else None
    return db, onset


def cut_times(video, thresh):
    r = run(["ffmpeg", "-i", str(video), "-an", "-vf", f"select='gt(scene,{thresh})',showinfo",
             "-f", "null", "-"])
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr)]


def motion_signal(video, T):
    r = run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-an", "-vf",
             "fps=5,scale=64:64,format=gray", "-f", "rawvideo", "-"], binary=True)
    frames = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(frames) // (64 * 64)
    if n < 2:
        return np.zeros(T)
    frames = frames[: n * 64 * 64].reshape(n, 64, 64).astype(np.float32)
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return per_second(d, (np.arange(1, n)) / 5.0, T)


def word_signal(tsv, T):
    if not tsv or not Path(tsv).exists():
        return np.zeros(T), None, []
    try:
        w = pd.read_csv(tsv, sep="\t", keep_default_na=False)  # a word "nan" is text, not NaN
    except pd.errors.EmptyDataError:
        return np.zeros(T), None, []
    if len(w) == 0:
        return np.zeros(T), None, []
    cnt = per_second(np.ones(len(w)), w["start"].values, T, how="sum")
    return cnt, float(w["start"].min()), w.to_dict("records")


# ---------------------------------------------------------------- neural series
def neural_series(preds):
    """Mean predicted response per functional group, (T,) each. None if no atlas."""
    try:
        from nilearn import datasets
        a = datasets.fetch_atlas_surf_destrieux(verbose=0)
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in a["labels"]]
        vert = np.concatenate([np.asarray(a["map_left"]), np.asarray(a["map_right"])])
        if len(vert) != preds.shape[1]:
            raise ValueError(f"atlas has {len(vert)} vertices, preds {preds.shape[1]}")
        out = {}
        for g, names in GROUPS.items():
            ids = [i for i, l in enumerate(labels) if l in names]
            mask = np.isin(vert, ids)
            out[g] = preds[:, mask].mean(axis=1)
        return out
    except Exception as e:  # network blocked, nilearn missing, ...
        print(f"[warn] atlas unavailable ({e}); using whole-cortex magnitude instead", file=sys.stderr)
        return {"cortex": np.linalg.norm(preds, axis=1)}


# ---------------------------------------------------------------- scoring
def analyse(out_dir, video=None):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    video = Path(video or meta["video"])
    preds = np.load(out_dir / "preds.npy")
    times = np.load(out_dir / "times.npy") if (out_dir / "times.npy").exists() else np.arange(len(preds))
    dur = probe_duration(video)
    T = max(int(math.ceil(dur)), 1)
    grid = np.arange(T)

    # -- brain response -> attention index
    nets = neural_series(preds)
    net_t = {g: np.interp(grid, times, v) for g, v in nets.items()}
    net_z = {g: zs(v) for g, v in net_t.items()}
    if "visual" in net_z:
        engaged = np.mean([net_z[g] for g in ("visual", "auditory", "language", "attention")], axis=0)
        N = zs(engaged - 0.5 * net_z["default_mode"])
    else:
        N = zs(net_t["cortex"])

    # -- cheap signals
    db, audio_onset = audio_signal(video, T)
    cuts = cut_times(video, CFG["scene_thresh"])
    cut_ps = per_second(np.ones(len(cuts)), cuts, T, how="sum")
    motion = motion_signal(video, T)
    words_ps, first_word, words = word_signal(meta.get("transcript"), T)
    # Transcription can be switched off per job: then speech is unknown, not absent.
    speech_known = meta.get("modalities", {}).get("text", True)
    sig = [zs(smooth(cut_ps)), zs(motion), zs(np.maximum(db, CFG["silence_db"]))]
    if speech_known:
        sig.append(zs(smooth(words_ps)))
    S = zs(np.mean(sig, axis=0))

    # -- combined per-second attention
    A_z = zs(smooth(CFG["w_neural"] * N + CFG["w_signal"] * S))
    A = 100 * sigmoid(CFG["attn_k"] * A_z)

    # -- hook score (first N seconds)
    hs = CFG["hook_seconds"]
    head = A_z[:hs]
    bonus = 0.0
    notes = []
    speech_early = first_word is not None and first_word <= 1.5
    if speech_early:
        bonus += 0.3
    elif speech_known:
        notes.append("no spoken line in the first 1.5s")
    if cuts and min(cuts) <= 2.0 or motion[:2].max() > np.median(motion) * 1.3:
        bonus += 0.3
    else:
        notes.append("static opening: no cut or strong movement in the first 2s")
    if audio_onset is None or audio_onset > 1.0:
        bonus -= 0.4
        notes.append("silent start: sound begins late or never")
    hook_raw = 0.5 * float(head.mean()) + 0.5 * float(head.max()) + bonus
    hook = float(100 * sigmoid(CFG["hook_k"] * hook_raw))

    # -- hold / survival
    h = CFG["hold_floor"] + CFG["hold_span"] * sigmoid(1.5 * A_z)
    surv = np.ones(T)
    for t in range(T):
        if t < hs:
            surv[t] = 1 - (1 - hook / 100) * (t + 1) / hs
        else:
            surv[t] = surv[t - 1] * h[t]
    finish = float(surv[-1] * 100)
    hold = float(finish / max(hook, 1e-6) * 100)

    # -- moments
    rise = np.array([A[min(t + 2, T - 1)] - A[t] for t in range(T)])
    peak_hook_t = int(np.argmax(rise)) + 1 if T > 2 else 0
    peak_t = int(np.argmax(A))
    win = 3
    wmean = np.array([A[i:i + win].mean() for i in range(max(T - win + 1, 1))])
    weakest = int(np.argmin(wmean))
    strongest = int(np.argmax(wmean))

    drops, run_start = [], None
    for t in range(T + 1):
        low = t < T and A[t] < CFG["drop_thresh"]
        if low and run_start is None:
            run_start = t
        if not low and run_start is not None:
            if t - run_start >= CFG["drop_min_len"]:
                drops.append((run_start, t))
            run_start = None
    if not drops:  # always surface the weakest stretch
        drops = [(weakest, min(weakest + win, T))]

    # -- recommendations (rule-based, tied to timestamps)
    recs = []
    for n in notes:
        recs.append({"at": "0:00-0:03", "issue": n, "fix": {
            "no spoken line in the first 1.5s": "Start with the claim, question or payoff line inside the first second.",
            "static opening: no cut or strong movement in the first 2s": "Add a cut, zoom or movement in the first 2s, or open on the most striking frame.",
            "silent start: sound begins late or never": "Put sound (voice, hit, music) on frame 0.",
        }[n]})
    if peak_t > hs + 2 and hook < 60:
        recs.append({"at": fmt(peak_t), "issue": "strongest moment is buried",
                     "fix": f"Your strongest moment is at {fmt(peak_t)}. Consider opening on it, or tease it in the first 3s."})
    for a, b in drops:
        seg = slice(a, b)
        why = []
        if cut_ps[seg].sum() == 0:
            why.append("no visual change")
        if motion[seg].mean() < np.median(motion) * 0.8:
            why.append("low movement")
        if db[seg].std() < 2.0:
            why.append("flat audio")
        if speech_known and words_ps[seg].sum() == 0:
            why.append("no speech")
        fix = "Re-hook: " + ("add a cut, zoom or b-roll; " if "no visual change" in why or "low movement" in why else "") \
              + ("add a sound hit or music lift; " if "flat audio" in why else "") \
              + "tease what is coming or ask a question."
        recs.append({"at": f"{fmt(a)}-{fmt(b)}", "issue": "attention drops" + (f" ({', '.join(why)})" if why else ""),
                     "fix": fix})
    if T > 6 and A[-3:].mean() < 40:
        recs.append({"at": f"{fmt(T - 3)}-{fmt(T)}", "issue": "weak ending",
                     "fix": "End on a payoff or loop back to the opening so viewers rewatch."})

    # -- raw features, comparable ACROSS videos (inputs for the virality fit; never account data)
    drop_share = float(sum(b - a for a, b in drops) / T)
    eng_raw = (np.mean([net_t[g] for g in ("visual", "auditory", "language", "attention")], axis=0)
               - 0.5 * net_t["default_mode"]) if "visual" in net_t else net_t["cortex"]
    features = {
        "hook": round(hook, 2),
        "engagement_raw": round(float(eng_raw.mean()), 5),          # absolute TRIBE engagement level
        "hook_lift": round(float((eng_raw[:hs].mean() - eng_raw.mean()) / (eng_raw.std() + 1e-9)), 3),
        "peak_share": round(float((A >= 70).mean()), 3),            # seconds at strong attention
        "drop_share": round(drop_share, 3),                         # seconds inside drop zones
        "ending": round(float(A[-3:].mean()), 2),
        "volatility": round(float(np.abs(np.diff(A)).mean()), 2) if T > 1 else 0.0,
        "cuts_per_s": round(len(cuts) / max(dur, 1e-6), 3),
        "motion": round(float(np.mean(motion)), 4),
        "loudness_db": round(float(np.mean(np.maximum(db, CFG["silence_db"]))), 2),
        "words_per_s": round(float(words_ps.sum() / max(dur, 1e-6)), 3) if speech_known else None,
        "duration_s": round(dur, 2),
        **{f"net_{g}": round(float(v.mean()), 5) for g, v in net_t.items()},
    }
    pk = [i for i in range(1, T - 1) if A[i] >= A[i - 1] and A[i] >= A[i + 1]
          and A[i] - max(A[max(0, i - 6):i].min(), A[i + 1:i + 7].min()) >= 15]
    peaks_kept = []
    for i in pk:  # at least 3s apart
        if not peaks_kept or i - peaks_kept[-1] >= 3:
            peaks_kept.append(i)
    gaps = np.diff(peaks_kept) if len(peaks_kept) > 2 else np.array([])
    features.update(peaks=len(peaks_kept), peaks_per_10s=round(10 * len(peaks_kept) / T, 3),
                    peak_gap_cv=round(float(gaps.std() / gaps.mean()), 3) if len(gaps) and gaps.mean() else 1.0)
    if VIRALITY_MODEL:
        m = VIRALITY_MODEL
        z = m["bias"] + sum(w * (((features.get(k) or 0.0) - m["mean"][k]) / m["std"][k])
                            for k, w in m["weights"].items())
        virality = float(100 * sigmoid(z))
        virality_method = m["label"]
    else:
        # Interim, uncalibrated: no constant term (avg attention is ~50 by design) and no
        # length penalty (hold). Hook, how much of the video avoids drops, strong moments, ending.
        virality = float(0.40 * hook + 0.25 * (1 - drop_share) * 100
                         + 0.20 * min(100.0, 200 * features["peak_share"]) + 0.15 * features["ending"])
        virality_method = "interim heuristic, uncalibrated (hook, drop-free share, strong moments, ending)"
    result = {
        "video": str(video), "duration_s": round(dur, 2), "method": "heuristic, within-video normalised, uncalibrated",
        "scores": {
            "hook_pct": round(hook, 1), "hook_band": band(hook),
            "hold_rate_pct": round(min(hold, 100), 1), "finish_pct": round(finish, 1),
            "avg_attention": round(float(A.mean()), 1), "virality": round(virality, 1),
            "virality_band": band(virality), "virality_tag": virality_tag(virality),
            "virality_method": virality_method,
        },
        "features": features,
        "moments": {
            "peak_hook_at_s": peak_hook_t, "peak_attention_at_s": peak_t,
            "strongest_window_s": [strongest, strongest + win], "weakest_window_s": [weakest, weakest + win],
            "drop_zones_s": [[a, b] for a, b in drops],
        },
        "network_mean_response": {g: round(float(v.mean()), 4) for g, v in net_t.items()},
        "recommendations": recs,
        "series": {"attention": [round(float(x), 1) for x in A], "survival_pct": [round(float(x) * 100, 1) for x in surv],
                   "cuts_s": cuts},
    }
    return result, dict(A=A, surv=surv, net_z=net_z, db=db, motion=motion, cuts=cuts, words=words,
                        drops=drops, peak_hook_t=peak_hook_t, peak_t=peak_t, T=T, hs=hs)


# ---------------------------------------------------------------- output
def finalize(report, extra):
    """Merge what the GPU Space knows (modalities, speech gate, transcript, audio match) into a
    scored report. Shared by the Space and the local re-scorer so both build identical reports."""
    extra = extra or {}
    mods = extra.get("modalities") or {"video": True, "audio": True, "text": True}
    report["neural_source"] = extra.get("neural_source", report.get("neural_source"))
    report["modalities"] = mods
    gate = extra.get("speech_gate")
    if gate:
        report["speech_gate"] = {k: v for k, v in gate.items() if not k.endswith("_per_second")}
        report.setdefault("series", {})["speech_prob"] = gate.get("speech_per_second", [])
        report["series"]["vocals_prob"] = gate.get("vocals_per_second", [])
    segs = extra.get("transcript_segments")
    info = extra.get("transcript_info") or {}
    report["transcript"] = {
        "used_by_model": bool(mods.get("text")) and bool(segs),
        "status": ("used" if mods.get("text") and segs else
                   f"skipped: {(gate or {}).get('reason', 'transcript off')}" if gate and not mods.get("text") else
                   "off" if not mods.get("text") else "no speech found"),
        "model": info.get("model"), "lang": info.get("lang"), "detected": info.get("detected"),
        "words": info.get("words"), "segments": segs or [],
        "dropped_by_guard": extra.get("transcript_dropped"),
    }
    am = extra.get("audio_match") or {}
    report["audio"] = {"type": (gate or {}).get("audio_type"), "song": am.get("song"),
                       "reused_by": am.get("reuse", []), "note": "context only, not a score input"}
    gated_off = bool(gate) and gate.get("decision") == "transcript off"
    if gated_off and mods.get("video") and mods.get("audio"):
        report["method"] += f"; transcript skipped by speech gate ({gate['reason']}), scored on visuals + sound"
    elif not all(mods.values()):
        used = "+".join(m for m, on in mods.items() if on)
        report["method"] += f"; partial modalities ({used}), diagnostic run"
    return report


def write_md(r, path):
    s, m = r["scores"], r["moments"]
    lines = [f"# Hookline report", "", f"`{Path(r['video']).name}`, {r['duration_s']}s", "",
             f"- **Hook score:** {s['hook_pct']}% ({s['hook_band']})",
             f"- **Hold rate:** {s['hold_rate_pct']}% of hooked viewers stay to the end",
             f"- **Finish rate:** {s['finish_pct']}%",
             f"- **Viral potential:** {s['virality']} ({s.get('virality_tag', s['virality_band'])}; "
             f"{s.get('virality_method', 'uncalibrated')})", "",
             f"Peak hook at {fmt(m['peak_hook_at_s'])}. Strongest window {fmt(m['strongest_window_s'][0])}-{fmt(m['strongest_window_s'][1])}, "
             f"weakest {fmt(m['weakest_window_s'][0])}-{fmt(m['weakest_window_s'][1])}.", ""]
    drops = ", ".join(f"{fmt(a)}-{fmt(b)}" for a, b in m.get("drop_zones_s", [])) or "none"
    lines += [f"Drop zones: {drops}.", ""]
    if SHOW_SUGGESTIONS:
        lines += ["## What to change", ""]
    for x in (r["recommendations"] if SHOW_SUGGESTIONS else []):
        lines.append(f"- **{x['at']}**, {x['issue']}. {x['fix']}")
    lines += ["", f"_{r['method']}_"]
    path.write_text("\n".join(lines))


def write_png(r, d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = r["scores"]
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.9, 2.2, 1.6, 1.3], hspace=0.55)
    cards = [("Hook", f"{s['hook_pct']:.0f}%", s["hook_band"]), ("Hold rate", f"{s['hold_rate_pct']:.0f}%", "stay to end"),
             ("Finish", f"{s['finish_pct']:.0f}%", "watch-through"), ("Virality", f"{s['virality']:.0f}", s["virality_band"])]
    for i, (t, v, sub) in enumerate(cards):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        ax.text(0.5, 0.72, v, ha="center", fontsize=30, weight="bold")
        ax.text(0.5, 0.40, t, ha="center", fontsize=12)
        ax.text(0.5, 0.15, sub, ha="center", fontsize=10, color="grey")
    t = np.arange(d["T"])
    ax = fig.add_subplot(gs[1, :])
    for a, b in d["drops"]:
        ax.axvspan(a, b, color="tab:red", alpha=0.15)
    ax.axvspan(0, d["hs"], color="tab:green", alpha=0.10)
    ax.plot(t, d["A"], color="tab:blue", lw=2, label="attention (0-100)")
    ax.plot(t, d["surv"] * 100, color="tab:orange", lw=1.5, ls="--", label="viewers still watching %")
    for c in d["cuts"]:
        ax.axvline(c, color="grey", alpha=0.25, lw=0.8)
    ax.scatter([d["peak_hook_t"], d["peak_t"]], [d["A"][min(d["peak_hook_t"], d["T"] - 1)], d["A"][d["peak_t"]]],
               color=["tab:green", "tab:purple"], zorder=5, s=60)
    ax.set_ylim(0, 105); ax.set_xlim(0, d["T"] - 1); ax.set_xlabel("seconds"); ax.legend(loc="upper right")
    ax.set_title("Attention over time (green = hook window, red = drop zones, grey = cuts)")
    ax2 = fig.add_subplot(gs[2, :])
    names = list(d["net_z"]); M = np.array([d["net_z"][n] for n in names])
    im = ax2.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=[0, d["T"], len(names), 0])
    ax2.set_yticks(np.arange(len(names)) + 0.5); ax2.set_yticklabels(names); ax2.set_xlabel("seconds")
    ax2.set_title("Predicted brain-network response (relative to this video)")
    fig.colorbar(im, ax=ax2, fraction=0.015)
    ax3 = fig.add_subplot(gs[3, :]); ax3.axis("off")
    if SHOW_SUGGESTIONS:
        txt = "\n".join(f"{x['at']}  {x['issue']}: {x['fix']}" for x in r["recommendations"][:6]) or "No issues found."
        ax3.text(0, 1, "What to change\n" + txt, va="top", fontsize=9, wrap=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--video", type=Path)
    args = ap.parse_args()
    r, d = analyse(args.out_dir, args.video)
    (args.out_dir / "report.json").write_text(json.dumps(r, indent=1))
    write_md(r, args.out_dir / "report.md")
    write_png(r, d, args.out_dir / "report.png")
    print((args.out_dir / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
