import sys, os, traceback

# Disable HF transfer in subprocesses to prevent missing module errors
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Authenticate globally if HF_TOKEN is provided in Space Secrets
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("🔑 Successfully authenticated to Hugging Face Hub with HF_TOKEN!")
    except Exception as e:
        print(f"⚠️ Hugging Face login note: {e}")
else:
    print("⚠️ Notice: HF_TOKEN is not configured in Space Secrets. Gated models require HF_TOKEN.")

# spaces MUST be imported at the very top before any CUDA/torch calls
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

import subprocess, json, tempfile, shutil, re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import gradio as gr
from tribev2.demo_utils import TribeModel

print("🚀 Starting MindMatrix / Hookline AI Space...")

# Auto-detect Hardware: Dedicated GPU vs ZeroGPU
is_zerogpu = "SPACES_ZERO_GPU" in os.environ
has_dedicated_gpu = torch.cuda.is_available() and not is_zerogpu
startup_device = "cuda" if has_dedicated_gpu else "cpu"

print(f"🚀 Initializing TRIBE v2 on device: {startup_device.upper()} (Dedicated GPU={has_dedicated_gpu}, ZeroGPU={is_zerogpu})...")
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder="./tribe_cache",
    device=startup_device
)
print(f"✅ TRIBE v2 foundation model and extractors loaded on {startup_device.upper()}!")

# Pre-cache speech model at startup so user requests don't download it on the fly
print("🎙️ Pre-caching speech transcription models...")
try:
    from huggingface_hub import snapshot_download
    snapshot_download("Systran/faster-whisper-base")
    print("✅ Speech models cached successfully!")
except Exception as e:
    print(f"⚠️ Speech pre-cache note: {e}")

if hf_token:
    print("🦙 Pre-caching LLaMA 3.2 model & tokenizer at startup...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("meta-llama/Llama-3.2-3B", token=hf_token)
        print("✅ LLaMA 3.2 model cached successfully!")
    except Exception as e:
        print(f"⚠️ LLaMA pre-cache note: {e}")




SEG_FALLBACK_SEC = 1.5
# TRIBE predictions start low while the model "warms up", so a raw hook-vs-rest comparison
# looks negative for almost every reel. Median hook-vs-rest across sample reels (n=7);
# recalibrate after the full 110-video batch.
TYPICAL_HOOK_VS_REST_PCT = -8.0


def segment_times(segments, n):
    """Start time (s) of each predicted brain-response segment."""
    try:
        t = np.array([float(s.start) for s in segments], dtype=float)
        if len(t) == n and n > 0 and np.all(np.diff(t) > 0):
            return t - t[0]
    except Exception:
        pass
    return np.arange(n) * SEG_FALLBACK_SEC


def _runs(mask, times, min_len_sec):
    """Contiguous [start, end] time spans where mask is True and lasting >= min_len_sec."""
    spans, start = [], None
    step = float(np.median(np.diff(times))) if len(times) > 1 else SEG_FALLBACK_SEC
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            t0, t1 = float(times[start]), float(times[i - 1]) + step
            if t1 - t0 >= min_len_sec:
                spans.append((round(t0, 1), round(t1, 1)))
            start = None
    return spans


def analyze_brain_response(preds, times):
    """Turns TRIBE v2 predicted cortical activity into a within-video engagement profile.

    All scores are RELATIVE to this video's own average (50 = average moment of this video).
    They show where a reel is strong or weak; they are NOT a view-count prediction.
    """
    n = preds.shape[0]
    strength = np.abs(preds).mean(axis=1)                      # overall cortical response per moment
    mu, sd = float(strength.mean()), float(strength.std()) or 1e-9
    z = (strength - mu) / sd
    curve = np.clip(50 + 15 * z, 0, 100)

    # Brain "scene change": how different the whole-cortex pattern is from the previous moment
    shift = np.zeros(n)
    for i in range(1, n):
        c = np.corrcoef(preds[i - 1], preds[i])[0, 1]
        shift[i] = 0.0 if np.isnan(c) else 1.0 - c
    dur = float(times[-1] + (times[1] - times[0] if n > 1 else SEG_FALLBACK_SEC))

    hook_mask = times < 3.0
    if hook_mask.sum() == 0:
        hook_mask[0] = True
    rest = strength[~hook_mask] if (~hook_mask).any() else strength
    hook_vs_rest = (strength[hook_mask].mean() / rest.mean() - 1) * 100
    hook_vs_typical = hook_vs_rest - TYPICAL_HOOK_VS_REST_PCT

    tail_mask = times >= dur * 0.8
    if tail_mask.sum() == 0:
        tail_mask[-1] = True
    ending_vs_avg = (strength[tail_mask].mean() / mu - 1) * 100

    # Peak moments: top 3 local maxima at least 3s apart
    peaks = []
    for i in np.argsort(-strength):
        if all(abs(times[i] - times[j]) >= 3.0 for j in peaks):
            peaks.append(int(i))
        if len(peaks) == 3:
            break

    # Only clear, sustained dips after the hook window: >=3s below -0.8 SD, averaging <= -1 SD
    dead_zones = [(t0, t1) for t0, t1 in _runs((z < -0.8) & (times >= 3.0), times, min_len_sec=3.0)
                  if z[(times >= t0) & (times < t1)].mean() <= -1.0]
    dead_sec = sum(b - a for a, b in dead_zones)

    return {
        "curve": [round(float(v), 1) for v in curve],
        "shift": [round(float(v), 4) for v in shift],
        "times": [round(float(t), 2) for t in times],
        "duration": round(dur, 1),
        "hook_vs_rest_pct": round(float(hook_vs_rest), 1),
        "hook_vs_typical_pct": round(float(hook_vs_typical), 1),
        "ending_vs_avg_pct": round(float(ending_vs_avg), 1),
        "peaks": [{"time": round(float(times[i]), 1), "score": round(float(curve[i]), 1)} for i in peaks],
        "dead_zones": dead_zones,
        "dead_time_pct": round(100 * dead_sec / max(dur, 1e-6), 1),
        # Raw (un-normalised) stats, comparable across videos — used for offline calibration
        "raw": {
            "strength_mean": mu,
            "strength_std": sd,
            "strength_first3s": float(strength[hook_mask].mean()),
            "strength_last20pct": float(strength[tail_mask].mean()),
            "shift_mean": float(shift[1:].mean()) if n > 1 else 0.0,
            "n_segments": int(n),
        },
    }


def video_cues(video_path, n_sec):
    """Per-second loudness (dB) and on-screen motion, using ffmpeg only."""
    n_sec = max(1, int(n_sec))
    sr = 16000
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path), "-ac", "1", "-ar", str(sr),
                              "-f", "s16le", "-"], capture_output=True, timeout=120).stdout
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        y = np.zeros(0, dtype=np.float32)
    loud = np.full(n_sec, -90.0)
    for i in range(n_sec):
        seg = y[i * sr:(i + 1) * sr]
        if len(seg):
            loud[i] = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)

    w, h, fps = 64, 112, 4
    motion = np.zeros(n_sec)
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path), "-vf", f"fps={fps},scale={w}:{h}",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, timeout=180).stdout
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).astype(np.float32)
        diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
        for i in range(n_sec):
            d = diffs[i * fps:(i + 1) * fps]
            motion[i] = d.mean() if len(d) else 0.0
    except Exception:
        pass
    return loud, motion


SCENE_CUT_THRESHOLD = 0.35


def detect_cuts(video_path):
    """Real shot cuts (seconds), from ffmpeg's frame-difference scene detector."""
    try:
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(video_path), "-an",
                            "-vf", f"scale=160:-2,select='gt(scene,{SCENE_CUT_THRESHOLD})',showinfo",
                            "-f", "null", "-"], capture_output=True, text=True, timeout=180)
        cuts = []
        for line in r.stderr.splitlines():
            if "pts_time:" in line:
                cuts.append(round(float(line.split("pts_time:")[1].split()[0]), 2))
        return sorted(cuts)
    except Exception as e:
        print(f"Cut detection note: {e}")
        return []


def shots_from_cuts(cuts, duration):
    edges = [0.0] + [c for c in cuts if 0 < c < duration] + [duration]
    return [(round(a, 1), round(b, 1)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def extract_words(events):
    """[(start_sec, end_sec, word)] from TRIBE's transcription (Whisper) events."""
    try:
        w = events[events["type"] == "Word"].sort_values("start")
        out = []
        for r in w.itertuples():
            text = str(r.text).strip()
            if text:
                dur = float(getattr(r, "duration", 0.3) or 0.3)
                out.append((float(r.start), float(r.start) + dur, text))
        return out
    except Exception as e:
        print(f"Transcript note: {e}")
        return []


def words_between(words, t0, t1, max_words=18):
    said = [w for t, _, w in words if t0 <= t < t1]
    if not said:
        return ""
    text = " ".join(said[:max_words]) + (" …" if len(said) > max_words else "")
    return text.replace("*", "").replace("|", "/")


def explain_span(t0, t1, loud, motion, shots=None):
    """Likely cause of low brain response in [t0, t1), compared with the rest of the video."""
    sel = np.zeros(len(loud), dtype=bool)
    sel[int(t0):max(int(t0) + 1, int(t1))] = True
    if sel.all() or not sel.any():
        return None
    causes = []
    zl, rl = loud[sel].mean(), loud[~sel].mean()
    if zl < -45:
        causes.append(("silence", "the audio goes **silent** — add music, a sound effect or voiceover"))
    elif zl < rl - 6:
        causes.append(("quiet", f"the audio gets **quieter** ({zl - rl:.0f} dB vs the rest) — raise the energy: music swell, SFX or louder delivery"))
    zm, rm = motion[sel].mean(), motion[~sel].mean()
    long_shot = None
    for a, b in (shots or []):
        overlap = min(b, t1) - max(a, t0)
        if b - a >= 6.0 and overlap >= min(3.0, (t1 - t0) / 2):
            long_shot = (a, b)
    if long_shot:
        a, b = long_shot
        what = "one long shot with **no cut for {:.0f}s** ({}s – {}s)".format(b - a, a, b)
        if rm > 0 and zm < 0.6 * rm:
            what += f" and {100 * (1 - zm / rm):.0f}% less movement"
        causes.append(("no_cut", what + " — add a cut, zoom, b-roll or text pop"))
    elif rm > 0 and zm < 0.6 * rm:
        causes.append(("static", f"the visuals are **static** ({100 * (1 - zm / rm):.0f}% less movement) — add a zoom, b-roll or text pop"))
    return causes


def estimate_retention(curve, times):
    """ESTIMATED retention shape (not measured): a generic short-form decay
    (steep in the first 3s, slow after) where each second's drop-off rate rises
    when predicted brain response is below the video's average and falls when above.
    Only the shape is meaningful; absolute % values are uncalibrated."""
    z = (np.array(curve, dtype=float) - 50.0) / 15.0
    ret, out = 100.0, []
    for t, zi in zip(times, z):
        out.append(round(ret, 1))
        base = 0.10 if t < 3 else 0.025
        ret *= 1 - min(0.5, base * float(np.exp(-0.45 * np.clip(zi, -3, 3))))
    return out


def label(pct, good=8.0, bad=-8.0):
    if pct >= good:
        return "🟢 Strong"
    if pct <= bad:
        return "🔴 Weak"
    return "🟡 Average"


def dead_label(pct):
    return "🟢 Strong" if pct == 0 else ("🔴 Weak" if pct >= 20 else "🟡 Average")


def detect_black_and_freeze(video_path):
    """Black stretches and exactly-frozen picture stretches (seconds), via ffmpeg."""
    try:
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(video_path), "-an", "-vf",
                            "scale=160:-2,blackdetect=d=0.1:pix_th=0.10,freezedetect=n=0.0002:d=0.5",
                            "-f", "null", "-"], capture_output=True, text=True, timeout=180).stderr
    except Exception as e:
        print(f"Black/freeze note: {e}")
        return [], []
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r)]
    fs = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", r)]
    fe = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", r)]
    freezes = []
    for i, a in enumerate(fs):                                  # merge back-to-back freeze pieces
        b = fe[i] if i < len(fe) else None
        if freezes and freezes[-1][1] is not None and a - freezes[-1][1] <= 0.3:
            freezes[-1] = (freezes[-1][0], b)
        else:
            freezes.append((a, b))
    return black, freezes


def frame_has_text(video_path, t):
    """True if a (mostly black) frame carries bright text/graphics."""
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video_path), "-frames:v", "1",
                              "-vf", "scale=160:284", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, timeout=30).stdout
        px = np.frombuffer(raw, dtype=np.uint8)
        return len(px) > 0 and float((px > 128).mean()) > 0.002
    except Exception:
        return False


def _mean_db(loud, t0, t1):
    seg = loud[int(t0):max(int(t0) + 1, int(np.ceil(t1)))]
    return float(seg.mean()) if len(seg) else -90.0


def find_frame_issues(video_path, duration, loud):
    issues = []
    black, freezes = detect_black_and_freeze(video_path)
    mid = [(a, b) for a, b in black if a > 0.05 and b < duration - 0.6]
    flashes = sum(1 for a, b in mid if b - a <= 0.2)
    for a, b in black:
        length = b - a
        if frame_has_text(video_path, (a + b) / 2):
            continue                                            # designed text card on black
        if a <= 0.05 and length >= 0.3:
            issues.append(dict(start=0.0, end=round(b, 1), icon="🖤", title="Video starts on a black screen",
                               what=f"The first {length:.1f}s are black — on a scrolling feed that looks like nothing is there.",
                               fix="Start on your first real frame."))
        elif b >= duration - 0.6:
            if length > 0.5:
                issues.append(dict(start=round(a, 1), end=round(b, 1), icon="🖤", title="Black frames at the end",
                                   what=f"{length:.1f}s of black at the end. Reels loop, so this breaks the loop and feels like a dead ending.",
                                   fix="Trim the trailing black so the reel loops straight back to the start."))
        elif length <= 0.2 and flashes >= 3:
            continue                                            # repeated flash transitions = style
        elif length >= 0.3:
            silent = _mean_db(loud, a, b) < -45
            issues.append(dict(start=round(a, 1), end=round(b, 1), icon="⬛",
                               title="Black screen mid-video" + (" with silence" if silent else ""),
                               what=("Picture and sound both drop out — viewers will think footage is missing."
                                     if silent else
                                     "The picture goes black while audio continues. Fine if it is a deliberate pause; otherwise it looks like missing footage."),
                               fix="Remove it, or put text/b-roll on screen if the pause is intentional."))
    for a, b in freezes:
        end_t = duration if b is None else b
        length = end_t - a
        at_end = end_t >= duration - 0.3
        if at_end and length >= 1.0 and not any(i["title"] == "Black frames at the end" for i in issues):
            issues.append(dict(start=round(a, 1), end=round(end_t, 1), icon="🧊", title="Ends on a frozen frame",
                               what=f"The picture is frozen for the last {length:.1f}s.",
                               fix="Cut the reel where the action ends so it loops cleanly."))
        elif not at_end and length >= 3.0:
            issues.append(dict(start=round(a, 1), end=round(end_t, 1), icon="🧊", title="Frozen picture",
                               what=f"The exact same frame stays on screen for {length:.1f}s — viewers may think the video is stuck.",
                               fix="Add a slow zoom/pan, or shorten the hold."))
    return issues


def find_speech_issues(words, loud, duration):
    issues = []
    if len(words) < 15:                                         # little/no speech: music or text-only reel
        return issues
    first = words[0][0]
    if first >= 1.5 and _mean_db(loud, 0, first) < -35:
        issues.append(dict(start=0.0, end=round(first, 1), icon="🔇", title="Silent opening",
                           what=f"Nothing is said or heard for the first {first:.1f}s.",
                           fix="Start speaking (or bring in sound) within the first second."))
    for (s0, e0, _), (s1, _, _) in zip(words, words[1:]):
        gap = s1 - e0
        if gap >= 1.5 and _mean_db(loud, e0, s1) < max(-40.0, float(np.median(loud)) - 12):
            issues.append(dict(start=round(e0, 1), end=round(s1, 1), icon="🔇", title="Dead air",
                               what=f"A {gap:.1f}s pause with no speech and almost no sound.",
                               fix="Cut the pause, or fill it with music, a sound effect or on-screen text."))
    return issues


def _cause_text(causes):
    return " and ".join(c[1] for c in causes)


def build_timeline(a):
    """Time-ordered list of what to improve and where, plus what already works."""
    issues, good = [], []
    best = a["peaks"][0]["time"] if a["peaks"] else None

    if a["hook_vs_typical_pct"] <= -10:
        what = f"Brain response in the first 3s is {a['hook_vs_typical_pct']:+.0f}% vs a typical reel's hook."
        if a.get("hook_causes"):
            what += " In the opening, " + _cause_text(a["hook_causes"]) + "."
        if a.get("hook_quote"):
            what += f' Opening line: *"{a["hook_quote"]}"*'
        issues.append(dict(start=0.0, end=3.0, icon="🎣", title="Weak hook", what=what,
                           fix=f"Open with your strongest moment (**{best}s**) or tease it in the first frame."))
    elif a["hook_vs_typical_pct"] >= 10:
        good.append(f"🎣 **Strong hook** ({a['hook_vs_typical_pct']:+.0f}% vs a typical reel)")

    ending_flagged = a["ending_vs_avg_pct"] <= -8
    ending_start = round(a["duration"] * 0.8, 1)
    folded = []
    for z in a.get("dead_zone_details", [{"start": t0, "end": t1, "causes": []} for t0, t1 in a["dead_zones"]]):
        if ending_flagged and z["start"] >= ending_start - 1:
            folded.append(z)                                    # reported as part of "Ending fades"
            continue
        what = "Predicted brain engagement dips here — viewers are most likely to swipe away."
        if z.get("quote"):
            what += f' Said here: *"{z["quote"]}"*'
        if z.get("causes"):
            fix = _cause_text(z["causes"])
            fix = fix[0].upper() + fix[1:] + "."
        else:
            fix = "Audio and visuals stay lively here, so the *content* is losing interest: tighten the script, cut filler or add new information."
        issues.append(dict(start=z["start"], end=z["end"], icon="📉", title="Likely drop-off", what=what, fix=fix))
    if not a["dead_zones"]:
        good.append("📉 **No likely drop-off zones** — engagement holds across the video")

    if a["ending_vs_avg_pct"] <= -8:
        t0 = round(a["duration"] * 0.8, 1)
        what = f"The last 20% is {a['ending_vs_avg_pct']:+.0f}% vs the video's average."
        for z in folded:
            what += f" Lowest point: **{z['start']}s – {z['end']}s**."
        if a.get("ending_causes"):
            what += " At the end, " + _cause_text(a["ending_causes"]) + "."
        if a.get("ending_quote"):
            what += f' Last line: *"{a["ending_quote"]}"*'
        issues.append(dict(start=t0, end=a["duration"], icon="🏁", title="Ending fades", what=what,
                           fix="End on a payoff, punchline or reveal — or cut earlier, right after your last strong moment."))
    elif a["ending_vs_avg_pct"] >= 8:
        good.append(f"🏁 **Strong ending** ({a['ending_vs_avg_pct']:+.0f}% vs average) — good for rewatches and shares")

    ls = a.get("longest_shot")
    covered = [(i["start"], i["end"]) for i in issues]
    if ls and ls[1] - ls[0] >= 8 and not any(s0 < ls[1] and e0 > ls[0] for s0, e0 in covered):
        issues.append(dict(start=ls[0], end=ls[1], icon="✂️", title="Long shot without a cut",
                           what=f"{ls[1] - ls[0]:.0f}s in one shot. Engagement holds, but it is the riskiest stretch for pacing.",
                           fix="Add a cut, punch-in zoom or b-roll in the middle."))

    issues += a.get("frame_issues", []) + a.get("speech_issues", [])
    issues.sort(key=lambda i: (i["start"], i["end"]))

    lines = []
    if issues:
        lines.append("### 🛠️ What to improve — and where")
        for i in issues:
            lines.append(f"**⏱️ {i['start']:.1f}s – {i['end']:.1f}s · {i['icon']} {i['title']}**  ")
            lines.append(f"{i['what']}  ")
            lines.append(f"👉 **Fix:** {i['fix']}\n")
    else:
        lines.append("### 🛠️ What to improve\nNothing major found — this reel is in good shape. 🎉\n")
    if good:
        lines.append("### ✅ What's working")
        lines += [f"- {g}" for g in good]
    a["issues"] = issues
    return "\n".join(lines)


def make_plot(a):
    t = np.array(a["times"])
    fig, ax1 = plt.subplots(figsize=(10, 4.6), facecolor="#ffffff")
    ax1.axvspan(0, min(3.0, a["duration"]), color="#fde68a", alpha=0.5, label="Hook (0–3s)")
    for i, (t0, t1) in enumerate(a["dead_zones"]):
        ax1.axvspan(t0, t1, color="#fecaca", alpha=0.6, label="Likely drop-off zone" if i == 0 else None)
    ax1.plot(t, a["curve"], color="#2563eb", linewidth=2.5, marker="o", markersize=3, label="Brain response")
    ax1.axhline(50, color="#94a3b8", linestyle=":", linewidth=1.5, label="This video's average")
    for i, iss in enumerate(x for x in a.get("frame_issues", []) if x["icon"] in ("🖤", "⬛", "🧊")):
        ax1.axvspan(iss["start"], iss["end"], color="#334155", alpha=0.25, hatch="//",
                    label="Black / frozen frames" if i == 0 else None)
    if a.get("cuts"):
        ax1.plot(a["cuts"], [101.5] * len(a["cuts"]), linestyle="none", marker="v", markersize=7,
                 color="#7c3aed", label="Cut (scene change)")
    for p in a["peaks"]:
        ax1.annotate(f"peak {p['time']}s", (p["time"], p["score"]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#1e3a8a")
    if a.get("estimated_retention_shape"):
        ax2 = ax1.twinx()
        ax2.plot(t, a["estimated_retention_shape"], color="#ea580c", linestyle="--", linewidth=2,
                 label="Estimated retention (shape only)")
        ax2.set_ylim(0, 105)
        ax2.set_yticks([])
        ax2.set_ylabel("Estimated retention (shape only)", color="#ea580c")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Response (50 = avg)")
    ax1.set_title("MindMatrix — Predicted Brain Response Over Time (TRIBE v2)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    handles, labels = ax1.get_legend_handles_labels()
    for ax in fig.axes[1:]:
        h, l = ax.get_legend_handles_labels()
        handles, labels = handles + h, labels + l
    ax1.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    ax1.set_xlabel("Video time (seconds)")

    plot_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig.savefig(plot_file, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return plot_file


@spaces.GPU(duration=180)
def predict_cortical_activity(events):
    """Adaptive inference: runs persistent CUDA on dedicated GPU, or dynamically on ZeroGPU."""
    if is_zerogpu and torch.cuda.is_available():
        if hasattr(model, "_model") and model._model is not None:
            print("⚡ ZeroGPU allocated: Moving TRIBE v2 to CUDA...")
            model._model.to("cuda")

    try:
        preds, segments = model.predict(events, verbose=True)
        return np.array(preds), segments
    finally:
        if is_zerogpu and torch.cuda.is_available():
            if hasattr(model, "_model") and model._model is not None:
                print("💤 Returning model to CPU & freeing ZeroGPU VRAM...")
                model._model.to("cpu")
                torch.cuda.empty_cache()


def analyze_video_pipeline(video_path):
    if not video_path:
        return None, None, "⚠️ Please upload a video file.", None

    print(f"🎬 Processing video: {video_path}")
    
    # 1. Optimize video to 480p @ 16 fps (Fast & memory safe)
    opt_path = str(video_path).replace(".mp4", "_opt.mp4")
    run_path = video_path
    try:
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "scale=-2:'min(480,ih)'",
            "-r", "16",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "copy",
            opt_path
        ], capture_output=True, text=True)
        if r.returncode == 0 and Path(opt_path).exists():
            run_path = opt_path
    except Exception as e:
        print(f"FFmpeg note: {e}")

    try:
        # Phase 1: Audio extraction & temporal alignment
        print("🎧 Extracting audio and temporal word events...")
        events = model.get_events_dataframe(video_path=run_path)

        # Phase 2: Feature extractors + Brain forward pass
        print("🧠 Running cortical response prediction...")
        preds, segments = predict_cortical_activity(events)

        times = segment_times(segments, preds.shape[0])

        # Drop trailing segments that lie (mostly) past the real end of the video:
        # the stimulus cut-off there produces an artificial response spike.
        try:
            real_dur = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True).stdout.strip())
            step = float(np.median(np.diff(times))) if len(times) > 1 else SEG_FALLBACK_SEC
            keep = times + step / 2 <= real_dur
            if keep.sum() >= 2:
                preds, times = preds[keep], times[keep]
        except Exception as e:
            print(f"Duration trim note: {e}")

        a = analyze_brain_response(preds, times)
        a["estimated_retention_shape"] = estimate_retention(a["curve"], a["times"])

        # Explain dips with what is happening in the video at that moment
        try:
            loud, motion = video_cues(video_path, int(np.ceil(a["duration"])))
            a["cuts"] = [c for c in detect_cuts(video_path) if c < a["duration"]]
            shots = shots_from_cuts(a["cuts"], a["duration"])
            a["longest_shot"] = max(shots, key=lambda ab: ab[1] - ab[0]) if shots else None
            a["cuts_per_10s"] = round(10 * len(a["cuts"]) / max(a["duration"], 1e-6), 1)
            a["dead_zone_details"] = [{"start": t0, "end": t1, "causes": explain_span(t0, t1, loud, motion, shots) or []}
                                      for t0, t1 in a["dead_zones"]]
            a["hook_causes"] = explain_span(0, 3, loud, motion) or []
            a["ending_causes"] = explain_span(a["duration"] * 0.8, a["duration"], loud, motion, shots) or []
        except Exception as e:
            print(f"Video cue note: {e}")

        # What is being said (TRIBE's own transcription)
        words = extract_words(events)
        a["transcript"] = " ".join(w for _, _, w in words)
        a["hook_quote"] = words_between(words, 0, 3)
        a["ending_quote"] = words_between(words, a["duration"] * 0.8, a["duration"] + 1)
        for z in a.get("dead_zone_details", []):
            z["quote"] = words_between(words, z["start"], z["end"])

        # Black screens, frozen frames, dead air, silent opening
        try:
            if "loud" not in locals():
                loud, _ = video_cues(video_path, int(np.ceil(a["duration"])))
            a["frame_issues"] = find_frame_issues(video_path, a["duration"], loud)
            a["speech_issues"] = find_speech_issues(words, loud, a["duration"])
        except Exception as e:
            print(f"Gap check note: {e}")

        plot_file = make_plot(a)

        # Raw TRIBE output (for offline calibration against real performance data)
        raw_file = tempfile.NamedTemporaryFile(suffix=".npz", delete=False).name
        np.savez_compressed(raw_file, preds=preds.astype(np.float16), times=times)

        timeline_md = build_timeline(a)
        results = {"status": "success", "model": "facebook/tribev2", **a}
        json_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)

        peaks_txt = ", ".join(f"{p['time']}s" for p in a["peaks"]) or "—"
        dead_txt = ", ".join(f"{t0}–{t1}s" for t0, t1 in a["dead_zones"]) or "None 🎉"
        if a.get("cuts") is None:
            cuts_txt = "—"
        elif not a["cuts"]:
            cuts_txt = "No cuts — one continuous shot"
        else:
            ls = a["longest_shot"]
            cuts_txt = (f"{len(a['cuts'])} cuts ({a['cuts_per_10s']} per 10s) · longest shot {ls[0]}–{ls[1]}s"
                        f" · at {', '.join(f'{c:.1f}s' for c in a['cuts'][:12])}{' …' if len(a['cuts']) > 12 else ''}")
        summary_text = f"""
### 🧠 Brain Response Report
*Predicted with Meta's TRIBE v2 brain-encoding model. Scores compare each moment to **this video's own average** —
they show where the reel is strong or weak, not how many views it will get.*

| | | |
|---|---|---|
| **🎣 Hook (first 3s)** | {label(a['hook_vs_typical_pct'], good=10, bad=-10)} | {a['hook_vs_typical_pct']:+.1f}% vs a typical reel's hook |
| **🏁 Ending (last 20%)** | {label(a['ending_vs_avg_pct'])} | {a['ending_vs_avg_pct']:+.1f}% vs average |
| **📉 Likely drop-off zones** | {dead_label(a['dead_time_pct'])} | {a['dead_time_pct']}% of the video ({dead_txt}) |
| **✂️ Cuts** | | {cuts_txt} |
| **⚡ Peak moments** | | {peaks_txt} |
| **⏱️ Duration** | | {a['duration']}s |

{timeline_md}
""" + (f"""
<details><summary>📝 Transcript (auto, English model)</summary>

{a['transcript']}
</details>
""" if a.get("transcript") else "")
        return plot_file, json_file, summary_text, raw_file

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"❌ PIPELINE ERROR:\n{err_msg}")
        
        # Friendly instructions if LLaMA 3.2 permission is needed
        if "meta-llama/Llama-3.2-3B" in err_msg or "GatedRepoError" in err_msg or "401" in err_msg:
            auth_guide = """
### 🔑 Hugging Face Authentication Required for LLaMA 3.2
Meta's foundation model uses `meta-llama/Llama-3.2-3B` for text embeddings, which is a **gated repository** on Hugging Face.

**Quick 2-Step Fix (100% Free):**
1. **Accept License (Instant)**: Open **[meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)** and click **"Agree and access repository"**.
2. **Add Token to Space**:
   * Copy your User Access Token from **[Hugging Face Tokens](https://huggingface.co/settings/tokens)**.
   * Go to your Space **[Settings](https://huggingface.co/spaces/wlddhq/mindMatrix/settings)**.
   * Under **Variables and secrets**, add a secret named `HF_TOKEN` with your token value.
3. Re-submit your video!
"""
            return None, None, auth_guide, None

        return None, None, f"### ❌ Pipeline Error\n```python\n{err_msg}\n```", None

    finally:
        if run_path != video_path and Path(run_path).exists():
            try: Path(run_path).unlink()
            except: pass


# Gradio Web Interface
demo = gr.Interface(
    fn=analyze_video_pipeline,
    inputs=gr.Video(label="Upload Short-Form Video (.mp4 / .mov)"),
    outputs=[
        gr.Image(label="Predicted Brain Response Over Time"),
        gr.File(label="Download Full Metrics (.json)"),
        gr.Markdown(label="Brain Response Report & Suggestions"),
        gr.File(label="Raw TRIBE v2 Output (.npz)")
    ],
    title="🧠 MindMatrix — AI Human Brain Video Analyzer",
    description="Upload a reel to see how a typical viewer's brain is predicted to respond moment by moment (Meta TRIBE v2): hook strength, likely drop-off zones, peak moments and ending — with concrete edit suggestions."
)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
