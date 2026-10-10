"""Render an editing suggestion as a real video (ffmpeg), so the model can check it.

Each edit becomes a list of segments of the original, concatenated, with an optional effect
per segment: trim, cold open (teaser first), end early, punch-in zoom, speed ramp, audio lift,
on-screen text. Simulations of creative edits are approximations (a zoom stands in for b-roll,
a gain boost stands in for a sound hit); the report says so.
"""
import json
import subprocess
from pathlib import Path

FONT = next((f for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if Path(f).exists()), None)
RENDERABLE = {"trim", "cold_open", "end", "zoom", "speed", "sound", "text"}
APPROXIMATE = {"zoom": "punch-in zoom stands in for b-roll / a new shot",
               "sound": "+6 dB lift stands in for a sound hit or music swell"}


def probe(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                        "-of", "json", str(video)], capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    return float(d["format"]["duration"]), int(v["width"]), int(v["height"]), \
        any(s["codec_type"] == "audio" for s in d["streams"])


def _esc(text):
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def segments(edit, p, T):
    """[(start, end, effect)] for an edit; effect is None or (kind, value)."""
    a, b = float(p.get("start", 0)), float(min(p.get("end", T), T))
    if edit == "baseline":  # re-encode only: isolates encoding effects from the edit itself
        return [(0, T, None)]
    if edit == "trim":
        return [(0, a, None), (b, T, None)]
    if edit == "cold_open":
        return [(a, b, None), (0, T, None)]
    if edit == "end":
        return [(0, float(p["end"]), None)]
    if edit == "zoom":
        return [(0, a, None), (a, b, ("zoom", p.get("zoom", 1.15))), (b, T, None)]
    if edit == "speed":
        return [(0, a, None), (a, b, ("speed", p.get("rate", 1.25))), (b, T, None)]
    if edit == "sound":
        return [(0, a, None), (a, b, ("gain", p.get("gain_db", 6))), (b, T, None)]
    if edit == "text":
        return [(0, a, None), (a, b, ("text", p.get("text", ""))), (b, T, None)]
    raise ValueError(f"cannot render edit {edit!r}")


def render(video, edit, params, out):
    T, W, H, has_audio = probe(video)
    segs = [(s, e, fx) for s, e, fx in segments(edit, params, T) if e - s > 0.05]
    parts, labels = [], []
    for i, (s, e, fx) in enumerate(segs):
        vf = f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS"
        af = f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS"
        if fx and fx[0] == "zoom":
            z = float(fx[1])
            vf += f",scale=trunc(iw*{z}/2)*2:trunc(ih*{z}/2)*2,crop={W}:{H}"
        elif fx and fx[0] == "speed":
            vf += f",setpts=PTS/{fx[1]}"
            af += f",atempo={fx[1]}"
        elif fx and fx[0] == "gain":
            af += f",volume={fx[1]}dB"
        elif fx and fx[0] == "text" and FONT:
            vf += (f",drawtext=fontfile='{FONT}':text='{_esc(fx[1])}':fontcolor=white:fontsize=h/22:"
                   f"box=1:boxcolor=black@0.55:boxborderw=18:x=(w-text_w)/2:y=h*0.72")
        parts.append(vf + f",setsar=1,format=yuv420p[v{i}]")
        labels.append(f"[v{i}]")
        if has_audio:
            parts.append(af + f"[a{i}]")
            labels.append(f"[a{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(segs)}:v=1:a={1 if has_audio else 0}[v]" + \
        ("[a]" if has_audio else "")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", graph, "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(out)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out
