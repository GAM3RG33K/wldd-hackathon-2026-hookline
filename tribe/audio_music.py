"""Rhythm and musical emotion of a reel's audio (signal processing, CPU, ~1-2s per reel).

Per second: energy (dB), onset density (events/s), brightness (spectral centroid), tempo (BPM),
mode (major minus minor key fit: >0 brighter/happier, <0 darker/sadder), harmonic change.
Events: build-ups (energy rising >= 6 dB over >= 3s), drops (>= 6 dB fall within 1s), silences.
Editing: beat-cut sync = share of scene cuts landing within 120 ms of a beat.
Mood labels per second come from CLAP on the Space (emotion.py); this module is the rhythm/energy side.
"""
import subprocess

import numpy as np

SR = 22050
HOP = 512
# Krumhansl-Kessler key profiles (major / minor), for a mode (valence) proxy
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _load(video, sr=SR):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def _per_second(x, frames_per_s, T, how=np.mean):
    out = np.zeros(T)
    for t in range(T):
        seg = x[int(t * frames_per_s):int((t + 1) * frames_per_s)]
        out[t] = how(seg) if len(seg) else (out[t - 1] if t else 0.0)
    return out


def _mode_score(chroma_seg):
    """Best major-key fit minus best minor-key fit over all 12 keys (Pearson r)."""
    c = chroma_seg.mean(1)
    if c.std() < 1e-6:
        return 0.0
    rs = lambda prof: max(np.corrcoef(c, np.roll(prof, k))[0, 1] for k in range(12))
    return float(rs(MAJOR) - rs(MINOR))


def analyse_music(video, T, cuts=()):
    import librosa
    y = _load(video)
    if len(y) < SR // 2 or float(np.abs(y).max()) < 1e-4:
        return None
    fps = SR / HOP
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    energy_db = _per_second(20 * np.log10(rms + 1e-6), fps, T)
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP, units="time")
    onset_density = np.array([((onsets >= t) & (onsets < t + 1)).sum() for t in range(T)], dtype=float)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0]
    brightness = _per_second(centroid, fps, T) / (SR / 2)  # 0..1 of Nyquist
    tempo_global, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP, units="time")
    tempo_global = float(np.atleast_1d(tempo_global)[0])
    # octave correction: beat trackers double slow music; sparse onsets at a "fast" tempo = half time
    dens = len(onsets) / max(T, 1)
    halved = bool(tempo_global > 95 and dens < 1.5 * tempo_global / 60 / 2)
    if halved:
        tempo_global /= 2
    # local tempo from beat intervals (BPM per second, held between beats)
    bpm = np.full(T, tempo_global)
    if len(beats) > 2:
        ibi = np.diff(beats)
        for t in range(T):
            near = ibi[(beats[1:] >= t - 2) & (beats[1:] < t + 3)]
            if len(near):
                bpm[t] = 60.0 / np.median(near) / (2 if halved else 1)
    beat_strength = _per_second(onset_env, fps, T, how=np.max)
    beat_strength = beat_strength / (beat_strength.max() + 1e-9)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    w = int(3 * fps)  # 3s windows for key fit
    mode = np.array([_mode_score(chroma[:, max(0, int(t * fps) - w // 2):int(t * fps) + w // 2 + 1]) for t in range(T)])
    chroma_s = np.stack([_per_second(chroma[k], fps, T) for k in range(12)])
    harm_change = np.r_[0.0, np.linalg.norm(np.diff(chroma_s, axis=1), axis=0)]

    # structure: build-ups, drops, silences (on smoothed per-second energy)
    e = np.convolve(energy_db, np.ones(3) / 3, mode="same")
    buildups, drops, silences = [], [], []
    t = 0
    while t < T - 3:
        end = t
        while end + 1 < T and e[end + 1] >= e[end] - 0.5:
            end += 1
        if end - t >= 3 and e[end] - e[t] >= 6:
            buildups.append({"start": int(t), "end": int(end), "rise_db": round(float(e[end] - e[t]), 1)})
            t = end
        t += 1
    for t in range(1, T):
        if energy_db[t - 1] - energy_db[t] >= 6:
            drops.append({"at": int(t), "fall_db": round(float(energy_db[t - 1] - energy_db[t]), 1)})
    quiet = energy_db < -45
    t = 0
    while t < T:
        if quiet[t]:
            s = t
            while t < T and quiet[t]:
                t += 1
            if t - s >= 1:
                silences.append({"start": int(s), "end": int(t)})
        t += 1
    sync = None
    if len(cuts) and len(beats):
        sync = round(float(np.mean([np.min(np.abs(beats - c)) <= 0.12 for c in cuts])), 3)

    r = lambda x, d=2: [round(float(v), d) for v in x]
    series = {"energy_db": r(energy_db, 1), "onset_density": r(onset_density, 1), "brightness": r(brightness, 3),
              "tempo_bpm": r(bpm, 1), "beat_strength": r(beat_strength), "mode": r(mode, 3),
              "harmonic_change": r(harm_change, 3)}
    summary = {"tempo_bpm": round(tempo_global, 1), "tempo_halved": halved, "beats": len(beats),
               "beats_per_s": round(len(beats) / max(T, 1), 2),
               "mode_mean": round(float(mode.mean()), 3),
               # key fit is a weak hint (unreliable on modal, melody-led music): only strong fits are labelled
               "mode_label": "major/brighter" if mode.mean() > 0.2 else "minor/darker" if mode.mean() < -0.2 else "key unclear",
               "mode_swing": round(float(mode.max() - mode.min()), 3),
               "energy_range_db": round(float(np.percentile(energy_db, 95) - np.percentile(energy_db, 5)), 1),
               "onset_density_mean": round(float(onset_density.mean()), 2),
               "buildups": buildups[:8], "drops": drops[:8], "silences": silences[:8], "beat_cut_sync": sync}
    features = {"music_tempo": summary["tempo_bpm"], "music_mode": summary["mode_mean"],
                "music_mode_swing": summary["mode_swing"], "music_energy_range": summary["energy_range_db"],
                "music_onset_density": summary["onset_density_mean"], "music_buildups": len(buildups),
                "music_drops": len(drops), "beat_cut_sync": sync if sync is not None else 0.0}
    return {"series": series, "summary": summary, "beat_times": r(beats, 2)}, features
