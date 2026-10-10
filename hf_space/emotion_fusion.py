"""Fuse per-second emotion cues into one evoked-emotion track (CPU, from report + raw cues).

Predicted emotion the video EVOKES for an average viewer (stimulus-driven), not a measured mood.
Psychology behind the features:
  - arousal matters more than valence for sharing (Berger & Milkman 2012)
  - memory of an experience follows its peak and its end (Kahneman, peak-end rule)
  - surprise / curiosity early opens an information gap (Loewenstein) = emotional hook
  - emotional contrast (peaks, shifts) sustains attention; arcs classify the story shape
"""
import numpy as np

EMOTIONS = ["amusement", "awe", "surprise", "curiosity", "tension", "outrage", "warmth", "sadness"]
# circumplex coordinates (arousal 0..1, valence -1..1) per emotion
AROUSAL = {"amusement": .70, "awe": .80, "surprise": .90, "curiosity": .55, "tension": .85,
           "outrage": .90, "warmth": .35, "sadness": .25}
VALENCE = {"amusement": .80, "awe": .70, "surprise": .20, "curiosity": .20, "tension": -.60,
           "outrage": -.80, "warmth": .80, "sadness": -.70}
HIGH_AROUSAL = {"amusement", "awe", "surprise", "tension", "outrage"}  # Berger-Milkman sharing drivers
# GoEmotions (transcript) -> our 8
GO_MAP = {
    "amusement": {"amusement": 1.0, "joy": 0.5},
    "awe": {"admiration": 0.7, "excitement": 0.7},
    "surprise": {"surprise": 1.0, "realization": 0.5},
    "curiosity": {"curiosity": 1.0, "confusion": 0.4},
    "tension": {"fear": 1.0, "nervousness": 1.0},
    "outrage": {"anger": 1.0, "annoyance": 0.6, "disgust": 0.8, "disapproval": 0.5},
    "warmth": {"love": 1.0, "caring": 1.0, "gratitude": 0.8, "joy": 0.5, "admiration": 0.3},
    "sadness": {"sadness": 1.0, "grief": 1.0, "remorse": 0.7, "disappointment": 0.7},
}
CHANNEL_WEIGHT = {"visual": 1.0, "text": 1.0, "sounds": 0.8, "music": 0.9}
# CLAP music mood -> our emotions ("calm"/"none" carry no emotion)
MUSIC_MAP = {"sad": {"sadness": 1.0}, "uplifting": {"warmth": 0.5, "awe": 0.5}, "epic": {"awe": 1.0},
             "tense": {"tension": 1.0}, "happy": {"amusement": 0.5, "warmth": 0.5}, "funny": {"amusement": 1.0}}
VISUAL_TEMPER = 0.5  # flattens CLIP's sharp softmax so one prompt can't take every second


def _seconds(x, n, default=None):
    x = list(x or [])
    return (x + [default] * n)[:n]


def fuse(report, extra):
    """Return the `emotion` report block plus emotion features; None if no cues were produced."""
    cues = (extra or {}).get("emotion") or {}
    gate = (extra or {}).get("speech_gate") or {}
    n = len(report.get("series", {}).get("attention") or [])
    if n == 0 or not (cues or gate.get("sounds_per_second")):
        return None, {}
    dist = np.zeros((n, len(EMOTIONS)))
    weight = np.zeros(n)
    used = []

    vis = cues.get("visual") or {}
    if vis:
        used.append("visual")
        # CLIP's softmax (logit scale 100) is winner-take-all; temper it (p^0.5, renormalised)
        keys = EMOTIONS + ["neutral"]
        P = np.array([_seconds(vis.get(k), n, 0.0) for k in keys], dtype=float).T ** VISUAL_TEMPER
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
        dist += CHANNEL_WEIGHT["visual"] * P[:, :len(EMOTIONS)]
        weight += CHANNEL_WEIGHT["visual"] * (1 - P[:, -1])

    text = (cues.get("text") or {}).get("probs") or []
    if any(text):
        used.append("text")
        for t, d in enumerate(_seconds(text, n)):
            if not d:
                continue
            row = np.array([sum(d.get(k, 0) * w for k, w in GO_MAP[e].items()) for e in EMOTIONS])
            mass = min(1.0, row.sum())
            if row.sum() > 0:
                dist[t] += CHANNEL_WEIGHT["text"] * mass * row / row.sum()
                weight[t] += CHANNEL_WEIGHT["text"] * mass

    music = cues.get("music") or {}
    if music:
        used.append("music")
        keys = list(music)
        M = np.array([_seconds(music.get(k), n, 0.0) for k in keys], dtype=float).T ** VISUAL_TEMPER
        M = M / np.maximum(M.sum(1, keepdims=True), 1e-9)
        for j, k in enumerate(keys):
            for e, w in MUSIC_MAP.get(k, {}).items():
                dist[:, EMOTIONS.index(e)] += CHANNEL_WEIGHT["music"] * w * M[:, j]
                weight += CHANNEL_WEIGHT["music"] * w * M[:, j]

    sounds = gate.get("sounds_per_second") or {}
    if sounds:
        used.append("sounds")
        for e, series in sounds.items():
            s = np.array(_seconds(series, n, 0.0), dtype=float)
            s = np.where(s > 0.3, s, 0.0)  # only confident sound events
            dist[:, EMOTIONS.index(e)] += CHANNEL_WEIGHT["sounds"] * s
            weight += CHANNEL_WEIGHT["sounds"] * s

    probs = dist / np.maximum(dist.sum(1, keepdims=True), 1e-9)
    intensity = np.clip(weight / max(1e-9, sum(CHANNEL_WEIGHT[c] for c in used)), 0, 1)  # how emotional at all

    # arousal: emotion-implied + voice (speech) + acoustic loudness; valence: emotion-implied + voice
    a_emo = probs @ np.array([AROUSAL[e] for e in EMOTIONS]) * intensity
    v_emo = probs @ np.array([VALENCE[e] for e in EMOTIONS]) * intensity
    loud = np.array(_seconds(report["series"].get("loudness_db"), n, -45.0), dtype=float)
    a_acoustic = np.clip((loud + 45.0) / 40.0, 0, 1)  # -45 dB -> 0, -5 dB -> 1
    voice = cues.get("voice") or {}
    va = np.array([x if x is not None else np.nan for x in _seconds(voice.get("arousal"), n)], dtype=float)
    vv = np.array([x if x is not None else np.nan for x in _seconds(voice.get("valence"), n)], dtype=float)
    if np.isfinite(va).any():
        used.append("voice")
    parts_a = [a_emo, 0.6 * a_acoustic]
    arousal = np.where(np.isfinite(va), (a_emo + 0.6 * a_acoustic + np.nan_to_num(va)) / 2.6, sum(parts_a) / 1.6)
    valence = np.where(np.isfinite(vv), (v_emo + (np.nan_to_num(vv) * 2 - 1)) / 2, v_emo)
    arousal = np.clip(arousal, 0, 1)
    valence = np.clip(valence, -1, 1)
    dominant = [EMOTIONS[int(np.argmax(probs[t]))] if intensity[t] > 0.15 else "neutral" for t in range(n)]

    # brain cross-check: does predicted arousal move with TRIBE's cortical salience response?
    sal = (report["series"].get("network_z") or {}).get("salience")
    brain_r = None
    if sal and len(sal) == n and np.std(arousal) > 1e-6 and np.std(sal) > 1e-6:
        brain_r = round(float(np.corrcoef(arousal, sal)[0, 1]), 3)

    feats = features(arousal, valence, probs, intensity, report)
    seq, last = [], None
    for d in dominant:
        if d != "neutral" and d != last:
            seq.append(d)
            last = d
    block = {
        "note": "predicted emotion the video evokes for an average viewer; not a measured mood",
        "channels": used, "brain_salience_r": brain_r,
        "series": {"arousal": [round(float(x), 3) for x in arousal], "valence": [round(float(x), 3) for x in valence],
                   "dominant": dominant,
                   "probs": {e: [round(float(x), 3) for x in probs[:, j]] for j, e in enumerate(EMOTIONS)}},
        "summary": {"dominant_emotions": _top(dominant), "sequence": seq[:6], **feats,
                    "peak_moments": _peaks(arousal, dominant)},
    }
    strong = strong_emotions(extra, n)
    block["strong"] = strong
    music = cues.get("music") or {}
    if music:  # music mood mix + its order over time (e.g. sad -> uplifting)
        keys = [k for k in music if k not in ("none",)]
        M = np.array([_seconds(music.get(k), n, 0.0) for k in keys], dtype=float).T
        top = [keys[i] for i in M.argmax(1)]
        from collections import Counter
        c = Counter(top)
        seq = []
        for m in top:
            if not seq or seq[-1] != m:
                seq.append(m)
        block["music_mood"] = {"shares": {k: round(v / n, 2) for k, v in c.most_common(4)}, "sequence": seq[:8],
                               "per_second": top}
    feats_out = {f"emo_{k}": v for k, v in feats.items() if isinstance(v, (int, float))}
    feats_out.update(emo_strong_share=strong["share"], emo_strength=strong["score"])
    return block, feats_out


# Strong emotions = intense PEAKS of awe, laughter, outrage/anger, sorrow (not tone, not mild feelings).
STRONG = {"laughter": "amusement", "awe": "awe", "outrage": "outrage", "sorrow": "sadness", "moved": None}
MOVED_LOOKBACK, MOVED_LOOKAHEAD = 12, 4  # sadness in the last 12s turning to uplift in the next 4s
STRONG_LEVEL = 0.6       # absolute evidence needed at the peak (0..1)
STRONG_PROMINENCE = 0.15  # must spike above its surroundings: a flat plateau (steady tone) never counts
STRONG_DROP = 0.15        # a peak lasts while evidence stays within this of the peak
STRONG_FULL = 0.30        # strong peaks covering 30% of the reel = emotional strength 100
VISUAL_STRONG_WEIGHT = 0.5  # visual alone (mostly scene tone) can't reach STRONG_LEVEL


def strong_emotions(extra, n):
    """Per strong emotion, an evidence curve from the channels that fit it; peaks -> moments."""
    from scipy.signal import find_peaks
    em = (extra or {}).get("emotion") or {}
    snd = ((extra or {}).get("speech_gate") or {}).get("sounds_per_second") or {}
    vis = em.get("visual") or {}
    text = (em.get("text") or {}).get("probs") or []

    def ser(x, d=0.0):
        return np.array([(d if v is None else v) for v in (list(x or []) + [d] * n)[:n]], dtype=float)

    mus = em.get("music") or {}
    txt = {e: np.clip(np.array([sum((d or {}).get(k, 0) * w for k, w in GO_MAP[e].items()) if d else 0.0
                                for d in (list(text) + [None] * n)[:n]]), 0, 1) for e in EMOTIONS}
    voice = ser((em.get("voice") or {}).get("arousal"))
    evidence = {}
    for s, base in STRONG.items():
        if base is None:
            continue
        parts = [VISUAL_STRONG_WEIGHT * ser(vis.get(base)), txt[base]]
        if s == "laughter":
            parts += [ser(snd.get("amusement")), 0.6 * txt[base] + 0.4 * voice,   # laughter sounds; funny line, lively voice
                      0.8 * ser(mus.get("funny"))]                                 # comedic music
        elif s == "awe":
            parts += [ser(snd.get("awe")), 0.8 * ser(mus.get("epic"))]           # cheering / applause; epic music
        elif s == "outrage":
            parts += [0.6 * txt[base] + 0.4 * voice]                               # angry words, raised voice
        elif s == "sorrow":
            parts += [ser(snd.get("sadness")), 0.8 * ser(mus.get("sad"))]        # crying, sighs; sad music
        evidence[s] = np.clip(np.max(parts, axis=0), 0, 1)
    # "moved" (kama muta): sadness turning into uplift - rescues, reunions, kindness
    sad = np.max([0.8 * ser(mus.get("sad")), VISUAL_STRONG_WEIGHT * ser(vis.get("sadness")), txt["sadness"]], axis=0)
    up = np.max([0.8 * ser(mus.get("uplifting")), 0.6 * ser(mus.get("epic")),
                 VISUAL_STRONG_WEIGHT * ser(vis.get("warmth")), txt["warmth"]], axis=0)
    moved = np.zeros(n)
    for t in range(n):
        before = sad[max(0, t - MOVED_LOOKBACK):t + 1].max()
        after = up[t:min(n, t + MOVED_LOOKAHEAD + 1)].max()
        moved[t] = min(before, after) * 1.4  # both halves must be present; 1.4 maps a clear turn (~0.45) to ~0.63
    build = np.zeros(n)
    for b in ((extra or {}).get("music_summary") or {}).get("buildups", []):
        build[b["start"]:b["end"] + 1] = 0.1  # an energy build-up landing on the turn strengthens it
    evidence["moved"] = np.clip(moved + build * (moved > 0), 0, 1)
    moments = []
    for s, x in evidence.items():
        idx, _ = find_peaks(x, prominence=STRONG_PROMINENCE)
        for p in idx:
            if x[p] < STRONG_LEVEL:
                continue
            thr = max(STRONG_LEVEL, x[p] - STRONG_DROP)
            a = b = int(p)
            while a > 0 and x[a - 1] >= thr:
                a -= 1
            while b < n - 1 and x[b + 1] >= thr:
                b += 1
            moments.append({"emotion": s, "start": a, "end": b + 1, "peak_at": int(p), "level": round(float(x[p]), 2)})
    seconds = sorted({t for m in moments for t in range(m["start"], m["end"])})
    durations = {s: sum(m["end"] - m["start"] for m in moments if m["emotion"] == s) for s in STRONG}
    share = len(seconds) / max(n, 1)
    return {"moments": sorted(moments, key=lambda m: m["start"]), "durations_s": durations,
            "seconds": len(seconds), "share": round(share, 3),
            "score": round(100 * min(1.0, share / STRONG_FULL), 1),
            "max_evidence": {s: round(float(v.max()), 2) for s, v in evidence.items()},
            "definition": "peaks of laughter, awe, outrage/anger, sorrow, being moved; steady tone does not count"}


def features(arousal, valence, probs, intensity, report):
    import hookline_report as HL
    n = len(arousal)
    hs = min(3, n)
    early_epistemic = float((probs[:hs, EMOTIONS.index("surprise")] + probs[:hs, EMOTIONS.index("curiosity")]).mean()
                            * intensity[:hs].mean())
    pk = HL.peak_features([100 * a for a in arousal])
    att = np.array(report["series"]["attention"], dtype=float)
    r = float(np.corrcoef(arousal, att)[0, 1]) if np.std(arousal) > 1e-6 and np.std(att) > 1e-6 else 0.0
    hi = [EMOTIONS.index(e) for e in HIGH_AROUSAL]
    thirds = [valence[i * n // 3:(i + 1) * n // 3].mean() if n >= 3 else 0.0 for i in range(3)]
    return {
        "hook": round(float(arousal[:hs].max() + early_epistemic), 3),             # arousal + surprise/curiosity in 0-3s
        "peak": round(float((arousal * np.abs(valence)).max()), 3),                 # strongest felt moment
        "high_arousal_share": round(float(((arousal > 0.6) & (probs[:, hi].sum(1) > 0.5)).mean()), 3),
        "peaks_per_window": pk["peaks_per_window"], "peak_gap_cv": pk["peak_gap_cv"],
        "peak_end": round(float(0.5 * arousal.max() + 0.5 * arousal[-3:].mean()), 3),
        "valence_mean": round(float(valence.mean()), 3),
        "attention_coupling": round(r, 3),                                          # emotion moves with attention
        "arc": _arc(thirds),
    }


def _arc(thirds):
    a, b, c = thirds
    if max(thirds) - min(thirds) < 0.1:
        return "flat"
    if c > a and b <= c:
        return "rise"
    if c < a and b >= c:
        return "fall"
    return "valley (fall-rise)" if b < min(a, c) else "peak (rise-fall)"


def _top(dominant):
    from collections import Counter
    c = Counter(d for d in dominant if d != "neutral")
    total = max(1, len(dominant))
    return [{"emotion": e, "share": round(k / total, 2)} for e, k in c.most_common(3)]


def _peaks(arousal, dominant):
    import hookline_report as HL
    idx = HL.attention_peaks([100 * a for a in arousal])
    return [{"at_s": int(i), "emotion": dominant[i], "arousal": round(float(arousal[i]), 2)} for i in idx[:5]]
