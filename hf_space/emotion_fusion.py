"""Fuse per-second emotion cues into one evoked-emotion track (CPU, from report + raw cues).

Predicted emotion the video EVOKES for an average viewer (stimulus-driven), not a measured mood.
Taxonomy: tribe/taxonomy.py (20 categories after Cowen & Keltner 2017; appraisal-based prompts).
Psychology behind the features:
  - arousal matters more than valence for sharing (Berger & Milkman 2012)
  - memory of an experience follows its peak and its end (Kahneman, peak-end rule)
  - surprise / interest early opens an information gap (Loewenstein) = emotional hook
  - stories have arcs (Reagan et al. 2016); a negative start that resolves upward moves people
"""
from collections import Counter

import numpy as np

import taxonomy as TX

EMOTIONS = TX.NAMES
AROUSAL = {e: TX.EMOTIONS[e][0] for e in EMOTIONS}
VALENCE = {e: TX.EMOTIONS[e][1] for e in EMOTIONS}
CHANNEL_WEIGHT = {"visual": 1.0, "text": 1.0, "sounds": 0.8, "music": 0.9, "face": 1.0}
FACE_SMOOTH_S = 3  # face expression per frame jitters (happy 0.99 -> sad 0.97 a second later): 3s mean
ZS_SCALE = {"visual": 40.0, "music": 40.0}  # softmax sharpness over raw similarities (no calibration yet)
Z_SCALE = {"visual": 1.5, "music": 1.5}      # softmax sharpness over calibrated z-scores
MIN_CAL_REELS = 15  # below this the reference set is too thin: one long reel would define "usual"


def _calibration():
    import json
    from pathlib import Path
    p = Path(__file__).with_name("prompt_calibration.json")
    try:
        cal = json.loads(p.read_text())
        return cal if cal.get("reels", 0) >= MIN_CAL_REELS else None
    except (OSError, ValueError):
        return None


CAL = _calibration()


def _calibrated(S, prompts, channel):
    """Raw similarities -> per-prompt z-scores ('higher than usual for this prompt') when calibrated."""
    c = (CAL or {}).get(channel) or {}
    if not c or not all(p in c for p in prompts):
        return S * ZS_SCALE[channel]
    mu = np.array([c[p][0] for p in prompts])
    sd = np.array([c[p][1] for p in prompts])
    return (S - mu) / sd * Z_SCALE[channel]
VISUAL_TEMPER = 0.5                          # for legacy probability cues only
VISUAL_STRONG_WEIGHT = 0.5                   # visual alone (scene) can't make a strong peak
STRONG_LEVEL, STRONG_PROMINENCE, STRONG_DROP, STRONG_FULL = 0.6, 0.15, 0.15, 0.30
# In a speech reel the music is a bed under the talk: it colours the mood but does not carry the
# emotion, so alone it must not make a strong peak (a quirky bed under a product review is not laughter).
MUSIC_BED_WEIGHT = 0.5
# GoEmotions is English-only; on Hinglish / Hindi transcripts its labels are unreliable.
TEXT_NON_ENGLISH_WEIGHT = 0.5
# legacy cue names (8-emotion version) -> taxonomy
LEGACY = {"tension": "fear", "outrage": "anger", "curiosity": "interest", "warmth": "moved", "neutral": "calm"}
LEGACY_MUSIC = {"sad": {"sadness": 1.0}, "uplifting": {"hope": 0.7, "moved": 0.3}, "epic": {"awe": 1.0},
                "tense": {"fear": 1.0}, "happy": {"amusement": 0.5, "excitement": 0.5}, "funny": {"amusement": 1.0},
                "calm": {"calm": 1.0}, "none": {"calm": 1.0}}


def _seconds(x, n, default=None):
    x = [] if x is None else list(x)
    return (x + [default] * n)[:n]


def _softmax(S, scale=1.0):
    S = S * scale
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)


def visual_dist(cue, n):
    """(n, len(EMOTIONS)) probabilities from CLIP cues (per-prompt sims, emotion sims, or legacy probs)."""
    if not cue:
        return None
    D = np.zeros((n, len(EMOTIONS)))
    if "prompt_sim" in cue:
        prompts = [p for p in TX.VISUAL_PROMPTS if p in cue["prompt_sim"]]
        S = np.array([_seconds(cue["prompt_sim"][p], n, 0.0) for p in prompts], dtype=float).T
        P = _softmax(_calibrated(S, prompts, "visual"))
        for j, p in enumerate(prompts):
            D[:, EMOTIONS.index(TX.VISUAL_PROMPTS[p])] += P[:, j]
        return D
    src = cue.get("sim", cue)
    keys = list(src)
    S = np.array([_seconds(src.get(k), n, 0.0) for k in keys], dtype=float).T
    P = _softmax(S, ZS_SCALE["visual"]) if "sim" in cue else S ** VISUAL_TEMPER
    for j, k in enumerate(keys):
        D[:, EMOTIONS.index(LEGACY.get(k, k))] += P[:, j]
    return D / np.maximum(D.sum(1, keepdims=True), 1e-9)


def music_dist(cue, n):
    """(n, len(EMOTIONS)) from CLAP cues, plus the per-second top music prompt (for the mood line)."""
    if not cue:
        return None, None
    D = np.zeros((n, len(EMOTIONS)))
    if "prompt_sim" in cue:
        prompts = [p for p in TX.MUSIC_PROMPTS if p in cue["prompt_sim"]]
        table = TX.MUSIC_PROMPTS
        S = np.array([_seconds(cue["prompt_sim"][p], n, 0.0) for p in prompts], dtype=float).T
        P = _softmax(_calibrated(S, prompts, "music"))
    else:
        src = cue.get("sim", cue)
        prompts = list(src)
        table = LEGACY_MUSIC
        S = np.array([_seconds(src.get(k), n, 0.0) for k in prompts], dtype=float).T
        P = _softmax(S, ZS_SCALE["music"]) if "sim" in cue else S
    for j, p in enumerate(prompts):
        for e, w in table.get(p, {}).items():
            D[:, EMOTIONS.index(e)] += w * P[:, j]
    top = [prompts[i] for i in P.argmax(1)] if len(P) else []
    return D, top


def text_dist(text, n):
    D = np.zeros((n, len(EMOTIONS)))
    seen = np.zeros(n, dtype=bool)
    for t, d in enumerate(_seconds(text, n)):
        if not d:
            continue
        seen[t] = True
        tc = (CAL or {}).get("text") or {}
        if tc:  # calibrated: only labels firing above their usual level count (removes label bias)
            d = {k: max(0.0, (v - tc[k][0]) / tc[k][1]) / 3.0 if k in tc else v for k, v in d.items()}
        for go, p in d.items():
            for e, w in TX.GO_MAP.get(go, {}).items():
                D[t, EMOTIONS.index(e)] += w * p
    return D if seen.any() else None


def face_dist(cue, n):
    """(n, len(EMOTIONS)) from per-frame expression probabilities, smoothed over FACE_SMOOTH_S seconds
    (mean over the frames that have a face; seconds with no face nearby stay empty)."""
    probs = _seconds((cue or {}).get("probs"), n)
    if not any(probs):
        return None
    D = np.full((n, len(EMOTIONS)), np.nan)
    for t, d in enumerate(probs):
        if d:
            row = np.zeros(len(EMOTIONS))
            for lab, p in d.items():
                for e, w in TX.FACE_MAP.get(lab, {}).items():
                    row[EMOTIONS.index(e)] += w * p
            D[t] = row
    h = FACE_SMOOTH_S // 2
    S = np.zeros((n, len(EMOTIONS)))
    for t in range(n):
        win = D[max(0, t - h):t + h + 1]
        ok = ~np.isnan(win[:, 0])
        if ok.sum() >= 2 or (ok.sum() == 1 and not np.isnan(D[t, 0])):
            S[t] = win[ok].mean(0)
    return S if S.any() else None


def sound_dist(sounds, n):
    D = np.zeros((n, len(EMOTIONS)))
    for k, s in (sounds or {}).items():
        e = TX.SOUND_MAP.get(k)
        if e:
            x = np.array(_seconds(s, n, 0.0), dtype=float)
            D[:, EMOTIONS.index(e)] += np.where(x > 0.3, x, 0.0)
    return D if D.any() else None


def channel_trust(extra):
    """Per-channel weights from what the reel is: music is a bed in speech reels, and the
    transcript emotion model is only trusted on plain-English transcripts."""
    extra = extra or {}
    gate = extra.get("speech_gate") or {}
    info = extra.get("transcript_info") or {}
    english = info.get("model") not in (None, "hinglish") and info.get("lang") == "en" \
        or extra.get("asr") == "whisper" and extra.get("lang") == "en"
    return {"music": MUSIC_BED_WEIGHT if gate.get("audio_type") in ("speech", "speech_over_music") else 1.0,
            "text": 1.0 if english else TEXT_NON_ENGLISH_WEIGHT}


NO_MUSIC_RATIO = 0.1  # speech gate: below this share of music / singing / bed seconds the reel has no music


def has_music(extra):
    """False when the speech gate heard (almost) no music: CLAP still ranks its mood prompts on plain speech
    and would invent a 'comedic' or 'mournful' score. True when unknown (no gate)."""
    r = ((extra or {}).get("speech_gate") or {}).get("ratios")
    if not r:
        return True
    return max(r.get("music", 0), r.get("vocals", 0), r.get("bed", 0), r.get("humming", 0)) >= NO_MUSIC_RATIO


def channels(extra, n):
    cues = (extra or {}).get("emotion") or {}
    gate = (extra or {}).get("speech_gate") or {}
    mus, mus_top = music_dist(cues.get("music"), n) if has_music(extra) else (None, None)
    ch = {"visual": visual_dist(cues.get("visual"), n), "music": mus,
          "text": text_dist((cues.get("text") or {}).get("probs"), n), "sounds": sound_dist(gate.get("sounds_per_second"), n),
          "face": face_dist(cues.get("face"), n)}
    voice = cues.get("voice") or {}
    va = np.array([x if x is not None else np.nan for x in _seconds(voice.get("arousal"), n)], dtype=float)
    vv = np.array([x if x is not None else np.nan for x in _seconds(voice.get("valence"), n)], dtype=float)
    return ch, mus_top, va, vv


def fuse(report, extra):
    """Return the `emotion` report block plus emotion features; (None, {}) if there are no cues."""
    n = len(report.get("series", {}).get("attention") or [])
    if n == 0:
        return None, {}
    ch, mus_top, va, vv = channels(extra, n)
    trust = channel_trust(extra)
    used = [k for k, v in ch.items() if v is not None]
    if np.isfinite(va).any():
        used.append("voice")
    if not used:
        return None, {}
    calm = EMOTIONS.index("calm")
    dist = np.zeros((n, len(EMOTIONS)))
    for k, D in ch.items():
        if D is not None:
            dist += CHANNEL_WEIGHT[k] * trust.get(k, 1.0) * D
    probs = dist / np.maximum(dist.sum(1, keepdims=True), 1e-9)
    intensity = 1.0 - probs[:, calm]                         # how emotional at all (vs calm baseline)
    emo = probs.copy()
    emo[:, calm] = 0
    emo = emo / np.maximum(emo.sum(1, keepdims=True), 1e-9)  # mix among the non-calm emotions

    a_vec = np.array([AROUSAL[e] for e in EMOTIONS])
    v_vec = np.array([VALENCE[e] for e in EMOTIONS])
    a_emo, v_emo = probs @ a_vec, probs @ v_vec
    loud = np.array(_seconds(report["series"].get("loudness_db"), n, -45.0), dtype=float)
    a_acoustic = np.clip((loud + 45.0) / 40.0, 0, 1)
    arousal = np.where(np.isfinite(va), (2 * a_emo + 0.5 * a_acoustic + np.nan_to_num(va)) / 3.5,
                       (2 * a_emo + 0.5 * a_acoustic) / 2.5)
    valence = np.where(np.isfinite(vv), (2 * v_emo + (np.nan_to_num(vv) * 2 - 1)) / 3, v_emo)
    arousal, valence = np.clip(arousal, 0, 1), np.clip(valence, -1, 1)
    dominant = [EMOTIONS[int(np.argmax(emo[t]))] if intensity[t] > 0.35 else "calm" for t in range(n)]

    sal = (report["series"].get("network_z") or {}).get("salience")
    brain_r = None
    if sal and len(sal) == n and np.std(arousal) > 1e-6 and np.std(sal) > 1e-6:
        brain_r = round(float(np.corrcoef(arousal, sal)[0, 1]), 3)

    strong = strong_emotions(ch, va, n, trust)
    # audible laughter (AST: laughter / giggle / chuckle in the audio track), separate from predicted amusement
    laugh = _seconds(((extra or {}).get("speech_gate") or {}).get("sounds_per_second", {}).get("amusement"), n, 0.0)
    strong["audible_laughter_s"] = int(sum(1 for x in laugh if (x or 0) > 0.3))
    feats = features(arousal, valence, emo, intensity, dominant, report)
    seq = []
    for d in dominant:
        if d != "calm" and (not seq or seq[-1] != d):
            seq.append(d)
    block = {
        "note": "predicted emotion the video evokes for an average viewer; not a measured mood",
        "taxonomy": "20 categories after Cowen & Keltner 2017 (emotions evoked by short videos)",
        "channels": used, "channel_trust": trust, "brain_salience_r": brain_r,
        "series": {"arousal": [round(float(x), 3) for x in arousal], "valence": [round(float(x), 3) for x in valence],
                   "dominant": dominant, "intensity": [round(float(x), 3) for x in intensity],
                   "probs": {e: [round(float(x), 3) for x in emo[:, j]] for j, e in enumerate(EMOTIONS) if e != "calm"}},
        "summary": {"dominant_emotions": _top(dominant), "sequence": seq[:8], **feats,
                    "peak_moments": [{"at_s": m["peak_at"], "emotion": m["emotion"],
                                      "arousal": round(float(arousal[m["peak_at"]]), 2)} for m in strong["moments"][:5]]},
        "strong": strong,
    }
    if mus_top:
        c = Counter(mus_top)
        order = []
        for m in mus_top:
            if not order or order[-1] != m:
                order.append(m)
        block["music_mood"] = {"shares": {k: round(v / n, 2) for k, v in c.most_common(4)}, "sequence": order[:8],
                               "per_second": mus_top}
    out = {f"emo_{k}": v for k, v in feats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out.update(emo_strong_share=strong["share"], emo_strength=strong["score"], emo_resolves=float(feats["resolves"]))
    for e in EMOTIONS:  # share of seconds each emotion dominates (fit candidates)
        if e != "calm":
            out[f"emo_share_{e}"] = round(dominant.count(e) / n, 3)
    return block, out


def strong_emotions(ch, voice_arousal, n, trust=None):
    """Peaks of intense emotions. Evidence per emotion = max over channels (visual at half weight,
    so a steady scene tone never qualifies); a lively voice backs high-arousal emotions in speech;
    'moved' also fires at a turn from suffering to hope / kindness. Only peaks that spike count."""
    from scipy.signal import find_peaks
    va = np.nan_to_num(voice_arousal)
    trust = trust or {}
    weight = lambda k: (VISUAL_STRONG_WEIGHT if k == "visual" else 1.0) * trust.get(k, 1.0)
    evidence = {}
    for e in TX.STRONG:
        j = EMOTIONS.index(e)
        parts = [np.zeros(n)]
        for k, D in ch.items():
            if D is not None:
                parts.append(weight(k) * np.clip(D[:, j], 0, 1))
        if ch.get("text") is not None and AROUSAL[e] >= 0.65:
            parts.append(trust.get("text", 1.0) * (0.6 * np.clip(ch["text"][:, j], 0, 1) + 0.4 * va))
        evidence[e] = np.clip(np.max(parts, axis=0), 0, 1)
    # 'moved' at a turn: suffering in the last 12s resolving into hope / kindness in the next 4s
    neg = np.max([evidence["sadness"], evidence["empathic_pain"]], axis=0)
    pos = np.zeros(n)
    for k, D in ch.items():
        if D is not None:
            pos = np.maximum(pos, weight(k) * np.clip(D[:, EMOTIONS.index("hope")] + D[:, EMOTIONS.index("moved")], 0, 1))
    turn = np.array([min(neg[max(0, t - 12):t + 1].max(), pos[t:min(n, t + 5)].max()) * 1.4 for t in range(n)])
    evidence["moved"] = np.clip(np.maximum(evidence["moved"], turn), 0, 1)

    moments = []
    for e, x in evidence.items():
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
            moments.append({"emotion": TX.STRONG_LABEL.get(e, e), "start": a, "end": b + 1, "peak_at": int(p),
                            "level": round(float(x[p]), 2)})
    seconds = sorted({t for m in moments for t in range(m["start"], m["end"])})
    labels = [TX.STRONG_LABEL.get(e, e) for e in TX.STRONG]
    share = len(seconds) / max(n, 1)
    return {"moments": sorted(moments, key=lambda m: m["start"]),
            "durations_s": {l: sum(m["end"] - m["start"] for m in moments if m["emotion"] == l) for l in labels},
            "seconds": len(seconds), "share": round(share, 3),
            "score": round(100 * min(1.0, share / STRONG_FULL), 1),
            "max_evidence": {TX.STRONG_LABEL.get(e, e): round(float(v.max()), 2) for e, v in evidence.items()},
            "definition": "predicted peaks of amusement, awe, outrage, sorrow, being moved, fear, disgust, "
                          "empathic pain (evoked in the viewer, not sounds in the video)"}


def features(arousal, valence, emo, intensity, dominant, report):
    import hookline_report as HL
    n = len(arousal)
    hs = min(3, n)
    early_epistemic = float((emo[:hs, EMOTIONS.index("surprise")] + emo[:hs, EMOTIONS.index("interest")]).mean()
                            * intensity[:hs].mean())
    pk = HL.peak_features([100 * a for a in arousal])
    att = np.array(report["series"]["attention"], dtype=float)
    r = float(np.corrcoef(arousal, att)[0, 1]) if np.std(arousal) > 1e-6 and np.std(att) > 1e-6 else 0.0
    thirds = [float(valence[i * n // 3:(i + 1) * n // 3].mean()) if n >= 3 else 0.0 for i in range(3)]
    third = max(1, n // 3)
    opening = Counter(d for d in dominant[:third] if d != "calm").most_common(1)
    ending = Counter(d for d in dominant[-third:] if d != "calm").most_common(1)
    return {
        "hook": round(float(arousal[:hs].max() + early_epistemic), 3),
        "peak": round(float((arousal * np.abs(valence)).max()), 3),
        "high_arousal_share": round(float((arousal > 0.6).mean()), 3),
        "intensity_mean": round(float(intensity.mean()), 3),
        "peaks_per_window": pk["peaks_per_window"], "peak_gap_cv": pk["peak_gap_cv"],
        "peak_end": round(float(0.5 * arousal.max() + 0.5 * arousal[-3:].mean()), 3),
        "valence_mean": round(float(valence.mean()), 3),
        "valence_range": round(float(valence.max() - valence.min()), 3),
        "attention_coupling": round(r, 3),
        "arc": _arc(thirds), "opening": opening[0][0] if opening else "calm",
        "ending": ending[0][0] if ending else "calm",
        "resolves": bool(thirds[0] < 0 and thirds[2] > thirds[0] + 0.15),  # negative start that lifts by the end
    }


def _arc(thirds):
    a, b, c = thirds
    if max(thirds) - min(thirds) < 0.1:
        return "flat"
    if c > a and b <= c:
        return "rise"
    if c < a and b >= c:
        return "fall"
    return "fall-rise (man in a hole)" if b < min(a, c) else "rise-fall"


def _top(dominant):
    c = Counter(d for d in dominant if d != "calm")
    total = max(1, len(dominant))
    return [{"emotion": e, "share": round(k / total, 2)} for e, k in c.most_common(3)]
