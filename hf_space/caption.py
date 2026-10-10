"""Optional creator caption: does it describe THIS reel, and what does it add? (CPU, from raw Space cues)

Two independent tests, each a z-score; the stronger one decides:
  frames      caption vs this reel's frames (mean of the best 30%) against the same caption vs ~55 other
              reels (robust z: median / MAD). A true caption fits its own reel better than other reels;
              a generic one ("follow for more #viral", "lorem ipsum") fits every reel about equally.
  transcript  caption vs the transcript (multilingual embeddings) against ~25 unrelated control captions.
Relevance ramps from 0 at z 1.6 to 1 at z 3.0 (thresholds set on true / generic / borrowed / unrelated
captions of two reels, see .ai/research/emotion.md). Irrelevant captions are ignored: no effect at all.
A relevant caption adds context: its emotion joins the emotion mix (weighted by relevance) and it lifts
viral potential by up to CAPTION_LIFT of the remaining gap. Uncalibrated, like every other weight.
"""
import numpy as np

import taxonomy as TX

Z_LOW, Z_HIGH = 1.6, 3.0
CAPTION_LIFT = 0.15        # relevance 1.0 closes 15% of the gap to 100
CAPTION_EMO_WEIGHT = 0.5   # reel-wide channel weight in emotion fusion, times relevance


def _robust_z(x, ref):
    ref = np.asarray([r for r in ref if r is not None], dtype=float)
    if len(ref) < 5:
        return None
    med = np.median(ref)
    mad = 1.4826 * np.median(np.abs(ref - med)) or (ref.std() or 1e-3)
    return float((x - med) / mad)


def assess(extra):
    """Caption block for the report, or None when no caption was given."""
    extra = extra or {}
    text = (extra.get("caption") or "").strip()
    if not text:
        return None
    cue = ((extra.get("emotion") or {}).get("caption")) or {}
    block = {"text": text, "relevance": 0.0, "verdict": "not checked", "evidence": {}, "emotion": None}
    if not cue:
        block["verdict"] = "not checked: the Space returned no caption signals (old run or emotion step off)"
        return block
    ev = {}
    frames = [x for x in cue.get("frame_sim") or [] if x is not None]
    if frames and cue.get("gallery_sim"):
        k = max(1, int(0.3 * len(frames)))
        this = float(np.mean(np.sort(frames)[-k:]))
        z = _robust_z(this, cue["gallery_sim"])
        if z is not None:
            ev["frames_z"] = round(z, 2)
            ev["frames_beats_other_reels"] = round(float(np.mean(np.asarray(cue["gallery_sim"]) < this)), 2)
    if cue.get("transcript_sim") is not None and cue.get("transcript_control_sim"):
        ctrl = np.asarray(cue["transcript_control_sim"], dtype=float)
        ev["transcript_z"] = round(float((cue["transcript_sim"] - ctrl.mean()) / (ctrl.std() or 1e-3)), 2)
        ev["transcript_beats_controls"] = round(float(np.mean(ctrl < cue["transcript_sim"])), 2)
    zs = [v for k, v in ev.items() if k.endswith("_z")]
    best = max(zs) if zs else None
    rel = 0.0 if best is None else float(np.clip((best - Z_LOW) / (Z_HIGH - Z_LOW), 0, 1))
    block["relevance"] = round(rel, 2)
    block["evidence"] = ev
    if best is None:
        block["verdict"] = "not checked: no frames or transcript to compare with"
    elif rel >= 0.5:
        block["verdict"] = "matches the video"
    elif rel > 0:
        block["verdict"] = "partly related"
    else:
        block["verdict"] = "ignored: does not describe this video"
    # what the caption says it feels like (GoEmotions labels -> taxonomy), only if it is relevant
    probs = cue.get("emotion") or {}
    if probs and rel > 0:
        dist = np.zeros(len(TX.NAMES))
        for go, p in probs.items():
            for e, w in TX.GO_MAP.get(go, {}).items():
                dist[TX.NAMES.index(e)] += w * p
        if dist.sum() > 0:
            dist = dist / dist.sum()
            top = [(TX.NAMES[i], round(float(dist[i]), 2)) for i in np.argsort(-dist)[:4]
                   if dist[i] >= 0.1 and TX.NAMES[i] != "calm"][:3]  # calm = no emotional context
            block["emotion"] = {"dist": [round(float(x), 3) for x in dist], "top": top}
    return block


def emotion_channel(block, n):
    """(n, len(EMOTIONS)) reel-wide distribution scaled by relevance, or None (feeds emotion fusion)."""
    if not block or not block.get("emotion") or block["relevance"] <= 0:
        return None
    d = np.asarray(block["emotion"]["dist"], dtype=float) * block["relevance"]
    d[TX.NAMES.index("calm")] = 0.0  # a caption adds emotional context; a neutral caption must not mute the reel
    return np.tile(d, (n, 1))


def lift(block, virality):
    """Points a relevant caption adds to viral potential (0 when ignored)."""
    if not block or block["relevance"] <= 0:
        return 0.0
    return float(CAPTION_LIFT * block["relevance"] * (100.0 - virality))
