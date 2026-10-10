#!/usr/bin/env python
"""Per-prompt calibration for the zero-shot emotion channels.

Some prompts match almost anything ("a funny prank or blooper" scores highest on every reel),
which biases the emotion mix. Calibration turns each raw similarity into "how much higher than
usual for this prompt": z = (sim - corpus mean) / corpus std, per prompt, from every stored raw
output. Same for the transcript emotion labels. Rebuild after adding reels:

  python tribe/calibrate_prompts.py      # writes tribe/prompt_calibration.json, then re-score
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "backend" / "out"
CAL = Path(__file__).resolve().with_name("prompt_calibration.json")


def collect():
    """Per prompt, one mean per reel (so every reel weighs the same, however long it is)."""
    vis, mus, txt, reels = {}, {}, {}, 0
    for f in OUT.glob("*/raw/meta.json"):
        em = (json.loads(f.read_text()).get("space") or {}).get("emotion") or {}
        got = False
        for name, store in (("visual", vis), ("music", mus)):
            ps = (em.get(name) or {}).get("prompt_sim")
            if ps:
                got = True
                for p, xs in ps.items():
                    xs = [x for x in xs if x is not None]
                    if xs:
                        store.setdefault(p, []).append(xs)
        per_label = {}
        for d in (em.get("text") or {}).get("probs") or []:
            for k, v in (d or {}).items():
                per_label.setdefault(k, []).append(v)
        for k, xs in per_label.items():
            txt.setdefault(k, []).append(xs)
        reels += got
    return vis, mus, txt, reels


def _stats(store):
    """[mean of per-reel means, pooled std of seconds] per prompt."""
    out = {}
    for p, per_reel in store.items():
        if len(per_reel) < 2:
            continue
        mu = float(np.mean([np.mean(x) for x in per_reel]))
        sd = float(np.std(np.concatenate(per_reel))) or 1e-3
        out[p] = [round(mu, 5), round(sd, 5)]
    return out


def main():
    vis, mus, txt, reels = collect()
    cal = {"reels": reels, "visual": _stats(vis), "music": _stats(mus), "text": _stats(txt),
           "note": "per-prompt [mean, std] of raw similarity (text: label probability) across the corpus"}
    CAL.write_text(json.dumps(cal, indent=1))
    print(f"calibration from {reels} reels: {len(cal['visual'])} visual, {len(cal['music'])} music, "
          f"{len(cal['text'])} text prompts -> {CAL.name}")


if __name__ == "__main__":
    main()
