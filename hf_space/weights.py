"""Viral-potential weights, editable from any report page (POST /weights) and stored in
tribe/virality_weights.json. Read at call time, so a change applies without restarting anything.

  hook, drop_free, strong_moments, ending   attention factors: must add up to exactly 1
  emotion_lift                              0 by default: emotion is report insight, not a score
                                             input (2026-10-14 ablation cut outlier-vs-low separation
                                             from +0.54 to +0.30, `.ai/research/scoring-heuristic.md`).
                                             A non-zero value turns the lift + breakdown row back on.
  caption                                   share of the gap to 100 that a fully relevant caption closes
Uncalibrated: set by hand until fitted on real view counts (tribe/fit_virality.py).
"""
import json
from pathlib import Path

FILE = Path(__file__).with_name("virality_weights.json")
DEFAULTS = {"hook": 0.40, "drop_free": 0.25, "strong_moments": 0.20, "ending": 0.15, "emotion_lift": 0.0, "caption": 0.15}
BASE_KEYS = ("hook", "drop_free", "strong_moments", "ending")
LIMITS = (0.0, 1.0)


def load():
    w = dict(DEFAULTS)
    try:
        w.update({k: float(v) for k, v in json.loads(FILE.read_text()).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return w


def validate(new):
    """Clean user input: known keys only, numbers within LIMITS, at least one attention factor above 0."""
    out = load()
    for k, v in (new or {}).items():
        if k not in DEFAULTS:
            raise ValueError(f"unknown weight {k!r}")
        v = float(v)
        if not LIMITS[0] <= v <= LIMITS[1]:
            raise ValueError(f"{k} must be between {LIMITS[0]} and {LIMITS[1]}")
        out[k] = round(v, 3)
    total = sum(out[k] for k in BASE_KEYS)
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"the attention weights (hook, few drop-offs, strong moments, ending) must add up to 1; "
                         f"they add up to {total:.2f}")
    return out


def save(new):
    w = validate(new)
    FILE.write_text(json.dumps(w, indent=1))
    return w


def reset():
    FILE.unlink(missing_ok=True)
    return dict(DEFAULTS)
