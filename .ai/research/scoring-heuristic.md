# Scoring heuristic — current CFG + fit plan (source: `tribe/hookline_report.py:24-33,169-285`)

## Current (HYPOTHESIS, uncalibrated)
- `w_neural=0.6, w_signal=0.4; hook_seconds=3; hook_k=1.3; attn_k=1.2; drop_thresh=35, drop_min_len=2; scene_thresh=0.3; hold_floor=0.93, hold_span=0.065; silence_db=-45.0`.
- Attention `A = 100*sigmoid(1.2*zs(smooth(0.6N+0.4S)))`; hook bonuses `+0.3` speech≤1.5s / `+0.3` cut-or-move≤2s / `−0.4` silent>1s; hold `h = 0.93+0.065*sigmoid(1.5A_z)`; virality `0.45hook+0.30hold+0.25mean(A)`. Bands `<30 Bad/<50 Weak/<70 Average/<85 Strong/else Viral-grade`.

## Fit plan (judging axis 4)
- Need 110-reels CSV (PENDING — not in repo; only `samples/sneaker.mp4`). Strata: faceless / face / viral-ref. Handle+follower range = strata only, never features.
- Procedure: run full pipeline per reel → collect hook/hold/virality + moments → correlate vs view counts within strata → grid-search `w_neural, hook_k, attn_k, hold_floor/span, drop_thresh` → report Spearman + top/bottom-quartile separation. Keep within-video normalisation; fit only squashes/thresholds.
- On fit: tag `FITTED YYYY-MM-DD + data hash`, update CFG + this file + `01-business-logic.md` in same change.
