# Scoring heuristic — current state + fit plan (source: `tribe/hookline_report.py`)

## Current (HYPOTHESIS, uncalibrated)
- `CFG`: `w_neural=0.6, w_signal=0.4; hook_seconds=3; hook_k=1.3; attn_k=1.2; drop_thresh=35, drop_min_len=2; scene_thresh=0.3; hold_floor=0.93, hold_span=0.065; silence_db=-45`.
- Attention `A = 100*sigmoid(1.2*zs(smooth(0.6N+0.4S)))`; hook bonuses `+0.3` speech ≤1.5s / `+0.3` cut-or-move ≤2s / `−0.4` silent >1s.
- Viral potential (`VIRALITY_WEIGHTS`): `0.40 hook + 0.25 drop-free share + 0.20 strong-attention share (×2, cap 100) + 0.15 ending`, then strong-emotion lift `+0.5*(strength/100)*(100-base)`. Tags `<35 Low reach, <55 Average, <70 Above average, <85 Popular, else Viral potential`.
- History: v1 `0.45 hook + 0.30 hold + 0.25 mean(A)` had a near-constant `mean(A)≈50` term and a length penalty via hold → replaced 2026-10-10.
- Explainability: `virality_breakdown` (pull vs typical = weight × (factor − 50)) + `virality_summary` (biggest boost / drag).

## Calibration status
- `tribe/fit_virality.py`: L2 logistic, known-viral outliers vs lowest-view reels; 20 reels → leave-one-out 65% (DRY RUN, not applied). Largest effect sizes: `net_visual −0.97, net_language +0.81, motion −0.76, peak_gap_cv −0.71, peaks/window +0.59`.
- Writing `tribe/virality_model.json` switches scoring to the fitted model (breakdown shows weight × standardised feature).
- Re-fit after the 2026-10-11 full re-run (45 reels with face channel + emotion features).

## Fit plan
- Hackathon reel set (`samples/`, strata face / faceless / outliers, views + follower band). Followers = strata only, never features.
- Per reel: full Space run → raw → features → fit within strata → report Spearman + top / bottom separation. Keep within-video normalisation; fit only weights / squashes / thresholds.
- On fit: tag `FITTED YYYY-MM-DD + data hash`, update `CFG` / weights + this file + `01-business-logic.md` together.
