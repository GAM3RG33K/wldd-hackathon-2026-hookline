# Scoring heuristic — current state + fit plan (source: `tribe/hookline_report.py`)

## Current (HYPOTHESIS, uncalibrated)
- `CFG`: `w_neural=0.6, w_signal=0.4; hook_seconds=3; hook_k=1.3; attn_k=1.2; drop_thresh=35, drop_min_len=2; scene_thresh=0.3; hold_floor=0.93, hold_span=0.065; silence_db=-45`.
- Attention `A = 100*sigmoid(1.2*zs(smooth(0.6N+0.4S)))`; hook bonuses `+0.3` speech ≤1.5s / `+0.3` cut-or-move ≤2s / `−0.4` silent >1s.
- Viral potential: attention base = weighted average of hook / drop-free share / strong-attention share (×2, cap 100) / ending, default weights 0.40 / 0.25 / 0.20 / 0.15 (must sum to 1, editable from any report page, `tribe/weights.py`), then strong-emotion lift `+0.5*(strength/100)*(100-v)` and caption lift `+0.15*relevance*(100-v)`. Tags `<35 Low reach, <55 Average, <70 Above average, <85 Popular, else Viral potential`.
- History: v1 `0.45 hook + 0.30 hold + 0.25 mean(A)` had a near-constant `mean(A)≈50` term and a length penalty via hold → replaced 2026-10-10.
- Explainability: `virality_breakdown` (pull vs typical = weight × (factor − 50)) + `virality_summary` (biggest boost / drag).

## Ablation 2026-10-14 — what actually moves the viral-potential score (measured on stored reports)
Question: "the emotion / music / song / face channels are extra work — do they contribute to the score?" Trace + measurement, no new GPU runs.

**Code trace — what feeds the number**
- Attention `A = 100·sigmoid(1.2·zs(0.6·N + 0.4·S))` (`hookline_report.py:284`), where `N` = TRIBE preds (`:261-268`) and `S` = z-scored mix of ffmpeg cuts / motion / loudness / ASR words (`:171-202`, `:278-281`). Hold + survival from `A` (`:307-316`).
- Virality base = weighted mean of hook / drop-free / strong moments / ending (`:399-414`, `weights.py:13-14`). Everything else in the report is prose.
- Only two lifts touch the base: `emotion_lift` (+0.5·strength/100·(100−v), `:514-521`) and the caption lift (+0.15·relevance·(100−v), `:527+`).
- **Channels that feed nothing in the score today:** song match ("context only, not a score input", `finalize()`), music rhythm/build-ups (`audio_music` → `features`, consumed only when `virality_model.json` exists — it does not), and each emotion channel individually (CLIP scenes, YuNet+ViT faces, CLAP music, voice arousal/valence, GoEmotions text) — those only matter through the fused `strength`. The speech gate per-second arrays feed the report and the emotion fusion, not attention.

**Measured impact of the lifts (107 stored reports)**
- Emotion lift: mean **+2.8** points on a 0–100 scale, median **0.0**, max +28.0. `emotional_strength > 0` on only 34/107 reels (32%); mean strength 12.7/100. Caption lift applied on 5/107 (max +15.1).
- On the 81 labelled full-TRIBE reels (10 outliers vs 71 lowest-view), the lift is counter-productive: a lift moved 28 reels, **26 of them non-outliers**; outliers-vs-low separation (Cohen's d) falls from **+0.54 (attention only) to +0.30 (with lifts)**. FL041 (low-view, sorrow) took +28.0 → 68.6.
- Spearman vs `log(views)` over the same 81 reels: virality as shown **+0.12**, attention-only base **+0.13**, `emotional_strength` **−0.02**, strong-emotion share **−0.02**, `hook` +0.15, `net_visual` −0.19, `engagement_raw` −0.04, `motion` −0.10, `cuts_per_s` −0.10, `peak_share` −0.05. Nothing is predictive at this n (consistent with `PENDING_FIT`).
- Known failure case behind this: CLAP labelled a music bed "comedic quirky funny music" → 34 s laughter, strength 100, viral 68 on a low-view reel (`.ai/logs/2026-10-10-tui-redesign-space-deps.md:310`).

**Cost**
- `MODALITY_COST = video 8.0 / text 2.5 / audio 0.3` s per video-second, `TRANSCRIBE_COST = 3.5` (`hf_space/app.py:56-58`). The emotion channels are **not in `_estimate()` at all** — they run inside the "events" step (`hf_space/app.py:1019-1037`), so the ETA under-promises and the demo waits past the estimate. `HOOKLINE_EMOTION=0` already disables them (`hf_space/app.py:82`).

**Conclusion (HYPOTHESIS, small n but same sign as the logged failure case):** the score should be computed from TRIBE preds + ffmpeg signals + ASR words only. Emotion/music/song are *report insight*, not score inputs, until a fit proves otherwise. Keep ASR: it feeds `words_ps` and the hook bonus, so it is not extra.

**Applied 2026-10-14 (option A):** `emotion_lift` default → 0.0 in `tribe/weights.py` / `tribe/virality_weights.json`, lift row + `+ strong-emotion lift` suffix made conditional in `hookline_report.py`, emotion labelled "insight only, not a score input" on web + TUI. Verified by same-code ablation on the re-scored set: 0 non-virality diffs, 32 reels lose the lift row, d outliers-vs-low **+0.24 (lift 0.4) → +0.42 (lift 0.0)**; `POST /weights` with `emotion_lift > 0` still re-enables it (`.ai/logs/2026-10-14-emotion-insight-only.md`). Re-apply this verdict after any fit: a channel only returns to the score if it improves held-out separation.

## Calibration status
- `tribe/fit_virality.py`: L2 logistic, known-viral outliers vs lowest-view reels. **2026-10-14: the old "LOO 65% on 20 reels" leaked** — top-k features were selected on the full set before cross-validating. Selection now runs inside each fold; on 81 reels the corrected LOO is 86%, *below* the 88% trivial baseline. Accuracy was never the right metric here. Use `tribe/validate_virality.py` (AUC + permutation p + per-channel ablation on held-out folds) — its result: heuristic AUC +0.62, 95% CI [+0.43, +0.80], **p=0.135, not distinguishable from chance**; refitting does not help and inverts the rank correlation. Full numbers: `.ai/research/validation.md`.
- Writing `tribe/virality_model.json` switches scoring to the fitted model (breakdown shows weight × standardised feature).
- Re-fit after the 2026-10-11 full re-run (45 reels with face channel + emotion features).

## Fit plan
- Hackathon reel set (`samples/`, strata face / faceless / outliers, views + follower band). Followers = strata only, never features.
- Per reel: full Space run → raw → features → fit within strata → report Spearman + top / bottom separation. Keep within-video normalisation; fit only weights / squashes / thresholds.
- On fit: tag `FITTED YYYY-MM-DD + data hash`, update `CFG` / weights + this file + `01-business-logic.md` together.
- Next fit must include an **ablation table** (per channel: effect d outliers-vs-low, Spearman vs log(views), GPU seconds): a channel only stays a score input if it improves separation on held-out reels. Template for that ablation = the 2026-10-14 run above.
