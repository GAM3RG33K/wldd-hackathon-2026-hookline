# 2026-10-14 — Virality ablation: which channels earn their GPU time

- **Goal:** Answer "the facial-expression / tone / rhythm work is extra — does it contribute to the viral-potential score?" with the code trace and the stored reports, no new GPU runs.
- **Commands:** `read tribe/weights.py`, `tribe/fit_virality.py`, `tribe/emotion_fusion.py`, `tribe/hookline_report.py` (analyse/finalize/CANDIDATES), `hf_space/app.py` (STEPS/MODALITY_COST/EMOTION_ON/_run_job_sync); `python3` one-offs over `backend/out/*/report.json` + `samples/*/metadata.csv` (Cohen's d outliers-vs-low, Spearman vs log(views)).
- **Result:** Finding written into `.ai/research/scoring-heuristic.md` ("Ablation 2026-10-14").
- **Findings (evidence):**
  - Score path: `A = 100·sigmoid(1.2·zs(0.6N+0.4S))` (`hookline_report.py:284`); `N` = TRIBE preds only, `S` = ffmpeg cuts/motion/loudness + ASR words (`:171-202`). Virality base = hook / drop-free / strong moments / ending (`:399-414`).
  - Only `emotion_lift` (+0.5·strength/100·gap) and `caption_lift` (+0.15·relevance·gap) touch the base (`:514-521`, `:527+`). Song match, music rhythm and each emotion channel are **not** score inputs (music/emotion features are only consumed if `virality_model.json` exists — it does not).
  - 107 reports: emotion lift mean **+2.8**, median **0**, `strength>0` on 34/107; caption lift on 5/107.
  - 81 labelled reels (10 outliers, 71 lowest-view): lift moved 28 reels, **26 of them non-outliers**; d outliers-vs-low **+0.54 → +0.30**. Spearman vs log(views): virality +0.12, attention-only +0.13, emotional strength **−0.02**. FL041 low-view took +28.0 (matches the logged CLAP-on-music-bed failure at `.ai/logs/2026-10-10-tui-redesign-space-deps.md:310`).
  - Cost: emotion channels are missing from `_estimate()`, so the ETA under-promises; `HOOKLINE_EMOTION=0` already exists (`hf_space/app.py:82`).
- **Verdict:** The extra channels do **not** earn their place as score inputs. Score = TRIBE + ffmpeg signals + ASR words. Keep ASR (it feeds `words_ps` + hook bonus). Emotion / music / song stay as report insight only.
- **Next (needs the user's pick, not started):** A = zero `emotion_lift` in `tribe/virality_weights.json` + drop "+ strong-emotion lift" from the method string + label those sections as insight; B = keep the lift but only on trusted channels (needs a calibration run); C = leave as is. With A, verify with `python tribe/rescore.py` (re-scores all 107 reports locally), restart the backend, re-run the ablation to confirm d goes back to ≈0.54, and update `.ai/rules/01-business-logic.md` + this research file together (rule 5).

- **Closed 2026-10-14:** option A applied + verified — see `.ai/logs/2026-10-14-emotion-insight-only.md` (d outliers-vs-low +0.24 → +0.42 with the lift off).
