# 01 — Business Logic (Hookline / TRIBE v2 heuristic)

Source of truth: `tribe/tribe_analyze.py` + `tribe/hookline_report.py`. Colab mirrors them (`tribe/tribe_colab.ipynb` cells `%%writefile`).

## Pipeline (order fixed)
1. **Ingest:** video file → optional `ffmpeg -t --max-seconds` trim → `out/<stem>/`.
2. **Events:** `TribeModel.get_events_dataframe(video_path)` → audio features + word-level transcript (`.tsv` next to video; faster-whisper in-process patch, `TRIBE_WHISPER_MODEL=small` default; whisperx fallback `uvx --python 3.12`, `int8` on CPU).
3. **Predict:** `model.predict(events)` → `preds (T,20484)` avg-subject, fsaverage5 mesh, −5s hemodynamic lag; `times` zero-based.
4. **Neural index:** Destrieux groups → `visual, auditory, language, attention` mean minus `0.5 * default_mode`; z-score → `N`. Atlas fail → fallback `cortex = ||preds||`, warn to stderr. Never crash on missing nilearn/network.
5. **Signal index:** per-second `S = zs(mean(zs(smooth(cuts)), zs(motion), zs(max(dB,-45)), zs(smooth(words_ps))))`. Cuts: `ffmpeg scene_thresh 0.3`; motion: `fps=5,scale=64:64,gray` mean abs-diff; audio: 16k mono RMS→dB, onset = first |a|>0.01; words: counts/sec from `.tsv`, empty → zeros.
6. **Attention:** `A_z = zs(smooth(0.6*N + 0.4*S))`, `A = 100*sigmoid(1.2*A_z)`. Grid = `ceil(duration)` seconds, interp neural onto grid.
7. **Hook (first 3s):** `raw = 0.5*mean(head) + 0.5*max(head) + bonus`; bonus `+0.3` speech ≤1.5s else note `no spoken line…`; `+0.3` cut ≤2s or motion[0:2] > 1.3×median else `static opening…`; `−0.4` audio onset missing/>1s + `silent start…`. `hook = 100*sigmoid(1.3*raw)`.
8. **Hold/survival:** `h = 0.93 + 0.065*sigmoid(1.5*A_z)`; first 3s linear `1 → hook/100` ramp; after `surv[t] = surv[t-1]*h[t]`; `finish = surv[-1]*100`, `hold = finish/hook*100` capped 100.
9. **Virality:** `0.45*hook + 0.30*min(hold,100) + 0.25*mean(A)`. Bands: `<30 Bad, <50 Weak, <70 Average, <85 Strong, else Viral-grade`.

## Moments + recommendations (rule table, keep)
- `peak_hook = argmax(rise 2s)+1`, `peak_attention = argmax(A)`, strongest/weakest 3s rolling windows.
- Drops: `A < 35` for `≥2s`; if none, weakest 3s window. Why-tags: `no visual change` (cuts=0), `low movement` (<0.8×median), `flat audio` (dB std<2), `no speech` (words=0). Fix template: `Re-hook: [cut/zoom/b-roll] [sound hit/lift] tease/question.`
- Buried peak: `peak_t > 5` and `hook < 60` → `opening on it / tease in first 3s`.
- Weak ending: `T > 6` and `mean(A[-3:]) < 40` → `payoff or loop`.
- Opening notes → fixed fixes: no-line → `claim/question/payoff in first second`; static → `cut/zoom/movement or best frame`; silent → `sound on frame 0`.

## Data models (artifacts per `out/<stem>/`)
- `preds.npy (T,20484) float`, `times.npy (T,)`, `meta.json {video, source, transcript|null, n_steps, seconds_total}`.
- `report.json {video, duration_s, method, scores{hook_pct,hook_band,hold_rate_pct,finish_pct,avg_attention,virality,virality_band}, moments{peak_hook_at_s,peak_attention_at_s,strongest_window_s,weakest_window_s,drop_zones_s}, network_mean_response, recommendations[{at,issue,fix}], series{attention,survival_pct,cuts_s}}` + `report.md` + `report.png`.

## Journeys
- **Creator:** upload (FastAPI `POST /analyze` → Colab job) → poll `GET /report/{id}` → read md/png + json series → edit at timestamps → re-upload.
- **Validation (judging axis 4):** gather reels with known outcomes (110 reels CSV is PENDING — not in repo; only `samples/sneaker.mp4`) → fit `CFG` weights vs view counts within strata (faceless/face/viral-ref); handle/followers = strata only.

## PENDING_IMPLEMENTATION
- `videos` CSV + 110 reels corpus; fitted (calibrated) weights; FastAPI↔Colab job bridge; auth (none — local prototype, Colab `HF_TOKEN` secret only).
