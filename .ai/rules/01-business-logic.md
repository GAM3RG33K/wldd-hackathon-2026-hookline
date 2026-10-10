# 01 — Business Logic (Hookline scoring contract)

Source of truth: `tribe/hookline_report.py` (attention, scores, virality), `tribe/emotion_fusion.py` + `tribe/taxonomy.py` (emotion), `tribe/audio_music.py` (music), `tribe/suggest.py` (editing suggestions). `hf_space/` holds copies for the Space's fallback report; reports shown to users are always built locally from the raw bundle (`tribe/rescore.py`).

## Pipeline (order fixed)
1. **Space (GPU):** upload → speech gate (AST, per-second speech / vocals / music / humming / laughter; auto mode turns the transcript off for songs and humming) → song match (Chromaprint vs `trending_audio.json`, context only) → Hinglish ASR (faster-whisper `Oriserve/Whisper-Hindi2Hinglish-Apex`, hallucination guards, transcribe before chunking) → TRIBE v2 events + `predict` in 10s chunks, fp16 → `preds (T,20484)` → emotion channels (CLIP scene prompts, YuNet + ViT faces, CLAP music, audeering voice, GoEmotions text). Raw bundle: `preds` (float16), `times`, `words.tsv`, `meta.json` (`space`: modalities, speech_gate, transcript, audio_match, emotion).
2. **Neural index:** Destrieux groups → mean of `visual, auditory, language, attention` minus `0.5 * default_mode`, z-scored → `N`. `salience` group kept as an arousal cross-check only. Atlas failure → whole-cortex magnitude, warning, never crash.
3. **Signal index (local video):** `S = zs(mean(zs(smooth(cuts)), zs(motion), zs(max(dB,-45)), zs(smooth(words_ps))))`; words dropped when the transcript is off (unknown, not absent).
4. **Attention:** `A_z = zs(smooth(0.6*N + 0.4*S))`, `A = 100*sigmoid(1.2*A_z)`, one value per second.
5. **Hook (first 3s):** `raw = 0.5*mean + 0.5*max + bonus` (`+0.3` speech ≤1.5s, `+0.3` cut or strong motion ≤2s, `−0.4` silent start); `hook = 100*sigmoid(1.3*raw)`.
6. **Hold / finish:** per-second hold `0.93 + 0.065*sigmoid(1.5*A_z)` after a linear hook ramp; `finish = surv[-1]*100`, `hold = finish/hook*100` (cap 100).
7. **Moments:** peak hook, peak attention, strongest / weakest 3s windows, drop zones (`A < 35` for ≥2s), peaks per window (window = 10% of length, max 10s; scipy `find_peaks` with prominence).
8. **Emotion:** see below.
9. **Viral potential (interim, uncalibrated):** `0.40*hook + 0.25*(1-drop_share)*100 + 0.20*min(100, 200*peak_share) + 0.15*ending` (`peak_share` = seconds at A ≥ 70; `ending` = mean of last 3s). Strong-emotion lift: `v = base + 0.5*(strength/100)*(100-base)`. Tags: `<35 Low reach, <55 Average, <70 Above average, <85 Popular, else Viral potential`. `virality_breakdown[]` = per factor `{value, weight, points, vs_typical = weight*(value-50)}` + `Emotional peaks` (lift points); `virality_summary` names the biggest boost and drag. A fitted model (`tribe/virality_model.json`, from `fit_virality.py`) replaces the formula only when present — currently absent (dry-run LOO 65% on 20 reels).
9b. **Caption (optional):** `caption.assess` → `relevance = clip((max(frames_z, transcript_z) − 1.6) / 1.4, 0, 1)`; frames_z = robust z of the caption's top-30% frame similarity vs ~55 gallery reels; transcript_z = caption–transcript cosine vs 25 `CAPTION_CONTROLS`. Relevant → reel-wide caption emotion channel (weight 0.5 × relevance, never a strong peak) + `lift = 0.15 × relevance × (100 − v)` (`Caption context` in the breakdown). Irrelevant → shown, no effect.
10. **Suggestions:** each drop zone vs the reel's strongest window (cuts, motion, loudness, speech, networks), one renderable edit (`trim, cold_open, end, zoom, speed, sound, text`), ranked by estimated viewers lost, max 5, end cuts ≤25%. Model check (`V`): render with ffmpeg, run on the Space, compare against a re-encode null edit.

## Emotion (predicted, evoked for an average viewer — not a measured mood)
- 20 categories after Cowen & Keltner 2017 (`taxonomy.EMOTIONS`, arousal / valence each). Prompts follow appraisal theory (the situation, not the look).
- Channels → per-second distributions: visual (softmax over raw CLIP sims, ×40), face (`FACE_MAP`, 3s mean over frames with a face), music (CLAP; **dropped** when the gate's music / vocals / bed / humming ratios are all < 0.1), text (GoEmotions via `GO_MAP`, `neutral → calm`), sounds (AST laughter / cheering / gasps …).
- `channel_trust`: music 0.5 in speech / speech-over-music reels (a bed); text 0.5 unless the transcript is plain English.
- Strong emotions (`STRONG`: amusement, awe, anger→outrage, sadness→sorrow, moved, fear, disgust, empathic pain): evidence = max over channels (visual at 0.5, so scenes alone never qualify; text backed 0.4 by voice arousal), peaks ≥ 0.6 with prominence 0.15; "moved" also fires on a turn from suffering to hope / kindness. `strength = 100*min(1, share/0.30)`. `audible_laughter_s` (AST) is reported separately.
- Prompt calibration (`calibrate_prompts.py` → `prompt_calibration.json`) only activates with ≥ 15 reference reels.

## Data model (`backend/out/<job>/`)
- `raw/` (from the Space, kept forever) + `report.json`, `report.md`, `report.png`.
- `report.json`: `video, local_video, source_id, duration_s, method, neural_source, modalities, scores{hook_pct, hook_band, hold_rate_pct, finish_pct, avg_attention, virality, virality_base, virality_band, virality_tag, virality_method, virality_breakdown[], virality_summary, emotional_strength}, features{}, moments{}, series{attention, survival_pct, cuts_s, motion, loudness_db, words_per_s, network_z, engagement_raw, speech_prob, vocals_prob}, caption{text, relevance, verdict, evidence, emotion, virality_points}, music{series, summary, beat_times}, emotion{channels, channel_trust, series, summary, strong, music_mood}, speech_gate, transcript, audio, suggestions[], words_timeline[]`.
- One report per video (lowercased file stem): a new import supersedes older reports (moved to `backend/out/.replaced/`).

## Journeys
- **Creator:** pick reel / `o` own file → `a` analyze (queued if busy) → live progress → report (TUI + `v` web) → act on timestamped suggestions → re-analyze.
- **Formula change:** edit `tribe/` → `python tribe/rescore.py` (or `g` / web Regenerate) → same raw data, new reports, no GPU.
- **Validation:** hackathon reel set (`samples/`, git-ignored) — outliers vs lowest-view per stratum; followers / handles = strata and display only.
