# 2026-10-10 — TUI redesign + Space dependency fixes

## TUI (`tui.py`, Textual)
- Replaced the stdlib menu loop with a Textual app using a fixed IDE-style layout:
  header (backend/Space status), left reels tree (demo, face/faceless/outliers,
  your files, saved reports), right report pane (score tiles, per-second attention
  strip, moments, timestamped "What to change" table), footer key hints.
- Opens on the precomputed demo report so the first screen is never empty.
- Judge path: arrow to a reel, Enter = preview (~3s locally). `f` = full TRIBE on the
  Space (direct upload + polling; result saved to `backend/out/s-<job>/`), `o` = own
  file, `r` = random reel, `w` = web view, `?` = help.
- Backend auto-starts in a worker thread (log: `/tmp/hookline_backend.log`). HF token
  read from `HF_TOKEN` or `hf auth token`.
- Attention strip uses text markers as well as colour (`h` hook window, `!` drop, `*` peak).
- Views are shown dimmed as validation context only; no handle is shown anywhere.
- `run-demo.sh` installs `backend/requirements.txt` (now includes `textual`) on first run.
- Verified headless with `App.run_test`: tree navigation, Enter-to-preview (3s, job
  646333c1), help modal, random reel.

## Space (`hf_space/`)
- Job failed with `partially initialized module 'torchvision' has no attribute 'extension'`.
  HF run logs showed nothing past startup, so `/health` now returns a `boot` import check
  and failed jobs return a `trace` field. Startup now imports torchvision submodules on
  the main thread. After redeploy, boot reports torch/ta 2.5.1+cu121, tv 0.20.1, tv_ops true.
- Next failure: `Could not import module 'AutoModel'`. transformers 5.19 was installed and
  needs torch >= 2.6. Pinned `transformers>=4.46,<5` in the Dockerfile layer 4, and the boot
  check now imports `AutoModel`.
- After the transformers pin (Space sha 1a3d965): boot shows transformers 4.57.6. The 2s clip
  (`/tmp/clip2s.mp4`, job 64993a19) completed end-to-end with `neural_source: tribev2-full (cuda)`.
  Scores: hook 65.7, hold 100, finish 77.1, virality 72.1. This is the first full TRIBE run on the Space.

## Progress + timing (Space + TUI)
- Space `/report/{id}` now returns progress while running: `stage`, `step/steps`, `elapsed_s`,
  `eta_s`, `timeout_s` (2x estimate + 120s), `overdue`, per-step `plan` (estimate and actual),
  current `substep`, `substeps` timings, `log` tail (30 lines) and `mem` (RAM/RSS/GPU).
  A heartbeat line is logged every 20s. ETA priors are per step (fixed + per video second),
  and `SCALE` learns from actual runs, clamped to 0.25-4x.
- Sub-step timing: each event transform (extract_audio, chunk_*, transcribe, add_text,
  sentences, word_context, remove_missing) and each feature extractor `prepare()` (wrapped at
  class level), plus `predict/loaders+inference` as the remainder. Logged as `>`/`<` lines and a
  "slowest" summary at the end. Timings are also written to `meta.json`.
- TUI: `f` shows a live progress view with a bar, steps (done/current/pending), the current
  sub-step, a time-per-sub-step bar chart, memory and the Space log tail. The status bar shows
  step, sub-step, elapsed, ETA and timeout, and flags when the Space stops answering for over 30s.

## Sneaker 28s run 1 (job e5d0161b): CUDA OOM
- load_model 4s (cached), events 114s, then predict failed at about 3 minutes.
- RAM: process RSS went up to 14.3 of 16.2 GB during predict (likely video frames), which is
  probably what caused the earlier full-Space hang.
- GPU: Wav2VecBert self-attention went OOM (14.35 GiB allocated). tribev2 only chunks Audio and
  Video events above 60s (min 30s), so the whole 28s clip went through attention in one pass.
- Fix: `HOOKLINE_CHUNK_S=10` (ChunkEvents max 10s / min 5s), set via our own copy of
  `get_audio_and_text_events`, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  and `empty_cache` after each sub-step.
- Transcription: faster-whisper `small` replaced by `base` (multilingual, about 3x faster),
  `int8_float16`, greedy decoding (beam 1) and VAD filter. Override with `TRIBE_WHISPER_MODEL`.
- Runtime ENV moved below the pip layers in the Dockerfile so tuning doesn't reinstall torch.

## Sneaker 28s run 2 (job df75072d): first full report, 20m14s on t4-small
- Scores: hook 23.3 (Bad), hold 40.9, finish 9.5, virality 35.5 (Weak). The preview had
  hook 61.4 and virality 50.7, so the synthetic preview brain differs a lot from TRIBE.
- Time: load 4s, events 111s (transcribe 97s, add_text 13s), text features (Llama-3.2-3B)
  270s, Wav2VecBert 8s (no OOM with 10s chunks), video features (V-JEPA2-g, 64 frames per
  clip, 2 Hz) about 810s, which is 67% of the total.
- ETA bug: once a step ran past its plan, the ETA was floored at 5s + 11s for the report, so
  it showed "16s" forever. Fixed: video ETA = time left in the current chunk + remaining chunks
  x average chunk time; other loops use their rate; past the plan with no live signal the ETA
  is reported as unknown together with the overrun. Priors refit from this run (predict 30s +
  40s per video second).
- Chunk visibility: neuralset's V-JEPA path loops over chunks without tqdm, only over frame
  steps ("Encoding video"). Each such loop is now counted as `video chunk i/n start-end`.
  Heartbeats moved out of the log tail (it had filled with them). `/jobs/{id}` serves
  progress after the job is done.
- Stop/restart: cooperative cancel at every step, sub-step and loop item
  (`POST /jobs/{id}/cancel`); restart re-queues the same upload (`POST /jobs/{id}/restart`).
  Tested locally: stops within 0.3s mid-loop. TUI: `x` stop, `ctrl+r` restart.

## Speed round 1 (deployed 433e000, verification run pending)
- fp16: `PreTrainedModel.from_pretrained` defaults to float16 on CUDA unless a dtype is
  passed, and predict runs under `torch.autocast(fp16)`. Neuralset loads Llama, V-JEPA2,
  DINOv2 and Wav2VecBert in fp32 by default. Env `HOOKLINE_PRECISION=fp32` reverts.
- Resident models: tribev2's `_free_extractor_model` is now a no-op while allocated GPU
  memory is at most `HOOKLINE_KEEP_GB` (9). On CUDA OOM, free them, set keep to 0 for the
  process, and retry predict once.
- Transcription proof: `/health.boot.transcript` shows whether the faster-whisper patch
  applied, and each job logs `whisper <model> on <device>: Xs audio in Ys`.
- Every HF model load is logged with its time (`model_loads`), so load and compute can be
  separated.

## Speed round 1 result (job 794fe136, fp16, all modalities)
- 634s total (10m34s) vs 1214s in fp32: 1.9x faster. Scores and moments match the fp32
  baseline (hook 23.4 vs 23.3, everything else equal), so fp16 costs no quality.
- Text features 270s -> 66s (42.5s of that is the Llama load). Video chunks 80-95s each
  (17-20 frame steps at about 4.8s). Model loads: Llama 42.5s, V-JEPA2 32.4s, w2v-bert 6.4s.
- KEEP_GB=9 kept Llama + w2v (8.3 GB), so V-JEPA went OOM on chunk 2; the fallback freed them
  and retried, wasting 166s. KEEP_GB is now 4 (V-JEPA + w2v only).
- Transcription is still about 95s. The proof hook never fired (an exception was swallowed in
  the patch). Parked: the user asked to skip transcription for now.

## Per-job modalities (deployed 6c32725)
- `POST /analyze?video=1&audio=1&text=0` (defaults: video+audio on, text off). Text off
  skips transcription and Llama; video/audio off drops those events, so tribev2 removes
  the extractor and the model never loads. TRIBE zero-fills missing modalities
  (model.py:188-190, trained with modality dropout).
- At job start, resident models (and whisper) that this job doesn't need are freed.
- Scorer: with text off, speech is treated as unknown (no "no spoken line" note, no
  "no speech" in drops, words excluded from the signal mix). The report's `method` notes a
  partial-modality diagnostic run.
- Note on AGENTS.md core belief 3 (multimodal: seen+heard+said): text off drops the "said"
  part. That's fine for diagnostics, but the final demo reports should run with all three on.
- TUI: `v`/`a`/`t` toggle the modalities (header shows checkboxes), and they are sent with `f`.
- Result on t4-small, video+audio (text off): run 1 (cold) 339s, run 2 (warm) 281s.
  Video features take 269-302s (80-95s per 10s chunk), so V-JEPA2 compute sets the floor on a T4.
  The V-JEPA2 reload took only 1.2s on the warm run (weights in page cache), so keeping
  models loaded matters little. neuralset builds V-JEPA2 as a local variable, so it isn't kept on the extractor anyway.
- Hardware moved to a100-large ($2.50/hr) at the user's request. Sleep time is still 48h; suggested 1800s.

## a100-large ($2.50/hr, sleep 1800s)
- The first A100 jobs failed in 2s with `Permission denied: '/.cache'`: this tier runs the
  container as a non-root user with HOME=/. The faster-whisper patch also failed (it couldn't
  write site-packages). Fix (final Dockerfile layers, no torch reinstall): HOME, HF_HOME,
  XDG_CACHE_HOME and TORCH_HOME under /tmp, NLTK_DATA readable, tribev2 dir and /app/cache
  writable, and spaCy en_core_web_lg pre-installed (neuralset pip-installs it on first use,
  which also fails as non-root).
- Test reel: samples/face/videos/FC013.mp4 (72.4s, random pick, instead of sneaker).
  - video+audio (job 4167e700): 200s total. Video 174s over 7 chunks of about 23s each
    (T4: 80-95s per chunk, so 3.8x). Model loads about 4s each. Scores: hook 45.7, hold 6.9,
    finish 3.2, virality 35.1.
  - all three (job d604de1a): failed on the missing spaCy model after transcribe 61.7s
    and add_text 7.6s. Re-running after the fix.
- Duplicate-word bug: tribev2 chunks Audio before transcription, and ExtractWordsFromAudio
  then copies the whole-file transcript into each chunk, shifted by the chunk start (n-fold
  words, n-1 of them mis-timed). FC013 had 2086 "words", 7x the real count. Fixed by
  transcribing before chunking. All earlier text-on runs had the bug (sneaker: 219 words for
  about 73 real ones). Real cancel tested on job c0990eb9: stopped 3s after the request.
- FC013 all three modalities after the fix (job d249dee9, a100-large): 301s for 72.4s.
  299 words (was 2086). Transcribe 64s, add_text 2s, text features 21s (was 424s+ and
  cancelled), w2v 3s, video 186s (7 chunks at 24-32s), loaders+inference 4s, report 3s.
  Model loads: Llama 6s, V-JEPA2 3s, w2v 2s. Scores: hook 38.0, hold 7.4, finish 2.8, virality 31.9.
  Remaining cost: video 62%, transcription 21% (still parked).
