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
