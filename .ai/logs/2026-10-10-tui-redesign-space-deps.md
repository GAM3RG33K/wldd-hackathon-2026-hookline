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
- Transcription speed: the old fast path was spliced into tribev2's source file as text and
  ended in `except ImportError: pass`, so any import problem silently fell back to
  `uvx whisperx` (installs whisperx and downloads an alignment model every job; this matches
  the steady ~60-95s). Replaced with a direct patch of `ExtractWordsFromAudio._get_transcript_from_audio`
  from app.py: one cached faster-whisper model, greedy decoding + VAD. On any error it logs the
  exact exception (job log, `transcript` field, `/health.boot.transcript`) and then falls back.
  `/health.boot.faster_whisper` shows the import and ctranslate2 CUDA status. Startup no longer
  edits site-packages. Deployed after FC021 (job 2e827081, 147s for 40.1s) finished.
- TUI polls the Space every 1s (was 5s).

## Report chart + missing PNG/MD
- Space jobs never returned a PNG or markdown: `/report/{job_id}` was declared before
  `/report/{job_id}.png`, so `<id>.png` matched as a job id ("unknown job"). Asset routes now
  come first. Some scripts had saved that 404 JSON as report.png/md; those files were removed,
  along with the s-* folders of failed jobs (error JSON instead of a report).
- Web view (`/report/{id}/view`): the chart is drawn as SVG from report.json (attention,
  viewers remaining, hook window, strongest window, drop zones, scene cuts, peak), full width
  and 30vh tall, so every saved report has a chart even without a PNG. Also: score tiles,
  correct preview vs full-TRIBE label, modalities, markdown generated locally when missing,
  PNG shown under "Full chart" only when it is a real PNG, and dark mode.
- TUI: AttentionStrip is 30vh tall and full pane width, with a 0/50/100 y-axis, bars coloured
  by zone, survival dots drawn over the bars, a marker row (h/!/*/|), time ticks and a legend.

## Batch of 6 random reels (a100-large, all three modalities), 15:43-15:56
- All 6 done, 12.5 min total. Run time per reel 43-260s for 13-92s of video, about 2.5-2.8s per video second.
- The faster-whisper fast path is confirmed working: 0.2-4.2s per reel (was 60-95s via uvx whisperx).
  Video features are now about 90-95% of each job.
- PNG and markdown now come back from the Space (route-order fix).
- Word counts look low for some reels: FC032 0 words in 18s, FC023 17 in 51s, FL034 10 in 27s.
  Language is hard-coded "english" in tribev2's ExtractWordsFromAudio; Hindi/Hinglish or
  music-only reels may lose speech. Needs a check (language detection, or `language=None`).
- Scores vs views (n=6, validation context only): both outliers (65M and 50M views) scored lowest
  (hook 47/46, virality 38/35), while FC032 (567k) and FC023 (56k) scored hook 88. The scores are
  within-video normalised and uncalibrated, so they are not comparable across reels. Even so, there
  is no positive relation in this sample. The CFG fit against views (research/scoring-heuristic.md) is needed before any claim.

## Hinglish ASR, speech gate, audio type + trending song match
- ASR is pluggable (`asr=hinglish|whisper`, `lang=auto|hi|en`). Default is
  Oriserve/Whisper-Hindi2Hinglish-Apex (Apache-2.0, large-v3-turbo fine-tune), converted to
  CTranslate2 at startup (13s). A/B: FL034 Hinglish kept as spoken ("To main thik karta hoon na
  tumko"), whereas stock Whisper translated it to English. It hallucinated "nan nan ..." on a
  song (FC032), so guards were added: compression ratio > 2.4, no-speech, segments with < 30%
  distinct words, repeated and glued tokens.
- Gemini as ASR was discussed and not built: it has no reliable word timestamps (it would need
  MMS forced alignment), and it sends audio to a third party.
- Speech gate (`text=auto`, default): AST AudioSet classifier (MIT/ast-finetuned-audioset-10-10-0.4593)
  on 2s windows with a 1s hop. Per-second groups: speech, vocals (song/singing/rap), music,
  music bed, Indian (Bollywood/Carnatic), humming, laughter. Audio type: song / speech_over_music
  / speech / instrumental / humming / other. Transcription runs only for speech types; after
  that, transcript confidence can still drop it. FC023 (14% speech over music) was wrongly
  gated at a 20% threshold; it is now speech_over_music, so the transcript stays on.
- Song match (report context only): Chromaprint fpcalc fingerprints against
  hf_space/trending_audio.json (11 iTunes previews: CreatorDB global list for the week of
  2026-10-07, plus 2 India picks marked unverified). Every analysed reel's fingerprint is also
  indexed, so audio reused across reels is flagged. No reel audio leaves the Space. Synthetic
  test: similarity 0.81 for a shifted copy with ~12% bit noise vs 0.53 for unrelated audio;
  match threshold 0.65.
- Report: the `audio` block (type, song, reused_by), `transcript` (used_by_model, status,
  timestamped segments), and `series.speech_prob` / `series.vocals_prob`. Web view: local video
  player (`local_video` saved by the TUI and batch, served at /report/{id}/video) next to the
  chart, with a playhead synced to playback; a transcript section with clickable timestamps;
  and an audio line. TUI: an audio line under the moments, and the transcript under "What to change".
- Bug (fixed): when inserting `_warm_catalog` I placed it under the `@app.on_event("startup")`
  decorator, so `_startup` never ran. That skipped the Hinglish warm-up, the faster-whisper check
  and the main-thread torchvision import. The batch waiter then hung on `hinglish_asr`, so no job
  ran. Check after any deploy: `/health.boot` must contain `ok`.
- Demo cleanup: backend/out reduced to demo/ only (removed 12 preview jobs, 27 s-* reports
  and the summary files), plus tribe/__pycache__ and .DS_Store files. The backend on :8000 was
  restarted with HF_TOKEN + HOOKLINE_SPACE_URL. The demo report plays tribe/samples/sneaker.mp4.

## Virality: why 52, interim fix, calibration run
- The old formula was 0.45*hook + 0.30*hold + 0.25*avg attention. Avg attention is ~50 by design
  (normalised within the video), so a quarter of the score was a constant (~12.5). Hold multiplies
  a 93-99.5% per-second keep rate, so long reels are penalised whatever their content.
- Interim (uncalibrated): 0.40 hook + 0.25 drop-free share + 0.20 strong-moment share
  (A>=70, x2, capped) + 0.15 ending. Tags: Low reach <35, Average <55, Above average <70,
  Popular <85, Viral potential. The screenshot reel (2cf9e419 preview) goes 52 -> 60
  "Above average".
- Every report now has a `features` block that is comparable across videos: raw TRIBE
  engagement, hook lift, peak/drop share, ending, volatility, cuts/s, motion, loudness,
  words/s, duration, raw network means. tribe/fit_virality.py fits viral outliers (label 1)
  vs lowest-view reels (label 0) on those features: Cohen's d per feature, L2 logistic
  regression on the top 5, leave-one-out accuracy. It writes tribe/virality_model.json,
  which hookline_report loads automatically. Account data is used only to pick the groups.
- Calibration batch: OL001-OL010 (10M-315M views), then the 10 lowest-view face/faceless reels
  FL019, FC021, FL024, FC046, FC040, FL028, FC037, FC019, FC045, FC039 (2k-12k views), labelled
  [i/20] in one sequential chain.

## Reports built locally from raw TRIBE output
- The Space now returns raw output at GET /jobs/{id}/raw: a zip of preds (float16, about 2-3 MB
  for 60s), times, words.tsv (the words the model used) and meta.json (gate, audio match,
  transcript info, modalities, timings). It lives in /tmp, so it is lost on a Space restart;
  clients fetch it right after "done".
- Local: tribe/rescore.py stores it in backend/out/s-<job>/raw/ and builds report.json/md/png
  against the local video. The backend has POST /import/{space_job}?video=..&source_id=.. and
  POST /report/{id}/regenerate. The web report has a "Regenerate from raw model output"
  button; the TUI has `g`. The TUI and batch import through the backend (fallback: the Space report).
- hookline_report.finalize() holds the merge logic (gate, transcript, audio, method notes)
  for both the Space and local builds.
- The calibration chain was stopped at [8/20]. The 7 outlier reports had no raw output, so they
  were removed and all 20 reels are being re-run, now with raw output stored.

## Calibration run complete (20 reels, all built locally from raw TRIBE output)
- Groups: 10 outliers (10M-315M views; 6 face / 4 faceless) vs the 10 lowest-view reels
  (2k-12k; 7 face / 3 faceless). The page types are roughly balanced, so the face/faceless
  confound is small.
- Effect sizes (Cohen's d, viral minus low): net_visual -0.97, net_language +0.81, motion -0.76,
  peak_gap_cv -0.71 (viral peaks are more evenly spaced), net_attention -0.61, peaks +0.60,
  peaks_per_window +0.59, cuts_per_s -0.54, hook_lift +0.45, hook +0.42. Avg engagement +0.18
  and drop_share +0.13 are near zero.
- Dry-run L2 logistic fit on the top 5 features: leave-one-out accuracy 65% (chance 50%).
  n=10 vs 10, so the 95% interval on each d is roughly +-0.9: directional evidence only.
- Not applied yet; it is waiting for the user's decision. The Space scorer was synced after the batch (1c3f1e7).

## Emotion + suggestions deploy: model loading fix
- The first deploy loaded no emotion models: transformers refuses `.bin` weights with
  torch < 2.6 (torch.load CVE), and TRIBE pins torch 2.5.1. openai/clip-vit-base-patch32 ships
  .bin only (its safetensors versions are unmerged community PRs). Switched to
  laion/CLIP-ViT-B-32-laion2B-s34B-b79K (MIT, safetensors). The voice (audeering) and text
  (SamLowe GoEmotions) models already ship safetensors. use_safetensors=True is forced
  everywhere, and the warm-up loads each channel independently (BOOT shows ready / PARTIAL).
  It now boots `emotion_models: ready (11s)`.
- The test chain had hung silently twice: the first time it waited on a model warm-up that
  never finished, the second time on a quoting bug in its health check. Lesson: the chain
  must check that the job was actually submitted, not just wait.

## Emotion + suggestions test (one reel per category, a100)
- Emotion overhead: 0.7-3.4s per reel (mostly frame extraction; each model <= 0.4s).
- FC032 (song): channels visual + sounds; audio typed "instrumental" (AST vocals barely fire
  on this mix). Vocals threshold to tune at the next Space deploy.
- FL011: tempering CLIP (p^0.5) flipped the dominant emotion from awe 77% to amusement 88%;
  the visual channel is near-tied, so the report should show the full mix, not just the winner.
- Brain salience cross-check r = -0.64 / -0.39 / -0.02: it gives no support for the arousal
  track yet. Shown honestly, not used.
- "End earlier" cut 72% of an 18s reel; now capped at 25% of the reel.
- Verification: re-encoding alone shifts mean engagement by -5.6% (OL003 null edit vs
  original). Variants are now compared with a null edit (re-encode only, one per reel). OL003's
  zoom at 0:04-0:08: edited stretch +5.1%, rest +0.8%, so "improved" (it was "worse"/"no clear
  change" when compared with the original).

## Taxonomy test (FL041, FC032, FL011, OL003; new per-prompt cues)
- FL041 compared with a reference evaluation of the reel (elephant rescue; mournful woodwind score):
  music "slow mournful melody" 80%, sadness 79%, sorrow peaks for 36s (strength 100), viral
  36 -> 68. Still missing: empathic pain (max 0.3), the hope/relief ending, and the resolving arc.
- A general bias: amusement dominates FC032 (94%), FL011 (100%) and OL003 (79%). The "funny prank
  or blooper" prompt has the highest raw similarity on every reel (hub prompt), and GoEmotions
  amusement fires on Hinglish. Deviation from each prompt's own mean recovers FL041's story
  (crying 0-10s, injured animal 10-20s / 60-70s, act of kindness 80-90s).
- Per-prompt calibration (tribe/calibrate_prompts.py -> prompt_calibration.json; z vs corpus,
  reels weighted equally) backfired with 4 reels: FL041 is 86 of the 165 seconds, so its sadness
  became the norm. Calibration is now off below 15 reels; it needs a balanced reference run.
- One report per video: a newer run replaces older ones (full beats preview); keyed by file name;
  replaced folders go to backend/out/.replaced/. 22 duplicates cleaned up.

## 2026-10-11 — TUI: GPU Space status + control modal
- Header shows the live Space stage from the HF Hub runtime API (`● live`, `◐ starting/building`, `○ paused/sleeping`, `✕ error`), polled every 15s; works while the Space is paused (no wake, no billing).
- `S` opens `SpaceScreen` modal: stage, hardware, sleep timer, health ping latency, active jobs; buttons/keys `p` ping, `s` start, `r` restart, `x` stop.
- Start = `POST /api/spaces/{repo}/restart` (resumes a paused Space) or a wake ping when SLEEPING; stop = `POST .../pause`. Stop and restart kill running jobs, so they need a second press within 5s.
- `/jobs` polling for "live on Space" now skipped unless the Space is running. No new dependencies (urllib only).
- Verified headless (Textual run_test): paused stage rendered, stop disabled when paused, restart confirm arms and expires. Start/stop not exercised (would bill).

## 2026-10-11 — TUI: Enter/p opens saved reports only
- Enter / `p` no longer runs the local synthetic-brain preview. They open the reel's saved report, found by file name (`report_for`: this session's run, else `report_index` built in `refresh_saved`; full TRIBE preferred over previews, newest first).
- No report: the reel card + status say "Press f to run the analysis and build the report". Own files (`o`) do the same.
- Removed the TUI `run_preview` worker; backend `/analyze?mode=preview` is kept for the web upload page.
- Verified headless: FC001 (saved) opens its report; FC002 (none) shows the press-f card on `p` and Enter.
- Trending audio from the YouTube Shorts chart (`tribe/update_trending.py`) is parked as low priority; the catalog rewrite is uncommitted.

## 2026-10-11 — Emotion: product review read as "laughter" (fix A + B)
- Cause (mohit-burad review, s-41cc98d9): CLAP labelled the music bed "comedic quirky funny music" (90%); music alone could make strong peaks at full weight -> 34s laughter, strength 100, viral 68. GoEmotions "neutral" (53% on Hinglish text) was dropped, leaving amusement 0.57 on a 30s segment; voice-arousal wobble made it peak.
- A: GO_MAP `neutral -> calm`; `channel_trust()`: music weight 0.5 in reels the speech gate calls speech / speech_over_music (bed), in both fusion and strong-peak evidence.
- B: transcript emotion weight 0.5 unless the transcript is plain English (GoEmotions is English-only).
- Rejected: requiring 2-channel agreement for strong peaks (removed FL041's 36s sorrow; FL041 is music-led).
- Re-scored 30 reports. Review 67.8 -> 35.5 (no strong peaks). FL041 sorrow 36s kept (68.2). Comedy reels with speech also lost laughter peaks (OL003 85.8 -> 78.1, FL011 62.1 -> 55.2, FC037 77.8 -> 55.5, FC029 61.0 -> 49.5): their laughter also came from the music bed, i.e. we cannot yet detect jokes in speech.
- Still open: the review's everyday mood track still reads amusement 83% (music bed at half weight + CLIP "hub" prompts "hilarious / prank"; no product-demo prompt). Needs informational prompts on the Space (re-run) and/or the >=15-reel prompt calibration.

## 2026-10-11 — TUI keys: a analyze, v view, o file picker, m run options
- `a` = analyze (full TRIBE on the Space, was `f`); `v` = view report in browser (was `w`).
- `o` opens the system file picker (macOS `osascript choose file`, Linux `zenity`; typed-path modal as fallback) and starts the analysis as soon as a file is picked. Re-picking the same file reuses its tree node.
- Modality toggles moved off `v`/`a`/`t`/`l` into a `RunOptions` modal on `m` (same keys inside it); header shows "m run: [x] video [x] audio [a] transcript auto".
- `a` refuses with "press S to start it" when the Space is paused/stopped/errored instead of failing the upload.
- Headless tests: picker stub -> node added -> run started; m modal toggles; v opens /report/{id}/view; paused guard. DEMO.md + ARCHITECTURE.md key references updated.

## 2026-10-11 — TUI: re-analyze from saved reports, local analysis queue
- `a` on a "saved reports" entry resolves the report's video (`video_of_report`: sample reel by id, else the local file) and re-runs it; "Re-analyzing X" when a report exists (supersede replaces it on import).
- Queue: `a` / `o` while a run is active (or queued) -> `enqueue` with the run options at that moment; `run_space` worker always clears `space_run` in `finally` and calls `start_next`. Duplicate requests are refused; `x` on a queued item removes it; tree shows "N queued here". If the Space is paused when the next item is due, the queue waits with a "press S" message.
- `begin_run` marks the Space busy before the upload starts, so a second request during a long upload queues instead of uploading in parallel. Stop/restart refuse politely while uploading. Fixed: the 3h give-up path left `space_run` set.
- Headless test with a fake job: 3 requests -> 1 running + 2 queued, duplicate refused, x removes, queue drains in order with each item's own options.

## 2026-10-11 — Emotion: deadpan monologue read as "laughter 31s" (rajpsych_2, s-a78b5828)
- Causes: (1) speech gate heard 0% music (87% speech) but CLAP still ranked mood prompts on the voice ("comedic quirky funny music", "mournful") -> invented amusement + sorrow peaks. (2) The running backend imported this reel with the OLD fusion code: `rescore` reloaded only `hookline_report`, never `emotion_fusion` / `taxonomy` / `suggest` / `audio_music` (report had no `channel_trust`). (3) Strong label "laughter" read as audible laughs; it meant predicted amusement.
- Fixes: `has_music()` drops the music channel when the gate's music / vocals / bed / humming ratios are all < 0.1. `rescore` reloads TX, AMU, EF, SG, HL (dependencies first); backend restarted. Strong label now "amusement" (predicted, evoked); new `strong.audible_laughter_s` from AST laughter / giggle / chuckle, shown in web + TUI ("no audible laughter in the audio").
- Re-scored 35 reports. rajpsych_2: no strong peaks, amusement 100% (evoked mood, plausible for comedy), viral 47.8. FL041 sorrow 36s kept. Audible laughter 0s on every reel so far.
- Limit: humour itself is still inferred from CLIP "hilarious / prank" prompts + GoEmotions on Hinglish, not from understanding the joke. Next: a semantic layer (VLM / Hinglish-capable LLM over transcript + frames).

## 2026-10-11 — Report ordering: web sortable, TUI latest first
- Web `/reports?sort=viral|latest|oldest` (default viral; `recent` kept as an alias for latest), links for all three, "Analysed" column with date + time.
- TUI "saved reports" always latest first.
- Order uses `analysed_at()` = mtime of `raw/preds.npy` (preview: `preds.npy`): re-scoring rewrites report.json AND raw/meta.json, so their mtimes made every re-scored report look new.
- Sort is now a dropdown (`<select id=sort>`, navigates on change). The 5s meta refresh became a JS reload that skips while the dropdown has focus, so the menu does not close under you. Checked in Chrome: picking "oldest first" reloads with ?sort=oldest, selection kept, FC030 first.

## 2026-10-11 — "Why viral potential N" section
- `hookline_report.virality_breakdown()`: per factor (hook, few drop-offs, strong-attention moments, ending) score /100, weight, points, and pull vs a typical reel (factor at 50 -> weight x (score - 50)). `finalize` appends "Emotional peaks" with the actual lift points. Fitted model path (when `virality_model.json` exists) reports weight x standardised feature instead.
- `scores.virality_summary`: "Biggest boost: X, +n points vs a typical reel (why). Biggest drag: Y, -n points (why)."
- Web report: section after the chart (summary, table, diverging bar, explanation, total = base + emotional lift). TUI: "viral potential:" line with factors sorted by pull + the summary. Markdown: table.
- Re-scored 35 reports, backend restarted. Checked FL041 in Chrome: emotional peaks +31.9 (boost), hook -18.6 (drag), total 36.3 + 31.9 = 68.

## 2026-10-11 — Face channel (emotion portrayed on screen)
- Why: rajpsych_2 (performed "Dhoka Diya" meltdown: pleading, rage, despair) showed no emotions. Voice arousal/valence alone confuses emphasis with anger (product review CTA ~ same signature as rage); CLIP "crying / argument" prompts never beat the "funny prank" hub prompt; per-reel prompt z-scores were too noisy (fear / disgust false peaks).
- Space (`hf_space/emotion.py`): `_face` = OpenCV YuNet face detection + `trpakov/vit-face-expression` (Apache-2.0, safetensors, FER2013 7 labels) on the 1 fps frames CLIP already uses; raw `emotion.face = {probs, faces, area}` per second. Dockerfile: opencv-python-headless 4.10 (if cv2 lacks FaceDetectorYN) + YuNet ONNX in /app/models. Warm-up loads it at boot.
- Local (`taxonomy.FACE_MAP`, `emotion_fusion.face_dist`): angry->anger, sad->sadness, fear, disgust, surprise, happy->amusement/excitement/hope, neutral->calm; 3s mean over frames with a face (single frames jitter). Face is a full-weight channel in fusion and strong peaks.
- Validated before deploy on CPU (88 frames in 5.7s): rajpsych -> sorrow 0:28, 1:01, outrage 1:07; product review -> nothing; FL041 unchanged.
- Deployed (Space commit 88e090c; this upload also shipped the parked trending catalog, now 66 tracks). Smoke test on the Space: rajpsych 185s, 79 face frames, same peaks, old report superseded. Re-running all 45 reels with reports (scratchpad rerun_all.py, ~95 min A100).

## 2026-10-11 — Docs refresh + checkpoint
- Rewrote stale docs to the current system: AGENTS.md (Space = raw data, local = reports; verification loop with rescore / Space deploy / headless TUI), DEMO.md (current keys and flow), `.ai/rules/01-03` (pipeline, emotion contract, virality + breakdown, data model, Space API, report surfaces), `.ai/research/` (new `emotion.md`; scoring, hf-space, README updated), `hf_space/README.md`, `hf_space/CLIENT_SNIPPET.md`, ARCHITECTURE.md (face channel, channel trust, why-viral, reports sort, re-analyze, trending script).
- Promoted the batch re-run script to `tribe/rerun_all.py` (repo-relative paths).

## 2026-10-11 — Optional caption + plain follower ranges
- Caption: TUI `CaptionPrompt` before every analysis (Enter empty = none, Esc cancels, prefilled with the last caption); multipart field `caption` -> Space `_caption` (CLIP frame sims + gallery sims, multilingual MiniLM transcript vs controls, GoEmotions) -> raw `emotion.caption`; local `tribe/caption.py` decides relevance / verdict / emotion / lift; report `caption` block; web + md + TUI sections; `rerun_all` keeps a report's caption.
- Validated on CPU with real frames: review true captions relevance 1.0 (+9.7), rajpsych true 0.54 (+3.8); lorem ipsum, "follow for more", borrowed and off-topic captions all 0. Gallery `hf_space/caption_gallery.npy` (55 sample reels x 8 frames, CLIP embeddings only; `tribe/build_caption_gallery.py`).
- Follower markers `! $ $$ $$$` replaced by plain ranges ("146k views · 200k–1M followers"); reels column 46 wide; reel card adds the size word.
- Pending: Space deploy after the 45-reel re-run finishes (deploy restarts the container).
- Deployed (Space commit 87c867c) after stopping the re-run at 27/45 (1 Space job cancelled). GPU check on the product review: borrowed caption -> ignored (frames z 0.53, speech z 0.91, +0); true Hinglish caption -> matches (2.75 / 3.14, relevance 1.0, +9.7, viral 45.3). Same numbers as the CPU test. Caption emotion list drops "calm".

## 2026-10-11 — Fix: x did not stop Space jobs started elsewhere
- `x` only knew this session's own run (`space_run`), so jobs from the batch script, another TUI or a TUI started before a restart answered "No Space job running". Now: selected "live on Space" job -> cancel; own run -> cancel; otherwise the only active Space job -> cancel on a second press within 5s; several -> asks to select one. Headless test covers all four paths (Space call stubbed).

## 2026-10-11 — Caption box on every Enter / file pick
- Enter on a reel or file -> `CaptionPrompt` -> (re-)analyze; Esc in the box opens the saved report (or the no-report card). `p` still opens the report without a box. `o` always shows the box.
- No more "Space paused, press S" refusal before the box: with the Space paused / stopped the run is queued, and `poll_space` starts the queue by itself once the stage is RUNNING.
- Headless test: Enter -> box, Esc -> report, Enter + caption with Space paused -> queued with caption, o -> box -> queued, Space live -> first queued run starts.
- Caption box is now a 5-line wrapping `CaptionArea` (TextArea subclass): Enter submits, pasted line breaks are kept, live counter n/2200.

## 2026-10-11 — Docs: README with feature list, diagrams with caption path
- New root `README.md`: overview, quick start, feature list (analyse / audio + words / score + explain / improve / reports), TUI keys, docs index.
- ARCHITECTURE.md: ASCII diagram (caption box, queue, caption.py, Space control) and mermaid flow (caption input, caption signals on the Space, caption check -> emotion + virality lift, why-viral breakdown, sortable reports). DEMO.md, AGENTS.md, `.ai/rules/03`, `hf_space/README.md` updated for the caption box, Enter flow and follower ranges.

## 2026-10-11 — Fix: live entry label + tree jumping to the top
- Live entries glued name and stage ("lift-prankpredict 16s"): now "name  ·  stage 16s", long names cut with "…".
- Root cause of the jump: `add_own_file` (and `r`) called `move_cursor` on a node added / expanded in the same refresh; the node has no line yet, so the cursor fell to line 0 ("★ Demo report", top of the panel). `focus_node` moves + scrolls after the next refresh. Reproduced headless before/after: o -> caption -> live job -> saved refresh now keeps "lift-prank.mp4" selected and the scroll position.
- Also hardened: live list updates add/remove only changed entries; saved-report rebuilds restore the cursor node and scroll (`cursor_snapshot` / `cursor_restore`, highlight side effects skipped while restoring).

## 2026-10-11 — Emotion: no clear emotion by default, stricter faces, confidence
- Calibration rebuilt from 36 reels (now active). Visual / music: raw softmax share x ramp(z) gate; text: probability x ramp(z) (neutral untouched); caption emotion never adds calm.
- Per-second label: strongest single trusted channel >= 0.30 else "no clear emotion"; summary `no_clear_share`, `confidence`, `channels_agreeing`; shown in web + TUI emotion lines.
- Faces: area >= 1%, top expression >= 0.6, minus 70% of the reel median per expression.
- Iterations on the 3 check reels (FL041, rajpsych_2, product review): absolute-z (FL041 lost everything), fused-prob dominance (FL041 99% none), text z explosion (rajpsych disgust 17%) -> final in .ai/research/emotion.md. Only these 3 reels re-scored; the rest still carry the previous emotion scoring until `python tribe/rescore.py`.

## 2026-10-11 — Pitch deck web app (/pitch)
- `pitch/deck.md` (12 slides mapped to the 5 judging criteria) rendered by `backend/pitch.py` at `/pitch`: keyboard / swipe navigation, speaker notes (`n`), full screen (`f`), progress bar; live blocks `{{scores|report|why|emotion|suggestions:KEY}}`, `{{architecture}}`, `{{validation}}` (sample reels with full runs: n, median views, mean viral potential, Spearman vs views: all +0.20, face +0.38, faceless -0.11, outliers +0.02 at build time). Edit the file and refresh; no restart. Link from /reports.
- GPU wrap-up: deployed Space commit 0492f58, re-ran the 3 check reels (review +caption: viral 45, no strong peaks; FL041 +caption: 66, sorrow 19s; rajpsych_2: 54, outrage 2s + sorrow 2s), no other jobs, Space PAUSED 03:38. No GPU work until next morning.

## 2026-10-11 — Pitch redesign + shared visual system
- `backend/theme.py`: one token set (cool neutrals, cobalt attention accent, semantic good / caution / loss, light + dark), Bricolage Grotesque display, IBM Plex Sans body, IBM Plex Mono for timecodes and numbers (Google Fonts with system fallbacks). Used by /pitch, /reports and the report view.
- Deck rewritten (16 slides): cover with the showcase reel's real attention curve; the brief mapped to the 5 judging criteria; 12 product problems we hit with what we built (no labels -> TRIBE v2; judge the video; fusion; Hinglish; songs; speed; explainable score; GPU/scoring split; wordless stories; false emotions; generic advice; spam captions); connected system diagram; 7-step file-to-verdict pipeline; extraction table; indicators (attention, hook, frame changes, drop-offs, strongest moments, emotions); caption evaluation; verdict with live why-viral; live showcase report; emotion; validation; recommendations; live demo; next. Renderer gained @cover, @section, ::: cards (problem -> what we built), ::: steps.
- /reports rebuilt: header bar, summary (reels, full GPU runs), sort dropdown, viral-potential value + meter + tier chip, timecode-style numbers. Report view moved onto the same tokens and type.

## 2026-10-11 — Deck v3: WLDD branding, graphics over text
- Visual reference: WLDD "MX Player case study" deck (navy ground, royal-blue blocks, heavy white Montserrat, teal ring charts, numbered circles, »» chevrons, phone mockups). Applied to /pitch only (single dark look by choice); /reports + report view keep the shared light/dark system.
- 18 slides, one idea each: cover (phone mockup playing the showcase reel + its attention curve), survival curve with big numbers, icon rows, 12-row problem » answer flow, speed bars + numbered stat blocks, connected architecture, 7-step icon pipeline, layer matrix, annotated indicator chart (hook, drop-offs, strongest, peak, cuts, viewers remaining), emotion timeline + before/after bars, caption z bars with thresholds, why-viral diverging bars, ring-chart scores + embedded report, validation scatter (viral potential vs log views by stratum, Spearman chips), edits on the timeline, live demo, next, closing.
- New: `/report/{id}/poster.jpg` (frame at 1 s, cached) so the phone shows the reel before playback; videos play only on their slide.

## 2026-10-11 — Deck naming + new showcase
- Deck wording: "the user interface" (TUI), "the server" (local backend), "the virality predictor" / "video analysis model" (GPU Space); architecture diagram and flow rows updated.
- Re-scored all reports with the current emotion scoring (38 ok; 3 failed: their videos were deleted) so the deck compares like with like.
- Showcase = highest viral potential among full-run reports after re-scoring: OL003 (78, Popular, 19 s, face data). The problem slide keeps the weak product review (2% reach the end) via a separate `problem:` key.

## 2026-10-11 — Deck v4 (21 slides): title screen, Virality Predictor page, tools page, report buttons
- Title screen "Hookline — From Video to Virality", author Harshvardhan Joshi; footer on every slide shows title + author. Section "From file to verdict" -> "From Video to Virality".
- Model naming: TRIBE v2 and "GPU" appear only on the new "The Virality Predictor" page (what TRIBE v2 is, how we use it, the Higgsfield idea we followed, CC BY-NC licence note, links to both articles). Everywhere else it is "the Virality Predictor"; new "Models and tools behind it" page (12 cards).
- Order: architecture -> Virality Predictor -> models and tools -> speed. No more report iframe: every slide showing one reel's data has an "Open full report" button in the header (cover: under the title; `@report KEY` for text-only slides).
- Card grids size by count (<=3 large, 4 -> 2x2, 5-6 -> 3 columns, 7+ -> 4 columns). "Emotion that is predicted". OL003 caption do / don't with real CPU-scored z values (scene captions frames z 6.5-6.9, speech z 3.7-4.8, +3.3; vague / filler / off-topic ignored). Verdict slide shows the optional caption row (+3.3, 78 -> 81). "See it" = ring charts + playing phone. Live slide: "Launch the user interface" button -> POST /launch-ui (localhost only, macOS Terminal running ./run-demo.sh). Next: "Tweak model and formula", "Understand and predict ... the vibe of the frame", "One tool to predict them all".

## 2026-10-11 — Final deck polish + reports wording
- Removed the speed slide (internal optimisation). Validation legend: large reel-type colour discs, bigger dots and chips. Models and tools cards carry link chips (top right) to every public model / library actually used.
- Reports page + report view: "Full analysis" (+ caption matched / ignored), "Quick preview", "N full analyses · M quick previews"; no GPU / TRIBE wording; footer says "Full analysis by the Virality Predictor".
- /reports lists full analyses only (quick previews stay on disk, reachable by URL); summary "N reels, full analysis".
