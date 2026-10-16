# Hookline

Predict how a short video (reel) holds attention **before** it is posted: second-by-second attention, hook / hold / finish, viral potential with an explanation, the emotions it evokes, and timestamped editing suggestions. Hookline judges the video, never the account.

Built for the WLDD hackathon 2026. Scores are within-video, uncalibrated heuristics (the average second is about 50); they are model estimates, not measured retention.

## Project Phoenix (v2, this branch: `project-v2`)

A ground-up rebuild: the retention model runs **locally on Apple Silicon** (no GPU Space), and the web app is a standalone Next.js frontend backed by a small FastAPI service. Everything user-facing is "Project Phoenix".

```bash
./setup.sh      # one time: creates .venv-v2 and verifies every import before saying OK
./run_dev.sh    # starts the backend (:8000) and the web app (:3000) together
```

- **Web app** (`wlddhq-v2/`): upload → live per-step progress (published by the backend, no internals exposed) → full report. Report matches the approved design: realtime video player synced per-frame with the retention chart (drag the chart to scrub; playhead follows playback), chart layers (Retention / Zones / Cuts / Emotion), "Why viral potential" with per-factor what-if switches and the step-by-step calculation, timestamped editing suggestions ranked by viewers lost, caption fit, export (PDF / JSON / Markdown), and a reports list at `/reports`.
- **Scoring** (`phoenix/scorer.py`): builds the whole report from a retention curve R(1..60) — hook / hold / finish, hooks (consecutive retention climbs), drop-off zones (consecutive drops), attention (within-video normalised, avg ≈ 50), timestamped suggestions. Method string stays `heuristic, within-video normalised, uncalibrated` until fitted on real view counts.
- **Model access** (`phoenix/ttcc.py`): calls the pinned retention model in its own environment as a subprocess and returns R(1..60) plus provenance; `phoenix/patch_ttcc_env.py` keeps that environment working (idempotent). If the model environment or enough free memory isn't available, the job falls back to a demo curve clearly marked `demo: true`.
- **Backend** (`backend/app.py`): `POST /api/analyze` queues a job and returns a job id; per-step + overall progress is published while it runs; `GET /api/report/{id}`, `GET /api/reports`, `GET /api/media/{id}` (streams the analysed video for the inline player).
- **Rules that carry over**: judge the video, never the account (follower range + views are display/validation context only); scores are within-video heuristics; every suggestion carries a timestamp and evidence.
- Still to build: the "Edit video" screen (cut / zoom tracks with apply/undo, final playback, re-analyse).

The rest of this README describes the v1 pipeline (TUI + GPU Space), which still works on `main`.

## Quick start (v1)

```bash
./launch-services.sh     # installs deps once, starts the local backend (:8000), opens the TUI
```

- GPU work runs on the Hugging Face Space `wlddhq/hookline-tribe` (A100). Start it from the TUI with `S` → `s`; it sleeps after 30 min idle. Needs an HF token (`hf auth login`).
- Web reports: `http://localhost:8000/reports` (full analyses only) · pitch deck: `http://localhost:8000/pitch` (←/→, `n` notes, `f` full screen).

## Features

**Analyse**
- Full multimodal run on the GPU: TRIBE v2 brain-response model (video + audio + speech), with live progress, chunk counter and ETA; stop (`x`) and restart (`ctrl+r`) without re-uploading.
- Pick a sample reel or your own file (`o`, system file picker); `Enter` / `a` asks for an optional **caption**, then analyzes. Re-analyzing replaces the old report (one report per video).
- Local **queue**: new requests wait while a run is going, or while the Space is paused (they start by themselves once it is live).
- **Run options** (`m`): video / audio / transcript on or off, transcription language (auto / Hindi / English, Hinglish model).
- **Space control** (`S`): live status, ping, start, restart, stop.

**Understand the audio and words**
- Speech gate: speech vs song vs humming vs music per second; songs and humming are judged on visuals + sound, not lyrics.
- Hinglish transcription (word timestamps, hallucination guards); the transcript used for judging is shown in the report.
- Song identification against a trending-audio catalog (context only; YouTube Shorts chart import script).
- Music rhythm: tempo, build-ups, drops, beat-cut sync.

**Score and explain**
- Attention per second, hook (first 3s), hold, finish, drop zones, strongest moments, peaks.
- **Viral potential** with a tag (Low reach → Viral potential) and a **"Why viral potential N"** breakdown: each factor's pull vs a typical reel, biggest boost and biggest drag.
- **Explore the score** on any report: switch each factor in or out to see a what-if score, read the step-by-step calculation, and **edit the weights** (attention weights must add up to 1; lifts 0-1). New weights apply to every report at once, no re-analysis. `/reports` has an **Ignore captions** switch to compare scores without captions.
- **Emotion**: 20 emotions predicted from scenes, faces, music, voice, words and sounds; strong peaks (amusement, awe, outrage, sorrow, being moved, fear, disgust, empathic pain); audible laughter counted separately; emotion dots on the chart follow the video playhead.
- Emotions are **insight, not a score input**: they appear on the report (chart dots, arc, strong peaks) and lift nothing by default — the 2026-10-14 ablation showed an emotion lift lowered the score's own outlier-vs-low separation (`.ai/research/scoring-heuristic.md`).
- **Caption check**: a caption counts only if it describes this video (frames and speech); filler or borrowed captions are ignored. A relevant caption adds context and lifts viral potential, shown in its own report section.

**Improve**
- Editing suggestions at timestamps, each backed by evidence from the reel's own best part (trim, open on the best moment, end earlier, zoom, speed, sound, on-screen text), ranked by estimated viewers lost.
- Model check (`V`): renders the top edits and re-runs them on the GPU against a no-edit baseline.

**Reports**
- TUI report pane, web report (`v`) with video player, chart, transcript and sections above; all reports page (`R`) sortable by viral / latest / oldest; JSON, Markdown and PNG.
- Formulas live locally: `g` (or `python tribe/rescore.py`) rebuilds reports from stored raw model output, no GPU.

## Keys (TUI)

| Key | Action |
|---|---|
| ↑ / ↓ | move; a reel's saved report opens as you land on it |
| Enter / `a` | caption box → analyze (Esc in the box: just the report) |
| `p` | open saved report |
| `o` | pick your own video file |
| `v` / `R` | web report / all reports |
| `m` | run options |
| `S` | GPU Space status and control |
| `x` / `ctrl+r` | stop job (or unqueue) / restart job |
| `g` / `V` | regenerate report / model-check suggestions |
| `?` / `q` | help / quit |

## Docs

- `ARCHITECTURE.md` — what runs where, flow diagram, what is analysed and how scores are built.
- `pitch/deck.md` — the pitch deck source (slides split by `---`, live report blocks; edit and refresh `/pitch`).
- `DEMO.md` — demo cues.
- `AGENTS.md` — core beliefs, rule index, verification loop.
- `.ai/rules/` — scoring contract, stack and Space API, report surfaces. `.ai/research/` — findings (emotion, scoring, Space). `.ai/logs/` — session logs.

Sample reels (`samples/`, including the three presentation videos in `samples/presentation-videos/`) are hackathon-only data and are not in this repo.
