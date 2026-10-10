# Hookline

Predict how a short video (reel) holds attention **before** it is posted: second-by-second attention, hook / hold / finish, viral potential with an explanation, the emotions it evokes, and timestamped editing suggestions. Hookline judges the video, never the account.

Built for the WLDD hackathon 2026. Scores are within-video, uncalibrated heuristics (the average second is about 50); they are model estimates, not measured retention.

## Quick start

```bash
./run-demo.sh            # installs deps once, starts the local backend (:8000), opens the TUI
```

- GPU work runs on the Hugging Face Space `wlddhq/hookline-tribe` (A100). Start it from the TUI with `S` → `s`; it sleeps after 30 min idle. Needs an HF token (`hf auth login`).
- Web reports: `http://localhost:8000/reports`.

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
- **Emotion**: 20 emotions predicted from scenes, faces, music, voice, words and sounds; strong peaks (amusement, awe, outrage, sorrow, being moved, fear, disgust, empathic pain); audible laughter counted separately; emotion dots on the chart follow the video playhead.
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
- `DEMO.md` — demo cues.
- `AGENTS.md` — core beliefs, rule index, verification loop.
- `.ai/rules/` — scoring contract, stack and Space API, report surfaces. `.ai/research/` — findings (emotion, scoring, Space). `.ai/logs/` — session logs.

Sample reels (`samples/`) are hackathon-only data and are not in this repo.
