---
title: Hookline TRIBE
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# Hookline TRIBE (private prototype backend)

Video in → TRIBE v2 cortical response → Hookline attention report.
Called by the FastAPI backend (`POST /analyze?mode=hf`), not by end users.

- `POST /analyze` (multipart video) → `{job_id}` → `GET /report/{job_id}` (poll) — see `CLIENT_SNIPPET.md`
- Full neural path: `get_events_dataframe` → `predict (T,20484)` → `hookline_report.analyse`
- Scores are `heuristic, within-video normalised, uncalibrated` (avg second ≈ 50)
