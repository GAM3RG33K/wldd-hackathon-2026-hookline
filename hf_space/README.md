---
title: Hookline TRIBE
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
---

# Hookline TRIBE (private prototype backend)

GPU worker for Hookline: video in → raw model output out. Reports are built by the local backend from the raw bundle.

- `POST /analyze?video=1&audio=1&text=auto&lang=auto` (multipart `video`) → `{job_id}`; one GPU job at a time, others queue.
- `GET /jobs/{id}` live progress (step, chunk, ETA) · `POST /jobs/{id}/cancel|restart` · `GET /jobs/{id}/raw` (zip: preds float16, times, words.tsv, meta.json) · `GET /health`.
- Runs: speech gate (AST), song match (Chromaprint), Hinglish ASR (faster-whisper), TRIBE v2 (`get_events_dataframe` → `predict (T,20484)`), emotion channels (CLIP scenes, YuNet + ViT faces, CLAP music, voice arousal / valence, GoEmotions text).
- Scores are `heuristic, within-video normalised, uncalibrated` (avg second ≈ 50). See `ARCHITECTURE.md` in the repo.
