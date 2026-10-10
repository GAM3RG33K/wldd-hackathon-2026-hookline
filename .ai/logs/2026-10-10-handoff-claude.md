# Handoff (2026-10-10) — Hookline wldd-hackathon-2026

## What this is
Hackathon prototype: video in → attention prediction → creator-actionable report.
Problem: creators learn a video failed only after posting. System must be multimodal
(seen+heard+said), judge video-not-account, give timestamped recs. Stack: Python/FastAPI
front door + HF Space (Docker, t4-small) running TRIBE v2 brain-encoding + heuristic scorer.

## Decisions (locked)
- Colab retired → HF Space `wlddhq/hookline-tribe` (private, t4-small $0.40/hr). Plain REST
  (POST /analyze → GET /report/{id}[.png/.md], GET /health), NOT Gradio.
- Scores = heuristic, within-video normalised, uncalibrated (avg second ≈50). Never call it retention.
- Samples live in private dataset `wlddhq/hookline-samples` (50 face FC + 50 faceless FL + 10 outliers OL + metadata.csv). NEVER bake videos into Space repo. metadata handle/views = validation only.
- Backend `?mode=preview` (synthetic brain + real ffmpeg signals, seconds) vs `?mode=hf` (full TRIBE, minutes). Same report.json schema.

## Current state (verify on resume)
- Space: BUILDING commit 5519a16 (Dockerfile split into cached layers + tribev2 pinned @af58661 + torch trio 2.5.1/2.5.1/tv0.20.1). Boot check prints torch/ta/tv versions. `hf spaces info`, `space.sh --status-only`.
- Local backend :8000 RUNNING with HF_TOKEN + HOOKLINE_SPACE_URL (restarted 12:3x; plain `pkill`+uvicorn loses env — re-export!). `hf_configured:true`.
- TUI (`./run-demo.sh`): options incl. [2] own-file→Space direct, [4] Space. TUI [4] on samples routes via backend ?mode=hf.
- Space transcription fixed via faster-whisper startup patch. Last full-run blocker chain: torchaudio/torchvision skew (fixed by pins) — next test: 2s clip /tmp/clip2s.mp4 (407KB, from sneaker.mp4).
- `hf` CLI authed (token hf-wldd-harshvardhan). Upload trap: `hf upload REPO f1 f2` maps f2 as DEST — upload files one per command, verify via `hf download ... --force-download`.

## Key files
`backend/app.py`, `backend/hf_client.py` (SPACE_URL/HF_TOKEN env), `hf_space/{Dockerfile,app.py,requirements.txt,hookline_report.py,tribe_analyze.py,CLIENT_SNIPPET.md}`, `tui.py`, `space.sh`, `run-demo.sh`, `DEMO.md`, `samples/`, `tribe/{tribe_analyze.py,hookline_report.py}`, `.ai/{rules,logs,RESEARCH}` (01-business-logic has scoring formulas/CFG; hf-space.md, colab-bridge.md retired).

## Next steps for Claude
1. Confirm Space RUNNING + boot imports OK (`torch=2.5.1+cu121 ta=2.5.1+cu121 cuda=True`).
2. POST /tmp/clip2s.mp4 → poll; expect scores (first run downloads ~18GB weights, slow).
3. Then full sneaker.mp4 via TUI [4]; compare vs preview job 60d65bcb.
4. Fit CFG vs views strata (research/scoring-heuristic.md plan) when time allows.
