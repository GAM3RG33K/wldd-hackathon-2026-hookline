# AGENTS.md — Hookline Harness (wldd-hackathon-2026)

Working prototype: video in → attention prediction → creator-actionable report.
Backend: Python FastAPI (`backend/app.py`, `uvicorn backend.app:app --port 8000`, demo script `DEMO.md`). Heavy compute: HF Space `hf_space/` (Docker+Gradio, t4-small) as processing backend; Colab notebook retired.

## Core Beliefs (non-negotiable)
1. **Judge video, not account.** Never use handle/follower count as a signal. Follower range + view count are validation context only.
2. **Scores are within-video, uncalibrated heuristics.** Avg second ≈ 50. Never present hook/hold/virality as measured retention. `method` field must say `heuristic, within-video normalised, uncalibrated` until fitted.
3. **Multimodal or nothing.** Every prediction must combine seen + heard + said over time. Single-modality (script-only) output is rejected.
4. **Timestamped recs or rejected.** Every recommendation carries `at` (e.g. `0:00-0:03`, `0:07-0:10`) + issue + fix. Generic advice fails review.
5. **Colab is the GPU backend; FastAPI is the front door.** Local CPU runs `tribe/` scripts for smoke tests only. Full videos go through Colab bridge. Never commit large videos/weights/preds.

## Rule index
- What runs where (TUI → Space → local backend), step by step: `ARCHITECTURE.md`
- Business logic + scoring contract: `.ai/rules/01-business-logic.md`
- Stack, commands, Colab↔FastAPI bridge: `.ai/rules/02-tech-stack.md`
- Report UI assets (md/png/json, no Figma/Lottie yet): `.ai/rules/03-ui-ux-assets.md`

## Logs + Research (track progress, reuse findings)
- Session logs: `.ai/logs/` (convention: `.ai/logs/README.md`; one file per session, append-only).
- Research: `.ai/research/` (`README.md` index, `tribe-v2-model.md`, `scoring-heuristic.md`, `colab-bridge.md`).
- Every change logs a session entry + promotes proven findings from research → rules.

## Verification loop (every change)
1. `python tribe/tribe_analyze.py --help` parses; `python tribe/hookline_report.py --help` parses.
2. Smoke: 5s trim of `tribe/samples/sneaker.mp4` → `out/` has `preds.npy (T,20484)`, `times.npy`, `meta.json` → report step writes `report.json/md/png` with `scores`, `moments`, `recommendations[]`, `series`.
3. New weights/thresholds → update `CFG` in `hookline_report.py` + this harness, note `PENDING_FIT` vs real view-count data.
4. No handle/follower leakage: grep new code for `follower|handle` as signal; only allow in validation scripts.
5. Backend change → `POST /analyze` (upload) → job → `GET /report/{id}` returns same schema as `report.json`.
