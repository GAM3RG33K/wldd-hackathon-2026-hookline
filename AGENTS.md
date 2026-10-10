# AGENTS.md — Hookline Harness (wldd-hackathon-2026)

Working prototype: video in → attention prediction → creator-actionable report.
Front end: Textual TUI (`tui.py`, `./run-demo.sh`) + web report (`backend/app.py`, `uvicorn backend.app:app --port 8000`), demo cues `DEMO.md`. Heavy compute: HF Space `wlddhq/hookline-tribe` (`hf_space/`, Docker + FastAPI, a100-large, sleeps after 30 min idle) returns raw model output only; every report is built locally from that raw output (`tribe/rescore.py`). Colab notebook retired. Feature list + keys: `README.md`; what runs where: `ARCHITECTURE.md`.

## Core Beliefs (non-negotiable)
1. **Judge video, not account.** Never use handle/follower count as a signal. Follower range + view count are validation context only.
2. **Scores are within-video, uncalibrated heuristics.** Avg second ≈ 50. Never present hook/hold/virality as measured retention. `method` field must say `heuristic, within-video normalised, uncalibrated` until fitted.
3. **Multimodal or nothing.** Every prediction must combine seen + heard + said over time. Single-modality (script-only) output is rejected.
4. **Timestamped, evidence-based suggestions or none.** Every editing suggestion carries `at` (e.g. `0:00-0:03`) + the evidence against the reel's own best stretch + one concrete edit (`tribe/suggest.py`). Generic template advice stays hidden (`SHOW_SUGGESTIONS=False` in `hookline_report.py`).
5. **The Space produces raw data; the local backend produces reports.** All GPU models (TRIBE, ASR, speech gate, song match, emotion channels incl. faces) run on the Space and travel in the raw bundle (`GET /jobs/{id}/raw`). Scoring, emotion fusion, virality and suggestions run locally and can be regenerated without a GPU. Never commit videos, weights, preds or the hackathon `samples/`.

## Rule index
- What runs where (TUI → Space → local backend), step by step: `ARCHITECTURE.md`
- Business logic + scoring contract: `.ai/rules/01-business-logic.md`
- Stack, commands, Space ↔ local backend bridge: `.ai/rules/02-tech-stack.md`
- Report surfaces (TUI, web report, md/png/json): `.ai/rules/03-ui-ux-assets.md`

## Logs + Research (track progress, reuse findings)
- Session logs: `.ai/logs/` (convention: `.ai/logs/README.md`; one file per session, append-only).
- Research: `.ai/research/` (`README.md` index, `tribe-v2-model.md`, `scoring-heuristic.md`, `emotion.md`, `hf-space.md`).
- Every change logs a session entry + promotes proven findings from research → rules.

## Verification loop (every change)
1. `uvx pyflakes tui.py backend/app.py tribe/*.py hf_space/*.py` shows no undefined names; edited `tribe/` scorer files are copied to `hf_space/` (`cmp`).
2. Scoring change → `python tribe/rescore.py` re-scores every report from its raw output (no GPU); compare before/after on known reels (FL041 = sorrow, product review = no strong emotion, rajpsych_2 = performed sorrow/outrage). Restart the backend (it caches modules; `rescore` reloads the scorer modules).
3. Space change → `hf upload wlddhq/hookline-tribe hf_space . --repo-type space`, wait for stage `RUNNING` (not `RUNNING_BUILDING`), `/health` `boot.emotion_models` = `ready`, then one short reel end to end (TUI `a`, or `python tribe/rerun_all.py --only=<name>`).
4. TUI change → headless Textual `run_test` with the GPU upload stubbed; never claim a run started without checking `/jobs`.
5. New weights/thresholds → update `CFG` / `VIRALITY_WEIGHTS` in `hookline_report.py` + `.ai/research/scoring-heuristic.md`; keep `PENDING_FIT` until fitted on real view counts.
6. No handle/follower leakage: grep new code for `follower|handle` as signal; only allow in display (follower range next to views) and validation scripts.
