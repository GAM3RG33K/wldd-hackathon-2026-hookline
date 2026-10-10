# HF Space — full neural backend (decision 2026-10-10, replaces Colab)

- Choice: Docker Space + Gradio (`hf_space/`), one `analyze` endpoint (queue, concurrency 1).
  Serverless Inference Providers rejected (custom TRIBE code + gated Llama + minute-long runs).
  Inference Endpoints rejected for prototype (cost from minute one, no demo URL).
- Target: `harshvardhan-joshi/hookline-tribe` (org `wlddhq` 403 on create with current token), private, `t4-small` ($0.40/hr).
- Space files: `Dockerfile` (py3.11, ffmpeg, cu121 torch, tribev2 git, faster-whisper), `app.py` (model warm at startup, `analyze` → report dict + png + md), `tribe_analyze.py` + `hookline_report.py` vendored copies, `CLIENT_SNIPPET.md`.
- Backend: `backend/hf_client.py` (`gradio_client`, `HOOKLINE_SPACE_ID`/`HF_TOKEN` env), `POST /analyze?mode=hf` (502 + error if Space unreachable), `/health` shows `hf_space` + `hf_configured`.
- Secrets needed on Space: `HF_TOKEN` (account with Llama-3.2-3B access) for gated text encoder.
- Blocker: CLI authed via browser OAuth with no stored token (`hf auth list` empty) → repo create 403 on org AND personal. Unblock: fine-grained write token → `hf auth login` → `hf repos create` + `hf upload` + secrets + hardware.
