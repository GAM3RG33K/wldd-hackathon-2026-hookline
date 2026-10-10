# 2026-10-10 — Move GPU backend Colab → HF Space
- Decision: Docker + Gradio Space, private, t4-small. Colab retired (kept for reference).
- Scaffolded `hf_space/` (README/Dockerfile/requirements/app.py/CLIENT_SNIPPET + vendored tribe scripts). Verified no local import (py3.9; Space runs py3.11).
- `hf repos create wlddhq/hookline-tribe` → 403 (no rights under wlddhq). Retry personal → 403. Cause: OAuth login, `hf auth list` empty = no write token.
- Backend wired meanwhile: `hf_client.py`, `POST /analyze?mode=hf`, `/health` hf fields. Restart verified: health 200, demo 200, `hf_configured:false`.
- Next (needs user): write-capable token → `hf auth login` → say go, then create+upload+HF_TOKEN secret+build watch. Then `HF_TOKEN=... HOOKLINE_SPACE_ID=...` + live `mode=hf` test on sneaker.
