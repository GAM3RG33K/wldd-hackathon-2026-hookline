# 2026-10-10 — Space reworked to plain REST + token guide
- Why: user wants explicit GET/POST video API, not Gradio protocol. Space `app.py` rewritten as FastAPI (POST /analyze → job, GET /report/{id}[.png/.md], GET /health, GPU asyncio.Lock, BackgroundTasks). Dockerfile CMD → uvicorn. `CLIENT_SNIPPET.md` = curl + requests. `backend/hf_client.py` rewritten to REST poll (HOOKLINE_SPACE_URL/HF_TOKEN), same return contract so `?mode=hf` unchanged. Import verified APP_OK.
- Token Q: permissions guide given in chat (fine-grained Write on Spaces + Read on Models, Llama access request, billing for t4-small).
- Still blocked on: write token → `hf auth login` → create + upload + secret + build watch.
