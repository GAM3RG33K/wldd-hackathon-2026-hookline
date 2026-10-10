# 02 — Tech Stack + Space ↔ Local Bridge

## Locked stack
- **GPU Space** `wlddhq/hookline-tribe` (`hf_space/`): Docker, FastAPI on :7860, a100-large ($2.50/hr), sleep after 1800s idle. Pins: `torch 2.5.1 / torchvision 0.20.1 / torchaudio 2.5.1`, `transformers<5` (safetensors only: `.bin` refused on torch < 2.6), tribev2 at a fixed commit, faster-whisper, Chromaprint, OpenCV (YuNet), spaCy `en_core_web_lg`. Runs as non-root: HOME / HF caches in `/tmp`.
- **Models:** TRIBE v2 (V-JEPA2 ViT-g, Wav2Vec-BERT 2.0, Llama-3.2-3B — needs `HF_TOKEN` Space secret with Llama access), `Oriserve/Whisper-Hindi2Hinglish-Apex` (CTranslate2), `MIT/ast-finetuned-audioset`, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, `laion/clap-htsat-fused`, `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (CC-BY-NC-SA, prototype only), `SamLowe/roberta-base-go_emotions`, `trpakov/vit-face-expression` + OpenCV YuNet.
- **Local:** Python 3.12, `backend/requirements.txt` (FastAPI, uvicorn, textual, scipy, librosa, nilearn, matplotlib …), ffmpeg / ffprobe on PATH. `./run-demo.sh` installs deps once and opens the TUI (which starts the backend).

## Space API (`hf_space/app.py`)
- `POST /analyze?video=&audio=&text=auto|on|off&lang=auto|hi|en&asr=hinglish|whisper` (multipart `video` + optional form field `caption`, ≤ 2200 chars) → `{job_id, estimate_s}`; one GPU job at a time, others queue.
- `GET /jobs`, `GET /jobs/{id}` (step, substep, loops, chunk progress, ETA), `POST /jobs/{id}/cancel`, `POST /jobs/{id}/restart` (no re-upload), `GET /jobs/{id}/raw` (zip), `GET /report/{id}` (fallback report), `GET /health` (`boot`: model warm-up status per component).
- Space control from the TUI uses the Hub API: `GET/POST https://huggingface.co/api/spaces/<repo>/{runtime,restart,pause}`.

## Local backend (`backend/app.py`, :8000)
- `POST /import/{space_job}?video=&source_id=` → fetch raw, rescore, supersede. `POST /report/{id}/regenerate`, `POST /report/{id}/verify`, `GET /report/{id}` (JSON), `/report/{id}.md|.png`, `/report/{id}/view` (web report), `/report/{id}/video`, `/reports?sort=viral|latest|oldest`.
- `POST /analyze?mode=preview` (synthetic-brain preview) is used by the web upload page only.

## Hard rules
- Scoring lives only in `tribe/` (`CFG`, `VIRALITY_WEIGHTS` in `hookline_report.py`). The backend imports it; never duplicate formulas. `rescore` reloads `taxonomy, audio_music, emotion_fusion, suggest, hookline_report` (dependencies first); `rescore.py` itself is cached — restart the backend after editing it.
- Edit `tribe/` first, copy changed scorer files to `hf_space/`, upload with `hf upload wlddhq/hookline-tribe hf_space . --repo-type space --exclude "__pycache__/*"`. A Space build takes ~10–15 min; `RUNNING_BUILDING` is the old container.
- Report times use the model output's mtime (`raw/preds.npy`), never `report.json` / `raw/meta.json` (re-scoring rewrites those).
- Never commit: videos, `samples/`, `backend/out/*` (except `demo/`), `cache/`, `*.npy`, weights, tokens. HF token from `hf auth token` / `HF_TOKEN`, never pasted into files.
- No third-party APIs receive reel audio or video (Gemini / Sarvam / AudD deferred). YouTube chart refresh is a manual browser paste (scripts get HTTP 429).
- Conventions: `snake_case` Python, per-second bins `int(t)`, `M:SS` timestamps, transcript may be absent (zeros / unknown, not crash).
