# 02 — Tech Stack + Bridge Rules (FastAPI + Colab)

## Locked stack
- **Inference:** Python, `facebook/tribev2` (`best.ckpt` via `prefetch.py`), encoders `facebook/w2v-bert-2.0`, `facebook/vjepa2-vitg-fpc64-256`, text `Llama-3.2-3B` (needs Colab secret `HF_TOKEN` with access), `torch`, `faster-whisper` (`TRIBE_WHISPER_MODEL=small`), `nilearn`, `pandas numpy matplotlib`, `ffmpeg/ffprobe` on PATH, `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Backend (new):** Python FastAPI + uvicorn. Uploads → job dir `backend/out/<jobid>/` mirroring `tribe/out/<stem>/`. Endpoints: `POST /analyze (multipart video, ?max_seconds) → {job_id}`, `GET /report/{job_id} → report.json`, `GET /report/{job_id}.md/.png`. No auth in prototype; validate mime + size cap; never overwrite `tribe/` scripts from backend.
- **Colab backend:** `tribe/tribe_colab.ipynb` is canonical GPU path: T4 runtime → keep-alive cell → upload → `apt ffmpeg` + `pip uv` + clone tribev2 + `pip -e .` + `torchaudio==<torch>` + punkt + eventstransforms patch + faster-whisper patch + `HF login` + `prefetch` + run `scripts/tribe_analyze.py --device cuda` + `hookline_report.py` → download `out/`. Keep notebook cells in sync with `tribe/*.py` (notebook `%%writefile` copies).
- **Local dev:** CPU/MPS smoke only: `python tribe/tribe_analyze.py <mp4> --max-seconds 5 --out /tmp/hl --device cpu`, then `python tribe/hookline_report.py /tmp/hl/<stem>`.

## Hard rules
- Absolute paths in tools/scripts; `ffmpeg -y -loglevel error`; ffprobe duration parse float; per-second bins `int(t)`; transcript `.tsv` may be None → zeros, not crash.
- `CFG` lives only in `hookline_report.py` (single source). Backend must not duplicate scoring — shell out or import it.
- Colab bridge (until queue exists): FastAPI stores upload + returns job_id with `status: queued_colab_manual`; operator runs notebook on `backend/out/<jobid>/input.*`, drops `report.*` back; `GET` serves them. Next step: signed upload URL + polling worker — mark `PENDING`.
- Never commit: videos, `out/`, `cache/`, `*.npy`, `*.ckpt`, `*.safetensors`, `.tsv` transcripts. `prefetch.py` fetches weights; document `HF_TOKEN` setup, never paste token.
- Conventions: `snake_case` py, `--kebab-case` CLI flags (`--max-seconds --out --device --cache`), `out/<stem>/{preds.npy,times.npy,meta.json,report.json,report.md,report.png}`, timing logs `[timing] …s` to stdout, atlas warnings to stderr, exit 1 on missing input.
- Python ≥3.11 (tribev2). Pin new deps in backend `requirements.txt`; `torchaudio` must match `torch` major.
