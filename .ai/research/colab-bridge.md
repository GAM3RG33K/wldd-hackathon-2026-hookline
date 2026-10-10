# Colab bridge — GPU backend + operator flow (source: `tribe/tribe_colab.ipynb`, `prefetch.py`)

- Runtime: T4 GPU, keep-alive JS cell (foreground tab; cannot beat hard cap/reclaim).
- Order: upload video(s) (sanitised names) → `nvidia-smi` + `uv` + `apt ffmpeg` + punkt + clone tribev2 + `pip -e . matplotlib pandas nilearn` + `torchaudio==<torch>` → patch eventstransforms (whisperx `uvx --python 3.12`, `int8` CPU, `TRIBE_WHISPER_MODEL` env) → faster-whisper patch → `HF login (HF_TOKEN)` → `prefetch` (`best.ckpt`, `w2v-bert-2.0`, `vjepa2`) → `tribe_analyze --device cuda` → `hookline_report` → download `out/`.
- Secrets: Colab secret `HF_TOKEN` (Notebook access on), account with Llama-3.2-3B access. Never paste token in repo.
- FastAPI manual bridge (current): backend stores upload at `backend/out/<jobid>/input.*`, returns `{job_id, status: queued_colab_manual}`; operator runs notebook on that file, drops `report.json/md/png` back; `GET /report/{id}` serves. Next: signed URL + polling worker (PENDING).
- Failure modes logged: numpy-restart loop (re-run cell, no runtime restart or re-upload needed); GPU reclaim (re-run install, re-upload).
