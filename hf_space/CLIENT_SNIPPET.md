# Calling the Space (plain REST)

Base: `https://wlddhq-hookline-tribe.hf.space` (private Space → send `Authorization: Bearer $HF_TOKEN`).
The TUI (`tui.py`) and `tribe/rerun_all.py` use exactly this flow.

```bash
BASE=https://wlddhq-hookline-tribe.hf.space
H="Authorization: Bearer $HF_TOKEN"
# 1. upload (modalities + transcript mode + language)
curl -H "$H" -F "video=@reel.mp4" "$BASE/analyze?video=1&audio=1&text=auto&lang=auto"
# -> {"job_id":"abc123","estimate_s":...}
# 2. live progress (step, chunk, ETA); ~2.5-2.8s of GPU per video-second on a100-large
curl -H "$H" $BASE/jobs/abc123
# 3. raw bundle once status == done (preds float16, times, words.tsv, meta.json with all channels)
curl -H "$H" $BASE/jobs/abc123/raw -o raw.zip
# 4. build the report locally (backend on :8000): fetch raw + score + replace older reports of this video
curl -X POST "localhost:8000/import/abc123?video=/abs/path/reel.mp4&source_id=reel"
```

Stop / restart a running job: `POST $BASE/jobs/{id}/cancel`, `POST $BASE/jobs/{id}/restart` (no re-upload).
`GET $BASE/report/{id}` still returns the Space's own fallback report (same schema), but the local import is the source of truth.
