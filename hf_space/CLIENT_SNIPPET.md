# Calling the Space (plain REST)

Base: `https://<your-space>.hf.space` (private Space → send `Authorization: Bearer $HF_TOKEN`).

```bash
BASE=https://wlddhq-hookline-tribe.hf.space   # adjust to real Space id
# 1. upload
curl -H "Authorization: Bearer $HF_TOKEN" -F "video=@reel.mp4" $BASE/analyze
# -> {"job_id":"abc123","status":"queued"}
# 2. poll (a 28s reel takes minutes on T4)
curl -H "Authorization: Bearer $HF_TOKEN" $BASE/report/abc123
# 3. assets
curl -H "Authorization: Bearer $HF_TOKEN" $BASE/report/abc123.png -o chart.png
```

Python (same flow the FastAPI backend uses in `backend/hf_client.py`):

```python
import os, time, requests
BASE = os.environ["HOOKLINE_SPACE_URL"]
H = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
job = requests.post(f"{BASE}/analyze", files={"video": open("reel.mp4","rb")}, headers=H).json()["job_id"]
while True:
    r = requests.get(f"{BASE}/report/{job}", headers=H).json()
    if "scores" in r: break
    assert r.get("status") != "failed", r
    time.sleep(15)
# r is report.json schema + neural_source="tribev2-full (cuda)"
```
