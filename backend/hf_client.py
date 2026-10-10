"""Client for the Hookline TRIBE Space (full neural path, plain REST).

Env:
  HOOKLINE_SPACE_URL  e.g. https://wlddhq-hookline-tribe.hf.space
  HF_TOKEN            token with access to the (private) Space
  HOOKLINE_SPACE_ID   Hub id, informational (default harshvardhan-joshi/hookline-tribe)
"""
import os
import time
from typing import Tuple

SPACE_ID = os.environ.get("HOOKLINE_SPACE_ID", "wlddhq/hookline-tribe")


def analyze_via_space(video_path: str, token: str = "",
                      timeout_s: int = 1800, poll_s: int = 15) -> Tuple[dict, str, str]:
    import requests

    base = os.environ.get("HOOKLINE_SPACE_URL", "").rstrip("/")
    tok = token or os.environ.get("HF_TOKEN", "")
    if not base:
        raise RuntimeError("HOOKLINE_SPACE_URL env not set")
    if not tok:
        raise RuntimeError("HF_TOKEN env required to call the private Space")
    headers = {"Authorization": f"Bearer {tok}"}

    with open(video_path, "rb") as f:
        r = requests.post(f"{base}/analyze",
                          files={"video": (os.path.basename(video_path), f, "video/mp4")},
                          headers=headers, timeout=120)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    waited = 0
    while True:
        r = requests.get(f"{base}/report/{job_id}", headers=headers, timeout=60)
        r.raise_for_status()
        report = r.json()
        if "scores" in report:
            break
        if report.get("status") == "failed":
            raise RuntimeError(f"space job failed: {report.get('error')}")
        if waited >= timeout_s:
            raise TimeoutError(f"space job {job_id} not done after {timeout_s}s")
        time.sleep(poll_s)
        waited += poll_s

    report["neural_source"] = report.get("neural_source", "tribev2-full")
    tmp = f"/tmp/hookline_hf_{job_id}"
    png_path, md_path = tmp + ".png", tmp + ".md"
    for url, dest in ((f"{base}/report/{job_id}.png", png_path),
                      (f"{base}/report/{job_id}.md", md_path)):
        d = requests.get(url, headers=headers, timeout=120)
        d.raise_for_status()
        with open(dest, "wb") as f:
            f.write(d.content)
    return report, png_path, md_path
