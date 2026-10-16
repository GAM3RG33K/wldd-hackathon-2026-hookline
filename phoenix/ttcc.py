"""Local TTCC inference client.

Calls the pinned TTCC runner (ttcc_infer.py) in its own environment as a
subprocess — the model env (torch 2.6, transformers 4.56.2, MPS) is separate
from the backend venv by design. One process per run loads the checkpoint,
predicts the retention curve R(1..60) for the video and writes raw output.

Nothing here substitutes another model: the runner verifies the checkpoint
and the retention-head weights before predicting.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned locations on this machine
TTCC_PY = Path("/private/var/folders/z0/1lmh900j49sddwj4x46x55g40000gn/T/opencode/local_tribe_env/bin/python")
RUNNER = REPO_ROOT / ".local" / "wlddhq:hooklinewldd-ttcc" / "ttcc_infer.py"
GO_VIRAL_REPO = Path("/private/var/folders/z0/1lmh900j49sddwj4x46x55g40000gn/T/opencode/go_viral")
CHECKPOINT = Path("/private/var/folders/z0/1lmh900j49sddwj4x46x55g40000gn/T/opencode/ttcc_ckpt")
CHECKPOINT_REPO_ID = "liangyuch/ttcc-sft-qwen25omni-3b-v8"
CHECKPOINT_REVISION = "1686d5f00408bfd8f6a32be12e71248b6f8ff447"

T_MAX = 60

PROMPTS = {
    "system": ("You are an expert in short-form video advertising. You forecast "
               "second-by-second audience retention curves. R(t) is the fraction of "
               "viewers still watching at second t, with R(0) = 1 by convention."),
    "user_template": "This ad is {T} seconds long. Estimate the per-second retention curve.",
    "assistant": "<cot></cot>",
}


def environment_ready() -> tuple[bool, str]:
    """True when the model env, runner and checkpoint are all present."""
    for path, what in ((TTCC_PY, "TTCC python"), (RUNNER, "runner"), (GO_VIRAL_REPO, "go_viral repo"),
                       (CHECKPOINT, "checkpoint")):
        if not path.exists():
            return False, f"missing {what}: {path}"
    return True, "ready"


def _ensure_version_patch() -> None:
    """Apply the transformers version-check patch if it isn't already in place
    (idempotent; keeps the model env working after venv reinstalls)."""
    import subprocess as sp
    try:
        sp.run([sys.executable, str(REPO_ROOT / "phoenix" / "patch_ttcc_env.py")],
               capture_output=True, text=True, timeout=30, check=True)
    except Exception:
        pass  # never block inference on the patch; the runner will report real errors


def run_inference(video_path: Path, out_dir: Path, video_id: str,
                  duration_s: float, timeout_s: int = 1800) -> list[float]:
    """Run TTCC on one video. Returns r_pred as a plain list R(1..T).

    Raises RuntimeError with a readable message on any failure.
    """
    ok, why = environment_ready()
    if not ok:
        raise RuntimeError(f"TTCC environment not ready: {why}")
    _ensure_version_patch()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.json"
    request_path = out_dir / "request.json"
    results_path = out_dir / "results.jsonl"
    for p in (raw_path, results_path):
        if p.exists():
            p.unlink()

    T = max(1, min(T_MAX, int(math.ceil(duration_s))))
    request = {
        "protocol_version": "1",
        "repo": str(GO_VIRAL_REPO),
        "checkpoint": {
            "repo_id": CHECKPOINT_REPO_ID,
            "revision": CHECKPOINT_REVISION,
            "local_dir": str(CHECKPOINT),
            "allow_download": False,
            "expected_files": {},
            "verify_weights_sha256": False,
        },
        "settings": {
            "env": {
                "RETENTION_HEAD_TYPE": "hazard",
                "ENABLE_AUDIO_OUTPUT": "False",
            },
            "device_map": "mps",
            "torch_dtype": "bfloat16",
            "attn_impl": "sdpa",
            "max_length": 32768,
        },
        "prompts": PROMPTS,
        "jobs": [{
            "video_id": video_id,
            "T": T,
            "shrink_to_fit": True,
            "video_path": str(video_path),
            "raw_output": str(raw_path),
        }],
    }
    request_path.write_text(json.dumps(request, indent=1))

    cmd = [str(TTCC_PY), str(RUNNER), "infer",
           "--request", str(request_path), "--results", str(results_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"TTCC inference timed out after {timeout_s}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError("TTCC runner failed:\n" + "\n".join(tail))

    # one line per job in results.jsonl
    line = None
    for raw in results_path.read_text().splitlines():
        if raw.strip():
            entry = json.loads(raw)
            if entry.get("video_id") == video_id:
                line = entry
    if line is None:
        raise RuntimeError("TTCC runner wrote no result line")
    if line.get("status") != "ok":
        raise RuntimeError(f"TTCC job failed: {line.get('error', 'unknown')}")

    raw_out = json.loads(Path(line["raw_output"]).read_text())
    r_pred = [float(x) for x in raw_out["r_pred"]]
    if not r_pred:
        raise RuntimeError("TTCC returned an empty retention curve")

    # keep the provenance next to the prediction
    prov = {
        "video_id": video_id, "T": raw_out.get("T"), "window": raw_out.get("window"),
        "input_tokens": raw_out.get("input_tokens"),
        "inference_seconds": raw_out.get("inference_seconds"),
        "checkpoint": f"{CHECKPOINT_REPO_ID}@{CHECKPOINT_REVISION[:12]}",
        "device": raw_out.get("provenance", {}).get("device"),
    }
    (out_dir / "r_pred.json").write_text(json.dumps(
        {"r_pred": r_pred, "provenance": prov}, indent=1))
    return r_pred
