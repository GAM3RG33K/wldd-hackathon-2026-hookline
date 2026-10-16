#!/bin/bash
# run-test.sh - Manual test script for TTCC model on Apple Silicon (MPS)
set -euo pipefail

echo "=== Starting TTCC test on Apple Silicon (MPS) ==="
echo "Free disk: $(df -h / | awk 'NR==2 {print $4}')"
echo "Current time: $(date)"
echo

# 1. Clone go_viral at pinned commit (893da03)
echo "=== 1. Cloning go_viral at commit 893da03 ==="
rm -rf go_viral
git clone --quiet -b ttcc-rl https://github.com/cliangyu/go_viral go_viral
cd go_viral
git checkout --quiet 893da03bccee6a18d221e96c5de55c9b35a8e350
echo "✓ Cloned go_viral at commit 893da03"

# 2. Install dependencies (skip decord - macOS incompatible, use torchvision backend)
echo "=== 2. Installing TTCC dependencies ==="
cd /Users/w1n/projects/wldd-hackathon-2026/.local/wlddhq:hooklinewldd-ttcc
VENV=/private/var/folders/z0/1lmh900j49sddwj4x46x55g40000gn/T/opencode/local_tribe_env
cd $VENV
$VENV/bin/python -m pip install --quiet torchvision==0.21.0
$VENV/bin/python -m pip install --quiet transformers==4.56.2 qwen-omni-utils==0.0.9 accelerate>=1.0 peft>=0.11,<0.20 safetensors>=0.4 huggingface_hub>=0.30 fastapi uvicorn av==0.19.1
echo "✓ Installed dependencies (av 19.0.1, torchvision 0.21.0)"

# 3. Download checkpoint (9.4 GB)
echo "=== 3. Downloading TTCC checkpoint (8.8 GB) ==="
mkdir -p ~/work/ttcc_ckpt
$VENV/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('liangyuch/ttcc-sft-qwen25omni-3b-v8',
                  revision='1686d5f00408bfd8f6a32be12e71248b6f8ff447',
                  local_dir='~/work/ttcc_ckpt',
                  local_dir_use_symlinks=False)
print('✓ Checkpoint downloaded to ~/work/ttcc_ckpt')
"
echo "✓ Checkpoint downloaded (8.8 GB at ~/work/ttcc_ckpt)"

# 4. Apply PyAV compatibility patch (fixes metadata_errors TypeError)
echo "=== 4. Applying PyAV compatibility patch ==="
mkdir -p /tmp/fix
cat > /tmp/fix/sitecustomize.py << 'EOF'
import av, sys
_orig = av.open
def _patched(*args, **kw):
    kw.pop("metadata_errors", None)
    return _orig(*args, **kw)
av.open = _patched
EOF
# Inject into the venv's site-packages (no system-wide install needed)
cp /tmp/fix/sitecustomize.py $VENV/lib/python3.12/site-packages/sitecustomize.py
echo "✓ PyAV patch applied (sitecustomize.py in venv)"

# 5. Monkey-patch transformers version-checker cascade
echo "=== 5. Monkey-patching transformers version checks ==="
mkdir -p /tmp/fix2
cat > /tmp/fix2/transformers_version_patch.py << 'PYEOF'
# Monkey-patch to bypass transformers dependency version checker
# This prevents: ValueError: Unable to compare versions for tqdm/regex/...: need=X found=None
import transformers.dependency_versions_check as dvc
_orig_check = dvc.require_version

def _patched_check(requirement, hint=None, *, loader=None):
    # Find the currently installed version
    import importlib
    try:
        mod = importlib.import_module(requirement.split(">=")[0].split("<=")[0].split("!=")[0].strip())
        got_ver = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
    except Exception:
        got_ver = None
    if got_ver is None:
        # Silently skip the check if version can't be determined
        return
    _orig_check(requirement, hint=hint)

dvc.require_version = _patched_check
print("✓ Transformers version-check bypass installed")
PYEOF
$VENV/bin/python /tmp/fix2/transformers_version_patch.py

# 6. Build test request
echo "=== 6. Creating test request ==="
TEST_DIR=$T/ttcc_test
mkdir -p $TEST_DIR
cat > $T/ttcc_test/request.json << 'REQJSON'
{
  "protocol_version": "1",
  "repo": "/private/var/folders/z0/1lmh900j49sddwj4x46x55g40000gn/T/opencode/go_viral",
  "checkpoint": {
    "repo_id": "liangyuch/ttcc-sft-qwen25omni-3b-v8",
    "revision": "1686d5f00408bfd8f6a32be12e71248b6f8ff447",
    "local_dir": "/private/var/folders/z0/1lmh900j49sddwj4x46x55g40000gn/T/opencode/ttcc_ckpt",
    "allow_download": false,
    "expected_files": {},
    "verify_weights_sha256": false
  },
  "settings": {
    "env": {
      "RETENTION_HEAD_TYPE": "hazard",
      "ENABLE_AUDIO_OUTPUT": "False",
      "device_map": "mps",
      "torch_dtype": "bfloat16",
      "attn_impl": "sdpa",
      "max_length": 32768
    },
    "prompts": {
      "system": "You are an expert in short-form video advertising. You forecast second-by-second audience retention curves. R(t) is the fraction of viewers still watching at second t, with R(0) = 1 by convention.",
      "user_template": "This ad is {T} seconds long. Estimate the per-second retention curve.",
      "assistant": "<cot></cot>"
    },
    "jobs": [{
      "video_id": "FC018_15s",
      "T": 15,
      "shrink_to_fit": true,
      "video_path": "/Users/w1n/projects/wldd-hackathon-2026/.local/wlddhq:hooklinewldd-ttcc/local_test/FC018/input_trimmed.mp4",
      "raw_output": "/Users/w1n/projects/wldd-hackathon-2026/.local/wlddhq:hooklinewldd-ttcc/ttcc_test/raw_0.json"
    }]
  }
}
REQJSON
echo "✓ Created test request.json"

# 7. Run the test
echo "=== 7. Running inference test ==="
START=$(date +%s)
$VENV/bin/python ttcc_infer.py infer --request $T/ttcc_test/request.json --results $T/ttcc_test/results.jsonl 2>&1 | tee $T/ttcc_test/infer.log
END=$(date +%s)
ELAPSED=$((END-START))

echo ""
echo "=== TEST COMPLETED ==="
echo "Status: $(grep -o '"status": "[^"]*"' $T/ttcc_test/results.jsonl | head -1)"
echo "Elapsed time: ${ELAPSED}s"
echo "Results saved to: $T/ttcc_test/results.jsonl"
echo "Raw output saved to: $T/ttcc_test/raw_0.json"
echo "Full log: $T/ttcc_test/infer.log"
echo "=== Done! ==="
