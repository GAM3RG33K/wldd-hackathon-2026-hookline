#!/usr/bin/env bash
# setup.sh — create/verify the Python environment for the Phoenix v2 backend.
# Run this once before ./run_dev.sh. Verifies every import before saying OK.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv-v2"
PY=python3

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${YELLOW}==> Phoenix v2 setup${NC}"

# 1. Python available?
if ! command -v "$PY" >/dev/null 2>&1; then
  echo -e "${RED}ERROR: python3 not found. Install Python 3.10+ first.${NC}"; exit 1
fi
echo "python: $($PY --version)"

# 2. Create venv if missing
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "creating venv at $VENV ..."
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet --upgrade pip

# 3. Pinned deps (pydantic pinned to a version compatible with modern fastapi)
echo "installing backend dependencies ..."
pip install --quiet \
  "fastapi>=0.115,<1" \
  "pydantic>=2.6,<3" \
  "uvicorn[standard]>=0.30" \
  "python-multipart>=0.0.9" \
  "numpy<2" \
  "matplotlib" \
  "pandas"

# 4. Verify EVERY import the backend needs, before declaring success
echo "verifying imports ..."
"$VENV/bin/python" - <<'EOF'
import sys
sys.path.insert(0, ".")
failures = []
try:
    import fastapi; print(f"  fastapi {fastapi.__version__}")
except Exception as e: failures.append(f"fastapi: {e}")
try:
    import pydantic; print(f"  pydantic {pydantic.VERSION}")
    from pydantic import BaseModel  # the exact import path fastapi uses
except Exception as e: failures.append(f"pydantic: {e}")
try:
    import uvicorn; print(f"  uvicorn {uvicorn.__version__}")
except Exception as e: failures.append(f"uvicorn: {e}")
try:
    import numpy; print(f"  numpy {numpy.__version__}")
except Exception as e: failures.append(f"numpy: {e}")
try:
    import multipart; print("  python-multipart ok")
except Exception as e: failures.append(f"python-multipart: {e}")
try:
    from phoenix.scorer import analyse; print("  phoenix.scorer ok")
except Exception as e: failures.append(f"phoenix.scorer: {e}")
try:
    from phoenix import ttcc; ok, why = ttcc.environment_ready()
    print(f"  phoenix.ttcc ok (model env {why})")
except Exception as e: failures.append(f"phoenix.ttcc: {e}")
try:
    from backend.app import app; print("  backend.app ok")
except Exception as e: failures.append(f"backend.app: {e}")
if failures:
    print("\nFAILED:")
    for f in failures: print(f"  - {f}")
    sys.exit(1)
print("all imports verified")
EOF

echo -e "${GREEN}==> Setup complete. Run ./run_dev.sh to launch.${NC}"
