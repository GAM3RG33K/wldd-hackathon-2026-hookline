#!/bin/sh
# Hookline service launcher — start everything with zero manual steps.
#   1. Installs Python deps if missing
#   2. Starts/waits for the HF Space (GPU) to be RUNNING
#   3. Starts the local backend (uvicorn on :8000)
#   4. Launches the TUI (foreground)
#
# Prereqs: `hf` CLI logged in (hf auth login) OR export HF_TOKEN.
#          Python with pip. No venv required.
set -eu

cd "$(dirname "$0")"

ROOT=$(pwd)
BACKEND_LOG="/tmp/hookline-backend.log"
SPACE_URL="${HOOKLINE_SPACE_URL:-https://wlddhq-hookline-tribe.hf.space}"
SPACE_ID="${HOOKLINE_SPACE_ID:-wlddhq/hookline-tribe}"
BACKEND_PORT="${HOOKLINE_BACKEND_PORT:-8000}"
BACKEND_URL="http://localhost:${BACKEND_PORT}"

say() { printf '[launch] %s\n' "$*"; }
need_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then return 0; fi
  if hf auth token >/dev/null 2>&1; then return 0; fi
  echo "Error: no HF token found. Run: hf auth login  (or export HF_TOKEN=...)"
  exit 1
}

say "== 0 / prerequisites =="
python3 -c "import textual, fastapi, uvicorn" 2>/dev/null \
  || { say "installing Python deps from backend/requirements.txt"; python3 -m pip install -q -r backend/requirements.txt; }
say "deps ok"

say "== 1 / HF Space (GPU) =="
need_hf_token

stage=$(hf spaces info "$SPACE_ID" 2>/dev/null \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['runtime'].get('stage','?'))" 2>/dev/null || echo "?")
say "Space stage: $stage"

case "$stage" in
  RUNNING|RUNNING_BUILDING|RUNNING_APP_STARTING|APP_STARTING|BUILDING)
    say "Space already live, skipping restart"
    ;;
  *)
    say "Starting / restarting the Space …"
    hf spaces restart "$SPACE_ID" >/dev/null 2>&1 || true
    ;;
esac

say "Waiting for Space to be RUNNING …"
hf spaces wait "$SPACE_ID" --timeout 1500s >/dev/null 2>&1 || true

say "Pinging Space health …"
TOKEN=$(hf auth token 2>/dev/null || echo "")
i=0
while [ $i -lt 60 ]; do
  if curl -sf -H "Authorization: Bearer $TOKEN" "$SPACE_URL/health" >/dev/null 2>&1; then
    say "Space is healthy"
    break
  fi
  i=$((i+1))
  sleep 5
done
if [ $i -ge 60 ]; then
  say "WARNING: Space did not respond to /health within 5 min. You can still run the TUI — it will queue jobs."
fi

say "== 2 / Local backend (:8000) =="
if curl -sf "$BACKEND_URL/health" >/dev/null 2>&1; then
  say "Backend already running on :$BACKEND_PORT"
else
  if [ -f "$BACKEND_LOG" ]; then : >"$BACKEND_LOG"; fi
  say "Starting backend on :$BACKEND_PORT (log: $BACKEND_LOG)"
  nohup python3 -m uvicorn backend.app:app --port "$BACKEND_PORT" \
    --host 127.0.0.1 >"$BACKEND_LOG" 2>&1 &
  PID=$!
  i=0
  while [ $i -lt 30 ]; do
    if curl -sf "$BACKEND_URL/health" >/dev/null 2>&1; then
      say "Backend is up (PID $PID)"
      break
    fi
    i=$((i+1))
    sleep 1
  done
  if [ $i -ge 30 ]; then
    say "WARNING: backend did not respond — check $BACKEND_LOG"
  fi
fi

say "== 3 / Terminal UI =="
say "Launching TUI …"
exec python3 tui.py