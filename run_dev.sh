#!/bin/bash
# run_dev.sh — launch the Phoenix v2 backend + frontend in one go.
# Run ./setup.sh first (it creates .venv-v2 and verifies all imports).

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/wlddhq-v2"
VENV="$PROJECT_ROOT/.venv-v2"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

trap 'echo -e "\n${YELLOW}Shutting down services...${NC}"; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' SIGINT SIGTERM

echo -e "${GREEN}Starting Phoenix v2 development environment${NC}"

[[ -d "$BACKEND_DIR" ]] || { echo -e "${RED}Error: backend dir missing${NC}"; exit 1; }
[[ -d "$FRONTEND_DIR" ]] || { echo -e "${RED}Error: frontend dir missing${NC}"; exit 1; }

# Guard: setup must have run and verified imports
if [[ ! -x "$VENV/bin/python" ]]; then
  echo -e "${RED}No venv found. Run ./setup.sh first (it installs + verifies everything).${NC}"
  exit 1
fi

# ── backend ──────────────────────────────────────────────────────────
echo -e "${YELLOW}Starting backend (FastAPI) on http://localhost:$BACKEND_PORT${NC}"
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
uvicorn backend.app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!
echo -e "${GREEN}Backend PID: $BACKEND_PID${NC}"

# wait briefly and confirm it actually came up
for _ in $(seq 1 20); do
  sleep 0.5
  if curl -sf "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1; then
    echo -e "${GREEN}Backend healthy: http://localhost:$BACKEND_PORT/health${NC}"
    break
  fi
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Backend crashed on startup. Run ./setup.sh to reinstall/verify.${NC}"
    kill $FRONTEND_PID 2>/dev/null
    exit 1
  fi
done

# ── frontend ─────────────────────────────────────────────────────────
echo -e "${YELLOW}Starting frontend (Next.js) on http://localhost:$FRONTEND_PORT${NC}"
cd "$FRONTEND_DIR"
if ! command -v npm &>/dev/null; then
  echo -e "${RED}Error: npm not found.${NC}"; kill $BACKEND_PID; exit 1
fi
if [[ ! -d node_modules ]]; then
  echo "installing frontend dependencies (one time) ..."
  npm install --no-audit --no-fund 2>&1 | tail -2
fi
npm run dev -- -p "$FRONTEND_PORT" &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend PID: $FRONTEND_PID${NC}"

echo ""
echo -e "${YELLOW}Backend API:  http://localhost:$BACKEND_PORT${NC}"
echo -e "${YELLOW}Frontend:     http://localhost:$FRONTEND_PORT${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop.${NC}"

# wait until either exits, then stop both
while kill -0 $BACKEND_PID 2>/dev/null && kill -0 $FRONTEND_PID 2>/dev/null; do
  sleep 0.5
done
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
echo -e "${GREEN}All services stopped.${NC}"
exit 0
