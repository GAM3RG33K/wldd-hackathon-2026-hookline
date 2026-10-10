#!/bin/sh
# Hookline Space setup: RUNNING -> restart, anything else -> bring up + verify.
# Usage: ./space.sh [--status-only]
set -eu
SPACE="${HOOKLINE_SPACE_ID:-wlddhq/hookline-tribe}"
BASE="${HOOKLINE_SPACE_URL:-https://wlddhq-hookline-tribe.hf.space}"

stage() { hf spaces info "$SPACE" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['runtime']['stage'])"; }

if [ "${1:-}" = "--status-only" ]; then
  echo "stage: $(stage)"
  hf spaces info "$SPACE" 2>/dev/null | python3 -c "import json,sys; r=json.load(sys.stdin)['runtime']; print('hardware:', r.get('requested_hardware'))"
  exit 0
fi

echo "== secrets =="
hf spaces secrets list "$SPACE" 2>/dev/null | head -n 5 || true
echo "== hardware (want t4-small) =="
hw=$(hf spaces info "$SPACE" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['runtime'].get('requested_hardware','?'))")
echo "requested: $hw"
[ "$hw" != "t4-small" ] && hf spaces settings "$SPACE" --hardware t4-small

echo "== stage: $(stage) -> restart =="
hf spaces restart "$SPACE"
echo "== waiting for RUNNING =="
hf spaces wait "$SPACE" --timeout 1500s | tail -n 2
echo "== health =="
if [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN not set, skipping API check"; exit 0; fi
curl -s -H "Authorization: Bearer $HF_TOKEN" "$BASE/health"; echo
