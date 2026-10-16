#!/bin/sh
# Hookline demo: installs deps on first run, then opens the terminal UI.
# The TUI starts the local backend itself if it is not already running.
cd "$(dirname "$0")"
python3 -c "import textual, fastapi, uvicorn" 2>/dev/null \
  || python3 -m pip install -q -r backend/requirements.txt
exec python3 tui.py
