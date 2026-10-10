# 2026-10-10 — TUI own-video upload + Space/run scripts
- TUI: new [2] own video file → direct Space POST + poll, nothing stored locally (validates ext, drag-drop paths). [4] routes own files direct, samples via backend ?mode=hf. Menu renumbered (3 preview, 5 review, 6 health). Verified piped (bad-path rejection, run-demo launch, corpus counts 50/50/10).
- `space.sh`: secrets list, enforce t4-small, restart, wait, token-gated /health. `--status-only` verified: RUNNING, t4-small.
- `run-demo.sh`: execs tui.py. All chmod +x, sh -n + py_compile clean.
