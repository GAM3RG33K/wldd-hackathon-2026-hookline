# 2026-10-10 — Working prototype for 15-min demo
- Goal: demoable prototype (upload → report in seconds), Colab as GPU upgrade, no local torch needed.
- Built: `backend/app.py` (FastAPI: `/`, `/health`, `/demo`, `POST /analyze`, `/report/{id}[./view/.md/.png]`, `POST /jobs/colab`), `backend/requirements.txt`, seeded `backend/out/demo/` from `tribe/out/demo_sneaker/`, `DEMO.md` script.
- Preview honesty: responses carry `preview:true, neural_source:synthetic-preview`; `report.json` schema identical to `hookline_report.py`; real signals (cuts/motion/dB) from uploaded video.
- Test: `/health` 200 demo_ready:true; `/demo` 200; upload sneaker → job `162ca8e4` in ~4s (hook 61.1/hold 35.4/finish 21.6/virality 50.3); fixed route-order bug (`/report/{id}` shadowed `.png/.md` → moved suffixed routes first, verified all 200).
- Verdict: 15-min demo ready at `localhost:8000` (server running). Full neural path = `tribe_colab.ipynb` T4 drop-in, no app change.
- Next: dry-run DEMO.md aloud; gitignore `backend/out/*` + `tribe/out/*`; add `POST /analyze?max_seconds` trim option to UI.
