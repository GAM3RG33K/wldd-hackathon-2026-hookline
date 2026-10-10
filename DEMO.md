# Hookline — demo cues

## Before
- `./run-demo.sh` (installs deps once, starts backend, opens TUI on demo report)
- Browser fallback tabs: `localhost:8000`, `/report/demo/view`
- Have `tribe/samples/sneaker.mp4` ready (not in git; copy in)

## 0:00 Problem
- Creators learn a video failed after posting
- Hookline previews attention second-by-second; judges video, not account

## 0:02 Live upload
- TUI: arrow to a reel, Enter → preview report in ~3s; `w` opens web view
- Own file: `o`, paste path; `f` = full TRIBE on GPU Space (minutes)
- Anything fails → `★ Demo report` in tree, or `/report/demo/view`
- Say: FastAPI front door, real ffmpeg signals now, full TRIBE pass runs on HF GPU Space, same JSON

## 0:06 Report
- Scores: Hook 61, Hold ~35%, Finish ~22%, Virality ~50; within-video heuristic, avg second ≈ 50
- Chart: strongest ~0:14, weakest ~0:10, drops 0:09–0:13 + end
- Recs: static open → cut/zoom first 2s; no early speech → payoff in second 1; drops → re-hook; weak end → payoff/loop

## 0:11 Architecture
- Video → events (audio+words) → TRIBE (T,20484) → neural + signal index → scores → report.json/md/png
- Local = preview; HF Space T4 = full neural; `POST /analyze → GET /report/{id}` schema fixed

## 0:13 Trust + close
- Validate on 110-reel corpus, fit CFG vs views within strata; handles/followers = strata only
- Ask: try your reel
