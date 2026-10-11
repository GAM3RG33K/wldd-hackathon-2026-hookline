# Hookline — demo cues

## Before
- Present from `localhost:8000/pitch`. No live run: slide 18 "The demo reels" links the three presentation videos' reports (`samples/presentation-videos/`, analysed in advance). Showcase and demo reels are set in the deck's front matter.
- On a report, the "Why viral potential" section can be explored live: factor switches (what-if), the calculation box, and the weights panel (applies to all reports; reset to defaults afterwards).
- Wake the GPU Space ~5 min early: `./run-demo.sh` → `S` → `s` (start). Header shows `space ● live`.
- Pick 2–3 reels with fresh full-TRIBE reports (TUI "saved reports", latest first): one emotional (FL041), one comedy (rajpsych_2), one product review.
- Browser tabs: `localhost:8000/reports` (sort dropdown: viral / latest / oldest), one `/report/<id>/view`.
- After the demo: `S` → `x` twice (pauses the Space, stops billing), or it sleeps after 30 min idle.

## 0:00 Problem
- Creators learn a video failed after posting.
- Hookline predicts second-by-second attention before posting; judges the video, never the account.

## 0:02 Live
- TUI: arrow to a reel; its saved report opens. `v` opens the web report. Enter = caption box, then analyze (Esc = just the report).
- Enter / `a` → caption box (optional, 5 lines; paste the post caption) → analyze on the GPU Space (progress, chunk, ETA live). More requests queue up; with the Space paused they wait and start once it is live.
- Caption demo: the product review with its real caption (+9.7) vs another reel's caption (ignored, +0).
- Own file: `o` opens the file picker, then the caption box. `m` run options (video / audio / transcript, language).
- Anything fails → `★ Demo report` in the tree, or any saved report.

## 0:06 Report
- Tiles: Hook, Hold, Finish, Viral potential + tag. Within-video heuristic, avg second ≈ 50, uncalibrated.
- Chart (30% height): attention, viewers remaining, hook window, drop zones, strongest window, cuts; emotion dots follow the playhead.
- "Why viral potential N": biggest boost and biggest drag, each factor's pull vs a typical reel.
- Emotion: predicted feeling of the viewer (scenes, faces, music, voice, words); strong peaks only when they spike. Audible laughter counted separately.
- Editing suggestions: timestamped, evidence vs the reel's own best part; `V` model-checks the top 3 on the GPU.
- Caption section (only if given): verdict, evidence, effect on viral potential.
- Transcript used for judging (Hinglish ASR; songs / humming skip the transcript).

## 0:11 Architecture
- TUI uploads → HF Space (A100) runs TRIBE v2 + ASR + speech gate + emotion channels → raw bundle → local backend builds the report.
- Formulas change locally: `g` regenerates a report from raw output, no GPU. See `ARCHITECTURE.md`.

## 0:13 Trust + close
- Validated on the hackathon reel set (outliers vs low-view); virality weights fit is a dry run (LOO 65%), not applied.
- Handles / followers = page-size context only, never inputs.
- Ask: try your reel.
