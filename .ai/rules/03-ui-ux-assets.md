# 03 — Report Surfaces (TUI, web report, md / png / json)

No Figma / Lottie / Miro sources exist. The surfaces below are the product; all read the same `report.json`.

## TUI (`tui.py`, Textual)
- Layout (fixed): header (backend status, run options `m`, Space stage `● live / ◐ starting / ○ paused / ✕ error`), left tree (demo, sample strata with "146k views · 50k–200k followers" per reel, your files, live on Space + "N queued here", saved reports latest first), right report pane, status line, footer.
- Report pane: 4 tiles (Hook, Hold, Finish, Viral potential + tag) → attention strip (30% height, emotion dots ●) → moments, viral-potential factors + summary, audio / music / emotion / strong-emotion lines (+ audible laughter) → suggestions table → transcript.
- Keys: Enter / `p` open saved report (never analyses), `a` analyze (asks for an optional caption first, prefilled with the last one; queues if busy; works on saved reports), `o` system file picker → analyze, `v` web report, `R` all reports, `g` regenerate, `V` model-check suggestions, `m` run options modal (`v a t l` inside), `S` Space modal (ping / start / restart / stop; stop + restart need a second press), `x` stop job or remove from queue, `ctrl+r` restart job, `r` random reel, `?` help, `q` quit.
- Progress view: step, substep, loop / chunk counters, ETA (chunk-based; "unknown" when no signal), speech-gate decision.

## Web report (`/report/{id}/view`)
- Order: title + file, speech gate / audio / song line, music line, tiles, video player + chart (SVG, 30vh; emotion dots with a playhead tooltip "time · emotion"), emotion band + strong line + emotion line, **Caption** (only when given: text, verdict, evidence, effect), **Why viral potential N** (summary, factor table with diverging bars, total), editing suggestions (verify button, badges, variant players), transcript used for judging, method / virality disclaimer, JSON / Markdown links, Regenerate.
- `/reports`: table (reel, run type, viral potential, hook, hold, finish, audio, length, analysed), sort dropdown (viral first default / latest / oldest), reloads every 5s except while the dropdown has focus. Lists only reports whose video exists and that are named after it.

## Markdown / PNG
- `report.md`: scores, moments, drop zones, "Why this viral potential" table, footer method line. `report.png`: score cards, attention + survival chart, network heatmap.

## Copy rules
- Never promise retention: show the method disclaimer near scores ("heuristic, within-video normalised, uncalibrated"; avg second ≈ 50).
- Emotions are "predicted, evoked for an average viewer; not a measured mood". Strong peaks are named by emotion (amusement, outrage, sorrow …), never "laughter" — audible laughter is its own line.
- Every suggestion keeps its `M:SS-M:SS` prefix. Followers / handles appear only as page-size context, never next to scores as a cause.
- New report fields: update `analyse()` / `finalize()` + md + web + TUI together.
