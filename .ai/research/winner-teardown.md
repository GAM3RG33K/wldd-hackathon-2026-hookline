# Winner teardown — mindMatrix (won) vs hooklinewldd-ttcc (special mention), against Hookline 2026

Sources (all local, no guessing):
- `.local/wlddhq:mindMatrix/app.py` — Gradio Space, Meta TRIBE v2 brain response (~717 lines, one file).
- `.local/wlddhq:hooklinewldd-ttcc/{app.py,ttcc_infer.py}` — Docker + FastAPI job API, TTCC retention head (~229 + ~470 lines).
- Ours: `tribe/*`, `hf_space/*`, `backend/app.py`, `tui.py`, `.ai/rules/*`.

Neither winner is referenced anywhere in this repo (grep for `ttcc|mindmatrix|go_viral` = no hits outside `.local/`), so they are separate submissions, not ancestors of a branch.

## What the winners actually did

### mindMatrix (winner) — "predict the human brain's response to your reel"
- **Model:** `facebook/tribev2`; transcribes the video itself (`model.get_events_dataframe`, `app.py:574`), predicts `(T, 20484)` cortical activity (`app.py:538`), reduces it to a per-second score `curve = clip(50 + 15*z, 0, 100)` from `strength = |preds|.mean(1)` (`app.py:115-118`).
- **Scores:** hook = first 3s vs the rest of *this* video; ending = last 20% vs average; dead zones = ≥3 s runs below −0.8σ averaging ≤ −1σ; top-3 maxima = peaks (`app.py:127-150`).
- **The fix layer cost no GPU at all** — ffmpeg only: per-second loudness dB + motion (`app.py:175-203`), scene cuts via `select='gt(scene,0.35)'` (`app.py:209-219`), `blackdetect` / `freezedetect` (`app.py:309-328`), `frame_has_text()` so a designed text card on black is *not* flagged (`app.py:331-340`), and flash suppression — ≥3 short blacks = style, not a bug (`app.py:366-369`).
- **Issue list, not a score dump:** hook / ending / drop-off / long shot ≥8 s / black at start / black at end ("reels loop, so this breaks the loop") / frozen end / frozen mid / silent opening / dead air — each with a time range, what, and a concrete fix, **plus a "✅ What's working" section** (`app.py:413-485`, `app.py:472-483`).
- **Zero-friction demo:** `gr.Interface` on a public Space; upload → chart + markdown + JSON + raw `.npz` for offline calibration (`app.py:703-714`, `app.py:631-633`).
- **Silent robustness that saved the live demo:** 480 p/16 fps pre-transcode (`app.py:554-566`), Whisper + LLaMA pre-cached at boot (`app.py:54-70`), ZeroGPU-aware `@spaces.GPU` moving the model to CUDA and back (`app.py:529-545`), trailing segments past the real duration trimmed (stimulus-cutoff artifact, `app.py:583-593`), and a Gated-Repo error page that tells the viewer which secret to add (`app.py:678-692`).

### hooklinewldd-ttcc (special mention) — "here is the retention curve of your reel"
- **Model:** `liangyuch/ttcc-sft-qwen25omni-3b-v8`, Qwen2.5-Omni-3B SFT from `cliangyu/go_viral@ttcc-rl` with a retention head; one forward pass → `r_pred` shape `(1, 60)` = predicted audience-retention curve R(1..60) (`ttcc_infer.py:17-19`, `ttcc_infer.py:440-449`).
- **Deliberately raw-only** — this is already our rule 5 ("the Space produces raw data; the local backend produces reports").
- **Integrity as a feature:** pinned repo + commit SHA (`Dockerfile:13-14`), expected file sizes + sha256 (`ttcc_infer.py:93-113`), retention-head check against the checkpoint in memory (`ttcc_infer.py:269`), runner version + runner sha256 + git state + library versions in `probe` (`ttcc_infer.py:86-90`, `ttcc_infer.py:208-260`), and requests for a different checkpoint revision are rejected (`app.py:135-137`).
- **Engineering guardrails:** 202 + poll (`app.py:206-227`), 503 while the checkpoint downloads (`app.py:210-211`), 413 body cap (`app.py:212-213`), timeouts, job store capped at 50, `timing: {queued_s, run_s}` on every response (`app.py:99-100`, `app.py:47-51`).
- **Batching:** many 30/15 s windows of one upload share a single model load, with `shrink_to_fit` / OOM-window-shrink because attention memory grows with the square of the input (`app.py:44`, `app.py:141-180`, `ttcc_infer.py:141-161`, `ttcc_infer.py:416-439`).

## Verdict — what we have vs what won

| Capability | mindMatrix | ttcc | Hookline 2026 |
|---|---|---|---|
| Public, no-token, no-install demo surface | yes (`app.py:703-714`) | n/a (job API) | **no** — TUI/backend + private GPU Space, HF token (`README.md:10-12`, `hf_space/README.md:1-4`) |
| Black screen / frozen frame / loop-break defects | yes (`app.py:309-388`) | – | **no** (grep `blackdetect\|freezedetect` in `tribe/` = 0 hits) |
| "What's working" positives | yes (`app.py:472-483`) | – | **no** (grep `what.s working\|positives` in `tribe/ backend/ tui.py` = 0 hits) |
| Error path that tells the viewer what to do | yes (`app.py:678-692`) | – | **no** — Space paused = silent TUI queue |
| Provenance of a scored report | raw `.npz` only | **strong** (`ttcc_infer.py:86-90`) | **no** (grep `provenance\|rev-parse` = 0 hits) |
| Cross-video comparable raw stats for calibration | `raw` block (`app.py:164-171`) | raw bundle | `features` block (`hookline_report.py:370-388`) + `fit_virality.py` |
| Per-second signals + cut/motion/loudness mining | yes | – | yes (`hookline_report.py:171-202`, `suggest.py:49-67`) |
| Ranked impact (viewers lost) in the report | implicit severity order | – | in TUI + web only (`tui.py:1033`, `backend/app.py:611`); **missing in `report.md` / `report.png`** (`hookline_report.py:601-628`, `:636-674`) |
| Multimodal depth (ASR, speech gate, song, rhythm, emotion, caption check) | ASR only | audio+video (Qwen-Omni) | **ahead of both** |
| Suggestions that are model-checked | no | – | **ours only** (`tribe/suggest.py`, `V` in TUI) |
| Survival model from attention | generic decay (`app.py:283-294`) | real retention curve (different model) | **ahead** (`hookline_report.py:307-316`) |
| Honesty of method strings | present | present | present (rule 2) |

## Cherry-pick

### Do first (before judging; no GPU, no scoring change, all re-scoreable)
Each item is independently shippable, uses the existing verification loop, and touches surfaces named in `.ai/rules/03-ui-ux-assets.md` together.

| # | Item | Where | Why first | Effort |
|---|---|---|---|---|
| **P1** | **Public read-only demo surface.** Serve 3–5 already-scored reports (sneaker strong, FL041 sorrow, product review flat, rajpsych_2 performed sorrow/outrage) as a public Space or static page with the video players and the existing chart/sections. The GPU Space stays private. | new `demo/` or a small public Space; reads `backend/out/*/report.json` | A judge sees the product in ~10 s with no install/token. Today the demo requires `./run-demo.sh` + token + a private Space. This is the single largest gap. | ~half day |
| **P2** | **Frame-level craft defects.** New `tribe/gaps.py`: `blackdetect` + `freezedetect` + text-card guard + flash suppression → issues "starts on a black screen", "black frames at the end (reels loop — this breaks the loop)", "ends on a frozen frame", "frozen picture", "dead air", "silent opening". Feed into `suggest.py` and the chart bands; reuses `series.loudness_db` / `series.motion`. | `tribe/gaps.py` (new), `tribe/suggest.py`, `hookline_report.py` | Highest signal-per-line of the missing detectors: no GPU, `tribe/rescore.py` re-scores every past reel instantly, and these are the defects judges can see with their own eyes on the video. | ~1 day |
| **P3** | **"What's working" positives block** on all four surfaces (report pane → md → web → PNG), from thresholds that already exist: no drop zones, strong hook, strong ending, peak density, cuts/10 s, strong emotion present. | `hookline_report.py` (`write_md`, `write_png`, `analyse`/`finalize`), `backend/app.py`, `tui.py` | A report that only lists problems reads like a bug tracker; winners showed successes. Pure presentation, zero scoring risk. | ~half day |
| **P4** | **Actionable error hints** in `backend/hf_client.py` / `tui.py` / web report: Space paused → "press `S` → start", no token → `hf auth login`, 503 → "checkpoint/video still preparing, retry". | `backend/hf_client.py`, `tui.py`, `hf_space/app.py` | Removes dead air in the demo; mindMatrix's gated-repo page is the direct precedent (`app.py:678-692`). | ~2 h |
| **P5** | **Show ranked impact in `report.md` / `report.png`** — the `viewers_lost_pct` that already exists in JSON, in the punch list, sorted by it. | `hookline_report.py:601-628`, `:636-674` | One-liner; makes the exported report match the TUI a judge sees on screen. | ~1 h |

### Add later (each needs a verification run before it is promoted to rules, per `.ai/research/README.md`)

| # | Item | Why not first | Depends on |
|---|---|---|---|
| **L1** | **"Hook vs a typical reel" with real baselines.** mindMatrix used `TYPICAL_HOOK_VS_REST_PCT = -8.0`, a median of 7 reels, labelled "recalibrate after the batch" (`mindMatrix/app.py:75-79`). Ours must come from stored reports: per-stratum baseline of `features.hook_lift` / `hook` over the 45+ reels in `backend/out`, shown with n and tagged uncalibrated. | The number must be *derived*, not copied, or it repeats their uncalibrated hack; `fit_virality.py`'s old LOO 65% leaked (selection before CV) and the corrected validation shows the score is not yet distinguishable from chance (`.ai/research/validation.md`). | P3 surfaces; `fit_virality.py` re-run on the full batch |
| **L2** | **Provenance block in every report:** Space job id, TRIBE checkpoint rev, ASR model, emotion model set, scorer version + `git rev-parse HEAD` at score time. | Pure addition; only valuable once the demo surface (P1) exists to display it. | P1 |
| **L3** | **Operational parity:** `timing {queued_s, run_s}` on `/jobs/{id}`, upload size cap, documented max video length + shrink-to-fit fallback (TTCC: `app.py:99-100`, `:44`). | Our A100 chunking + ETA already works; these are polish, not demo-critical. | – |
| **L4** | **Optional second opinion from the TTCC curve.** The `hooklinewldd-ttcc` Space exists but is not wired in. Adding a second, differently-shaped retention curve risks contradicting our own survival model and confusing the judge. | Keep as a research-only "second-opinion channel" on 3 reels; only show if it agrees. | Needs an agreement test before any UI copy |

## Do NOT copy (we already do it better)
- `estimate_retention`'s generic short-form decay (`mindMatrix/app.py:283-294`) — our `hold_floor`/`hold_span` survival model is attention-driven (`hookline_report.py:307-316`). Do not regress to a fixed decay.
- The median-of-7 `TYPICAL_HOOK_VS_REST_PCT` constant as a public number — replace with L1 (fitted, tagged uncalibrated) or drop it.
- "Raw-only Space" — already rule 5; nothing to add.
- Single-file monolith Gradio app — our Space is FastAPI on purpose (rule 5: reports built locally); keep it.

## Risks / guardrails while implementing P1–P5
- P2 must keep mindMatrix's two false-positive guards: text-card-on-black (`frame_has_text`) and repeated-flash-as-style, or reels with intentional black cuts get bogus "defects" and trust in the report dies.
- No scoring change in P2–P5 unless L1 lands; `method` strings stay `heuristic, within-video normalised, uncalibrated` (rule 2). Black/freeze findings are *frame facts*, not scores.
- P1 is read-only and baked: no public GPU endpoint, no tokens, no account data (rule 1). Verify with the AGENTS.md grep (`follower|handle` as signal) on every new surface.
- Promotion rule: P1–P5 findings go into `.ai/rules/01-business-logic.md` / `03-ui-ux-assets.md` only after the logged verification run (`uvx pyflakes`, `tribe/rescore.py` before/after on FL041 / product review / rajpsych_2, backend restart, headless Textual `run_test` with the GPU upload stubbed).
