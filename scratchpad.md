# HARNESS SCRATCHPAD - Hookline (wldd-hackathon-2026)

## Step1 Triage - DONE
- No Excel. No Figma/Jira/Miro links. No Lottie. No videos CSV in repo.
- Found: hookline-problem-statement.md (judge video not account, 110 reels, no labels), tribe/ prototype.
- tribe/ files: tribe_analyze.py (TRIBE v2 infer -> preds.npy/times.npy/meta.json), hookline_report.py (heuristic scores -> report.json/md/png), tribe_colab.ipynb (T4 GPU, HF_TOKEN, upload->install->patch->analyze->report), prefetch.py, fast_transcript_patch.py, samples/sneaker.mp4, repo/ = facebook tribev2 clone.
- Stack: Python, torch, tribev2 (facebook/tribev2 + w2v-bert-2.0 + vjepa2 + Llama-3.2-3B), faster-whisper/whisperx, ffmpeg/ffprobe, numpy/pandas/matplotlib/nilearn, Colab T4.
- Backend app: PENDING_IMPLEMENTATION. Colab-as-backend: PENDING (no API/bridge yet).
- Tool caps: read local OK. No Figma MCP. Only mcp-video templates_resource. No Excel to read. No hallucinate.

## Step2 Business Logic - EXTRACTED
- Pipeline: video in -> get_events_dataframe (audio+word transcript) -> TribeModel.predict (T,20484 avg-subject, fsaverage5, -5s hemodynamic lag) -> neural_series (Destrieux groups: visual/auditory/language/attention - 0.5*default_mode) -> S signals (cuts scene0.3, motion 5fps 64x64 gray, audio dB silence -45, words/sec) -> A=100*sigmoid(1.2*zs(smooth(0.6N+0.4S))) -> hook (first 3s, 0.5mean+0.5max+bonus speech<=1.5s +0.3, cut/move<=2s +0.3, silent>1s -0.4, squash k1.3) -> hold/surv (floor .93 span .065, hook interp first 3s) -> virality=0.45hook+0.30hold+0.25avgA.
- Bands: <30 Bad, <50 Weak, <70 Average, <85 Strong, else Viral-grade. Within-video norm (~50=avg sec). Uncalibrated heuristic.
- Moments: peak_hook (max 2s rise+1), peak_attention (argmax A), strongest/weakest 3s window, drops (A<35 >=2s, else weakest).
- Recs: timestamped, rule-based (0:00-0:03 opening, buried peak, drop zones with why: no change/low move/flat audio/no speech, weak ending last 3s<40).
- Outputs per video-stem/: preds.npy, times.npy, meta.json, report.json/md/png. Series: attention, survival_pct, cuts_s.
- Models: visual=w2v-bert? audio=w2v-bert-2.0, video=vjepa2-vitg-fpc64-256, text=Llama-3.2-3B, brain=Transformer FmriEncoder.
- Naming: tribe_analyze.py, hookline_report.py, out/<stem>/, cache/, scripts/ in colab. CLI: videos + --max-seconds --out --device cpu|mps|cuda.
- Errors: missing video->exit1, atlas fail->fallback whole-cortex norm, empty transcript->zeros, short audio-> -80dB.
- Journeys: (1) creator upload->analyze->report (colab now, backend later), (2) judge batch: 110 reels validate/fit weights vs view counts.
- Judge axes: architecture, signals considered, method choice, validation vs real performance, specific recs.

## Step3 Harness - DONE (.ai/rules + FastAPI)
- Verify: tribe_analyze --help OK; hookline_report --help blocked by missing numpy (env, not code); py_compile OK for tribe_analyze/hookline_report/prefetch. fast_transcript_patch.py is Colab-cell snippet (!pip) - not importable by design.
- Files: 01-business-logic.md, 02-tech-stack.md, 03-ui-ux-assets.md (+ backend-colab bridge rules).
- Constraints: judge video not account, no follower graft, within-video scores, heuristic label, no Figma/Lottie yet.
