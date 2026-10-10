title: Hookline — see how a reel lands before it's posted
showcase: mohit-burad-review-79k-in-7hours
emotional: FL041

---

# Hookline

## See how a reel lands **before** it's posted

Second-by-second attention · hook · hold · viral potential with reasons · emotions · timestamped edits

Judge the video, never the account.

Note: 15 s. One line: we predict how a typical viewer's brain responds to every second of a reel, then turn that into edits a creator can make today.

---

# The problem

- Creators learn a reel failed **after** posting: the viewers who scrolled away at second 2 are already gone.
- No tool shows, before publishing, **where** viewers are pulled in, where attention fades, and **why**.
- Script-only feedback misses most of it: a reel lands through what is **seen, heard and said together, over time**.

Note: Problem statement in our words. Point at "why": every number in our report has a reason attached.

---

# What Hookline gives a creator

{{scores:showcase}}

- **Hook / hold / finish** and attention for every second, with drop zones and the strongest stretch
- **Viral potential** with a breakdown: the biggest boost and the biggest drag
- **Emotions** the reel evokes, and the strong peaks (sorrow, outrage, amusement, awe …)
- **Editing suggestions** at exact timestamps, each backed by evidence from the reel's own best part
- **Caption check**: used only if it really describes this video

Note: Read the tiles out loud for the showcase reel. Then click through to the full report on the next slide.

---

# One reel, end to end

{{report:showcase}}

Note: Scroll the embedded report: tiles, chart with emotion dots (play the video, the playhead tooltip follows), caption section, "Why viral potential", suggestions, transcript.

---

# How it's built

{{architecture}}

- **TUI** (Textual): pick a reel or your own file, add a caption, live progress with ETA, queue, stop / restart, Space control.
- **GPU Space** (Hugging Face, A100): runs every model and returns **raw data only**.
- **Local backend** (FastAPI): builds every report from the raw data. Formula changes re-score all reels in seconds, no GPU.

Note: Judging criterion 1. Emphasise the split: GPU produces evidence, scoring is transparent local code.

---

# What we look at

| | What | Model |
|---|---|---|
| **Seen** | frames over time · scene situations · faces · cuts and motion | V-JEPA2 · CLIP · YuNet + ViT expressions · ffmpeg |
| **Heard** | audio over time · speech vs song vs humming · music mood · voice tone · rhythm | Wav2Vec-BERT · AST · CLAP · wav2vec2 · librosa |
| **Said** | Hindi / English / Hinglish words with timestamps · meaning · emotion | faster-whisper Hinglish · Llama-3.2-3B · GoEmotions |
| **Brain** | predicted response of 20,484 cortex points per step | **TRIBE v2** (Meta) |
| **Caption** | does it describe this reel? | CLIP vs frames + 55 other reels · multilingual MiniLM |

Songs and humming are judged on visuals + sound, not lyrics. Every signal is per second, on one timeline.

Note: Criterion 2. Why each: TRIBE predicts how a typical brain responds to the combined stimulus, so seen + heard + said are fused by a model trained on real brain recordings, not by hand-written rules.

---

# Method: from brain response to a score

1. **Attention per second** = 60% brain engagement (sensory + language + attention regions minus the mind-wandering network) + 40% edit signals (cuts, motion, loudness, words/s). Normalised within the video: the average second ≈ 50.
2. **Hook** = first 3 s (+ speech early, + a cut or movement, − silent start). **Hold / finish** from a simulated viewer curve.
3. **Emotion**: 20 emotions (Cowen & Keltner) from scenes, faces, music, voice, words, sounds. **No clear emotion** unless a channel is clearly above its usual level.
4. **Viral potential** = 40% hook + 25% few drop-offs + 20% strong moments + 15% ending, lifted by strong emotional peaks and a relevant caption.

{{why:showcase}}

Note: Criterion 3. Why TRIBE over a classifier trained on views: no watch-time labels exist; a brain-response model gives moment-level signal without labels. Why heuristics on top: transparent, re-scorable, honest about being uncalibrated.

---

# Emotion that is earned, not forced

{{emotion:emotional}}

- Fixed-list classifiers always pick *something*. We abstain: in the product review, false "amusement" fell from **84% to 14%** once each signal had to beat its own usual level.
- Faces are read only when big and confident, against the person's resting face.
- Audible laughter is counted separately from predicted amusement.

Note: Tell the FL041 story: a rescued elephant, no speech, mournful music: sorrow peaks found from music and scenes alone.

---

# How we know it works

{{validation}}

- **Known cases** re-checked after every scoring change: emotional (FL041), performed comedy (rajpsych), product review.
- **Edits are model-checked**: each suggestion is rendered with ffmpeg and re-run through TRIBE against a no-edit baseline (re-encoding alone shifts engagement ~5%).
- **Captions**: borrowed / filler captions ignored (z ≤ 1.5), true ones matched (z 2.3–3.9) on real reels.

Note: Criterion 4. Be honest: the virality weights are hand-set and the logistic fit (leave-one-out 65% on 20 reels) is a dry run, not applied. That is the next step with more labelled reels.

---

# Recommendations a creator can act on

{{suggestions:showcase}}

- Each one names **one edit at one timestamp**, with the evidence against the reel's own best stretch.
- Ranked by estimated viewers lost. `V` renders the edit and re-runs it on the GPU to check it really helps.

Note: Criterion 5. Point at the evidence column: it compares the weak part with this reel's own best part, so advice is specific to the video.

---

# Live demo

1. **Give us a reel** (and its caption, optional).
2. TUI: `o` → pick the file → paste the caption → **Enter**.
3. Watch the live progress: steps, chunks, ETA (~2.5 s of GPU per video second).
4. The report opens: tiles, chart, emotions, caption verdict, why-viral, edits. `v` opens it in the browser.

All reports: **/reports** · this deck: **/pitch** (edit `pitch/deck.md`, refresh)

Note: Before the demo: wake the Space (TUI S, then s) about 5 minutes ahead; it sleeps after 30 minutes idle. Fallback: the showcase report on slide 4.

---

# Limits and next

- Scores are **within-video and uncalibrated** — model estimates, not measured retention.
- Humour and abstract captions need meaning, not pattern matching: next is a video-language model ("what does this evoke, if anything?").
- Fit the virality weights on reels with known outcomes; per-niche calibration.
- Same pipeline, more creators: the GPU step is a single API.

# Thank you

Note: Close with the one-liner and invite the judges to bring their own reel.
