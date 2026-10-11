title: Hookline
subtitle: From Video to Virality
author: Harshvardhan Joshi (harshvardhanjoshi@wldd.in)
team: WLDD Hackathon 2026 · Problem 01
showcase: OL003
problem: mohit-burad-review-79k-in-7hours
emotional: FL041
caption_example: Konsa colour bola? 😂 Police ask the shop owner the stolen phone's colour: fuchsia pink
caption_lift: 3.3

---

@title

# Hookline

## From Video to Virality

Note: Introduce the problem statement and yourself.

---

@cover

# Hookline

## See how a reel lands **before** it's posted

{{curve:showcase}}

Note: One line: Hookline predicts how a typical viewer responds to every second of a reel, explains the verdict, and tells the creator what to change and where.

---

@section The problem

# Creators find out too late

{{survival:problem}}

Note: A real reel from the hackathon set. The predicted share of viewers who reach the end is tiny, and the creator would only learn it after posting.

---

@section Our answer

# Every second, judged before posting

::: icons
eye | Seen, heard, said | one timeline
hook | Hook | first 3 seconds
drop | Drop-offs | where and why
heart | Emotions | what it evokes
gauge | Viral potential | with reasons
cut | Edits | at timestamps
:::

Judge the video, never the account.

---

@section How we got here

# Twelve problems, twelve answers

::: flow
No watch-time labels → Predict the viewer's brain response
Big pages bias views → Video-only inputs
Visuals, sound and words → One per-second timeline
Hinglish speech → Hinglish speech model
Songs aren't speech → Speech gate + song match
20-minute runs → Faster virality predictor
A score nobody can question → Breakdown: boost and drag
Tuning needed the predictor → Predictor gives evidence, server scores
Stories without words → Scenes, music, faces, voice
Emotions that weren't there → "No clear emotion" by default
Generic advice → Edits at timestamps, model-checked
Spam captions → Relevance test
:::

---

@section How it's built

# Three parts, one timeline

{{architecture}}

---

@section How it's built

# The Virality Predictor

{{model}}

Note: This is the only slide that names the underlying model. TRIBE v2 is a research model under CC BY-NC (non-commercial): fine for this prototype, a licence question for any product.

---

@section How it's built

# Models and tools behind it

::: icons
ear | Speech gate | AST · speech, song, humming, music | AST=https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
spark | Hinglish speech | Whisper Hinglish (faster-whisper) | Whisper Hinglish=https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex, faster-whisper=https://github.com/SYSTRAN/faster-whisper
eye | Scenes | CLIP · the situation on screen | CLIP=https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
heart | Faces | YuNet + ViT · expressions | YuNet=https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet, ViT=https://huggingface.co/trpakov/vit-face-expression
wave | Music mood | CLAP · and librosa rhythm | CLAP=https://huggingface.co/laion/clap-htsat-fused, librosa=https://librosa.org
heart | Voice tone | wav2vec2 · arousal, valence | wav2vec2=https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
check | Words | GoEmotions · emotion in speech | GoEmotions=https://huggingface.co/SamLowe/roberta-base-go_emotions
target | Caption check | multilingual MiniLM + CLIP | MiniLM=https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, CLIP=https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
clock | Song match | Chromaprint fingerprints | Chromaprint=https://acoustid.org/chromaprint
film | Cuts and motion | ffmpeg | ffmpeg=https://ffmpeg.org
laptop | User interface | Textual | Textual=https://textual.textualize.io
report | Server | FastAPI · reports, deck | FastAPI=https://fastapi.tiangolo.com
:::

---

@section From Video to Virality

# How a reel becomes a verdict

::: icons
film | Take apart | frames · audio · words
ear | Classify sound | speech · song · music
brain | Virality Predictor | brain response / second
gauge | Measure | 9 signal channels
wave | Fuse | attention 0–100 / second
target | Detect | hook · drops · peaks
check | Verdict + edits | with reasons
:::

---

@section From Video to Virality

# What we read from every second

::: matrix
Seen | frames · situations · faces · cuts · motion
Heard | sound type · music mood · voice tone · rhythm
Said | Hinglish words · meaning · emotion
Brain | Virality Predictor: 20,484 cortex responses
Caption | fits this reel? matches the speech?
:::

---

@section From Video to Virality

# The indicators, on a real reel

{{indicators:showcase}}

---

@section From Video to Virality

# Emotion that is predicted

{{emotionband:emotional}}

::: bars False "amusement" on a product review (share of seconds)
Before: forced choice | 84 | 84%
*After: "no clear emotion" by default | 14 | 14%
:::

---

@section From Video to Virality
@report showcase

# A caption counts only if it fits

::: examples OL003: two cops, a stolen phone, "which colour?"
good | Konsa colour bola? 😂 Police ask the shop owner the stolen phone's colour: fuchsia pink | frames z 6.9 · speech z 3.7 · +3.3
good | Cop asks which phone and colour, X phone 15 plus pro max in fuchsia pink 😂 #comedy | frames z 6.5 · speech z 4.8 · +3.3
bad | This scene never gets old 😂 | z 1.0 · too vague · ignored
bad | follow for more #viral #reels #trending | z −1.2 · filler · ignored
bad | Best skincare routine for glowing skin | z 0.2 · another topic · ignored
:::

A caption counts from z 1.6 on the frames or the speech, and fully at z 3.0. It must describe what is seen or said.

---

@section The verdict

# Viral potential, with its reasons

{{whybars:showcase}}

---

@section See it

# One reel, end to end

{{seeit:showcase}}

---

@section Does it work?

# Checked against real views

{{scatter}}

Note: The weights are hand-set; the ranking is only weakly aligned with views so far. The next step is fitting on reels with known outcomes.

---

@section What a creator does next

# Edits, placed on the timeline

{{edits:showcase}}

---

@section Live

# Your reel, live

::: icons
film | Your reel | + caption
laptop | Pick it | o · caption · Enter
clock | Watch it run | ≈ 2.5 s / second
report | Read the verdict | v opens the web report
:::

{{launch}}

---

@section Next

# Where it goes next

::: icons
gauge | Tweak model and formula | to be more accurate
brain | Understand and predict | complex human emotions, the "vibe" of the frame
spark | One tool to predict them all | for any creator
:::

---

@closing

# Thank you

## Bring us a reel. We'll show you how it lands.
