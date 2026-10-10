# Emotion — channels, limits, failure cases (sources: `tribe/emotion_fusion.py`, `tribe/taxonomy.py`, `hf_space/emotion.py`, session log 2026-10-10/11)

## Model of the problem
- Report the emotion a video EVOKES in an average viewer (stimulus-driven), plus what is PORTRAYED on screen (faces). A performed meltdown can portray rage while evoking amusement.
- Taxonomy: 20 categories after Cowen & Keltner 2017; appraisal-theory scene prompts (the situation, not the look). Sharing literature: arousal > valence (Berger & Milkman 2012), peak-end (Kahneman), story arcs (Reagan et al. 2016).

## What each channel can see
| Channel | Sees | Blind spots found |
|---|---|---|
| CLIP scene prompts | situations (rescue, injury, food, prank) | "hub" prompts ("a funny prank or blooper") top every reel; can't see expressions or jokes |
| Faces (YuNet + ViT FER2013) | crying, anger, fear, smiles on the biggest face | per-frame jitter (3s smoothing); no face → no signal; FER2013 is posed-expression data |
| CLAP music mood | soundtrack mood | invents a mood for plain speech (dropped when the gate hears no music); a quirky bed under talk is not the content's emotion (half weight) |
| Voice (audeering) | arousal / valence of speech | cannot separate emphasis from anger (product CTA ≈ rage signature) → corroboration only |
| GoEmotions text | emotion words | English-only; Hinglish → mostly neutral / spurious amusement (half weight, neutral → calm) |
| AST sounds | audible laughter, cheering, gasps, crying sounds | rarely fires on reels (0s laughter on every reel so far) |

## Failure cases → fixes (all verified by re-scoring)
- Product review (mohit-burad) "laughter 34s, strength 100": music bed + dropped "neutral" → music bed weight, neutral → calm, text trust. 2026-10-11.
- Deadpan monologue (rajpsych_2) "laughter 31s, sorrow 5s": CLAP on speech with 0% music + backend running stale modules → `has_music()`, rescore reloads all scorer modules; label "laughter" → "amusement" + separate audible-laughter count. 2026-10-11.
- Same reel, performed crying / rage not detected → face channel; result sorrow 0:28, 1:01, outrage 1:07, product review stays clean. 2026-10-11.
- Rejected: 2-channel agreement for strong peaks (killed FL041's music-led sorrow); per-reel prompt z-scores (false fear / disgust).

## Caption relevance (2026-10-11)
- Fixed control captions alone fail on frames: CLIP rewards anything that fits "a person in a reel" ("follow for more #viral #reels" z 2.67; another reel's caption z 2.39). Specificity vs a gallery of other reels separates them: true captions frames z 2.3–3.9, generic / borrowed / unrelated ≤ 1.51. Transcript similarity (multilingual MiniLM) catches Hinglish captions (true z 3.14 vs ≤ 1.05). Rule: stronger z, 0 at 1.6, full at 3.0.
- Miss: abstract true captions ("betrays you before navratri") — neither test sees betrayal.

## Forced choice → no clear emotion (2026-10-11)
- Audit of 36 reels before: calm won 0% of seconds (median), amusement topped 28/36 reels, strong "sorrow" in 12/36 (faces read talking faces as sad; sad = top label in 17% of face frames).
- Rejected: absolute ramped z alone (music-only reels have every music prompt "above usual" vs a speech-heavy corpus; "scary horror" beat "mournful" on FL041 through a tiny corpus std) and fused-probability dominance (abstaining channels outvoted the one clear channel → FL041 99% no emotion).
- Kept: raw ranking × calibration gate, single-channel clear evidence ≥ 0.30, face gates (area ≥ 1%, conf ≥ 0.6, 70% resting-face baseline), text labels gated the same way (rare GoEmotions labels had exploded to 0.83).
- Check reels: FL041 sadness 59% / no clear 29% / sorrow peaks kept; rajpsych amusement 35%, sadness 21%, no clear 37%, sorrow 0:28 + 1:01, outrage 1:07; product review satisfaction 49%, amusement 84% → 14%, no clear 35%, no strong peaks. Confidence "low" on all three (evidence mostly from one channel at a time).

## Open
- Humour / jokes need semantics (Hinglish-capable LLM over transcript, or a VLM story layer, e.g. Qwen2.5-VL-7B).
- Prompt calibration activates at ≥ 15 reference reels (`calibrate_prompts.py`); rebuild after the full re-run.
