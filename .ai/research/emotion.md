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

## Open
- Humour / jokes need semantics (Hinglish-capable LLM over transcript, or a VLM story layer, e.g. Qwen2.5-VL-7B).
- Prompt calibration activates at ≥ 15 reference reels (`calibrate_prompts.py`); rebuild after the full re-run.
