# Validation — does the score predict outcomes? (2026-10-14)

Tooling: `tribe/validate_virality.py` (AUC + permutation test + per-channel ablation on
held-out folds). Runs on stored reports + `samples/*/metadata.csv`; no GPU. Views and
follower bands build the *label only*, never a feature (rule 1).

**Dataset:** 81 labelled full-TRIBE reels with view counts — 10 viral outliers (10M–315M
views) vs 71 low-view (2.3k–4.9M), split 29 face / 42 faceless.

## The headline: the score is not yet distinguishable from chance

| Statistic | Value | What it means |
|---|---|---|
| Shipped heuristic, AUC outlier-vs-low | **+0.615** | better than chance, not by much |
| bootstrap 95% CI on that AUC | [+0.434, +0.801] | **includes 0.5** |
| permutation p (2000 label shuffles) | **0.135** | 13.5% of shuffles beat it |
| Spearman vs log10(views) | +0.178 | weak ranking signal |

On 10 outliers the error bars are wide enough to hide almost any truth. The honest
statement to a client is *"directionally positive, not yet proven"* — not a number.

## Fitting does not help, and makes ranking worse

Repeated 5-fold × 30 with feature selection **and** standardisation inside each fold:

- held-out AUC all channels **+0.674** (p=0.080) — a +0.06 move, within noise
- but Spearman vs log10(views) **−0.118**, worse than the heuristic's +0.178

A fit at this n chases the 10 outliers and loses the ranking. `virality_model.json`
stays unwritten; the hand-tuned weights stay.

## Per-channel ablation — nobody earns removal

Drop a channel, re-fit, re-test on the *same* folds:

| Removed | held-out AUC | Δ | folds it helped |
|---|---|---|---|
| attention | +0.755 | +0.081 ± 0.043 | 1/30 |
| ffmpeg | +0.679 | +0.006 ± 0.031 | 12/30 |
| music | +0.690 | +0.016 ± 0.036 | 9/30 |
| asr | +0.674 | 0.000 ± 0.000 | 0/30 |
| emotion | +0.674 | 0.000 ± 0.000 | 0/30 |

- **emotion delta is exactly 0** — no emotion feature is ever selected. This is
  independent confirmation of the 2026-10-14 ablation that set `emotion_lift` to 0.
- **ASR delta is exactly 0** too — `words_per_s` is never selected *as a fitted feature*,
  but it stays in the pipeline because it feeds `words_ps` and the hook bonus in the
  attention score (`.ai/research/scoring-heuristic.md`). Different question.
- **attention** removal *raising* AUC is the interesting one, and it points at a concrete
  suspicion — see below.

## Which individual features actually separate (and what the weighting ignores)

Effect size (Cohen's d, outliers vs low-view) on the full set — descriptive, ~23 features
tested so treat as suggestive, not confirmatory:

| feature | d | direction |
|---|---|---|
| `peaks` (count of strong moments) | **+0.95** | outliers have ~2× the peaks |
| `peaks_per_window` | +0.87 | same signal, normalised |
| `hook` | +0.63 | outliers hook harder |
| `volatility` | −0.66 | outliers are *less* volatile |
| `peak_gap_cv` | −0.61 | outliers' peaks are evenly spaced |
| `drop_share` | +0.24 | outliers drop *more* → drop-free points backwards |
| `peak_share` (share of seconds 70+) | −0.20 | backwards |
| `ending` | −0.11 | backwards |

The shipped weighting (hook .30 / drop-free .25 / strong-moments .30 / ending .15) spends
**40% of its weight on drop-free + ending, which point backwards on this set**, and its
"strong moments" factor is `peak_share` (d=−0.20) rather than `peaks` (d=+0.95).

Re-weighting to hook + peaks raises AUC 0.623 → 0.661 and the top-10-vs-bottom-10 gap
from +6.8 to +13.1 points — **but the bootstrap CI on that difference is [−0.079, +0.114]
and the re-weighting wins only 65% of resamples.** Not significant, so no weight changes
are made. This is the first thing to re-test once more outliers are labelled.

## The extremes do separate — marginally

The product's real claim is "which of these will outperform", not a fine ranking:

| comparison | virality gap | p |
|---|---|---|
| top-10 vs bottom-10 by views | +6.8 pts | 0.061 |
| top-15 vs bottom-15 | +3.1 pts | 0.299 |
| top-20 vs bottom-20 | +3.6 pts | 0.153 |

Closest thing to a positive result, and it decays as the comparison widens.

## The framing an agency actually uses — and it looks best

They don't ask "is this viral", they ask "which of *this client's* reels should we boost".
That's within-account pairwise, which cancels account size without ever using it.
Only 6 accounts have 2+ scored reels: **4/4 pairs called correctly, p=0.062.**
Promising, and the single most useful validation to build out — see "What would settle it".

## What would settle it

Power calculation (AUC vs 0.5, α=.05, ~80% power, 71 low-view reels held):

- detect AUC **0.70** → ~10 outliers (have 10 — so a true 0.70 would just reach significance)
- detect AUC **0.65** → ~19 outliers
- detect AUC **0.60** → ~58 outliers

**Two concrete paths, both cheap:**

1. **More outliers.** ~10 more reels that broke out (10M+ on small accounts) takes the set
   to 20 and makes AUC 0.65 detectable. Label them from public view counts; no new GPU cost
   to label, one Space run each to score.
2. **Within-account pairs.** ~30 accounts × 2 reels gives ~90 pairs, where 60% accuracy is
   detectable. This is also the framing that sells — it answers the question they actually
   ask.

## Fixed along the way

`tribe/fit_virality.py` selected its top-k features **on the full dataset, then** ran
leave-one-out — the held-out reel's label influenced its own selection. That leaked, and
is why it reported 65% on 20 reels. Selection now runs inside each fold. The corrected
number is **86% — below the 88% trivial baseline** (71/81 reels are low-view, so "always
say low" wins 88%). Accuracy was never the right metric here; the script now prints the
baseline beside itself and says so. `validate_virality.py` (AUC + permutation) is the
number to quote.
