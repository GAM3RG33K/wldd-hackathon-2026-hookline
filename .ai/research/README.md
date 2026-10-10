# Research — reusable findings (cite source file/line, no hallucinating)

- `tribe-v2-model.md` — what TRIBE v2 is, encoders, prediction shape, atlas fallback.
- `scoring-heuristic.md` — attention / hook / hold / virality formulas, calibration status, fit plan vs view counts.
- `emotion.md` — emotion channels, what each can and cannot see, failure cases found on real reels and their fixes.
- `hf-space.md` — GPU Space decisions: hardware, speed, model pins, deploy loop.
- `colab-bridge.md` — RETIRED 2026-10-10 (kept for reference).

## Rules
- Every claim cites its source (`tribe/hookline_report.py:24-33`, a log entry, a model card).
- Hypotheses tagged `HYPOTHESIS`; fitted facts tagged `FITTED + date + data`.
- Promote a finding to `.ai/rules/01-business-logic.md` / `02-tech-stack.md` only after a logged verification run.
