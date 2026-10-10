# 03 — UI/UX + Report Assets

## Status
No Figma/Jira/Miro links provided. No Lottie files. No Excel feature map. This file governs the only UI surface: the generated report (`report.md` + `report.png` + `report.json` series). If design links arrive later, verify read tool first; if unreadable ask for exported images/PlantUML — never hallucinate components.

## Report contract (from `hookline_report.py`, keep layout)
- `report.md`: title `# Hookline report`, line `` `<name>`, <dur>s ``, bullets Hook/Hold/Finish/Virality with bands, one-line moments (`Peak hook at M:SS…`), `## What to change` bullets `- **<at>**, <issue>. <fix>`, footer `_heuristic, within-video normalised, uncalibrated_`.
- `report.png` (12×9, gridspec 4 rows): row0 4 score cards (Hook/Hold/Finish/Virality + band sublabel); row1 attention blue solid + survival orange dashed, hook window green shade, drops red shade, cuts grey vlines, peak dots; row2 network heatmap `RdBu_r ±2.5` labeled per group; row3 top-6 recs text. `Agg` backend, `dpi=130`, `bbox_inches=tight`.
- Time format `M:SS` via `fmt()`; windows `start-end`; attention 0–100 1-decimal; survival %.
- Backend serving: `GET /report/{id}` returns `report.json` verbatim; render md as HTML with same order; png as-is. No restyling that drops timestamps/bands/method disclaimer.

## Component mapping (when frontend built)
- Score cards → 4 stat tiles (value + band caption). Attention chart → line chart with shaded hook/drop zones + cut markers (data: `series.attention`, `series.survival_pct`, `series.cuts_s`). Network strip → heatmap rows (`network_mean_response` keys). Rec list → timeline rows keyed by `at`, each expandable to `fix`. Upload → `POST /analyze`; polling → `GET /report/{id}` status chip (`queued_colab_manual|done|failed`).
- Copy rules: bands verbatim (`Bad/Weak/Average/Strong/Viral-grade`); never promise retention; always show method disclaimer near scores; recs keep timestamp prefix.

## Don’ts
No follower/handle display as signal. No invented Figma components. No Lottie triggers (none exist). No new report fields without updating `analyse()` + md + png + backend schema together.
