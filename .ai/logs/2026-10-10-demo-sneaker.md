# 2026-10-10 — Demo: sneaker.mp4 (signal + synthetic-neural report)
- Goal: Utilise tribe pipeline for a first demo despite no GPU/torch locally.
- Probe: `sneaker.mp4` = 28.33s, 720x1280 9:16, h264+aac (video_info). 1 scene at thresh 0.3 (no cuts). Audio no silence regs, onset ~0.5s, mean −18.6dB (waveform 28 bins).
- Env: local python3.9 lacks torch/tribev2/pandas → full TRIBE inference NOT possible locally; installed `numpy<2 pandas matplotlib nilearn` via pip (DEPS_OK numpy 1.26.4).
- Synthetic neural stand-in: `tribe/out/demo_sneaker/{preds.npy (28,20484), times.npy, meta.json}` via seeded rng (hook bump s1-3, peak s14-17, dip s10-12); transcript None.
- Report: `python3 tribe/hookline_report.py tribe/out/demo_sneaker --video tribe/samples/sneaker.mp4` → PASS. Outputs: report.json/md/png (149K png). Scores: hook 61.4 Average, hold 35.9, finish 22.0, virality 50.7 Average. Drops 0:09-0:13 + 0:26-0:29, static opening flagged (matches 0 cuts). Atlas fetched OK (Destrieux groups present).
- Verdict: Report stage demoed end-to-end on real video signals + synthetic brain. Real neural preds still require Colab T4 run (see colab-bridge.md).
- Next: (1) run `tribe_colab.ipynb` on sneaker.mp4 with `--device cuda`, download `out/`; (2) diff real vs synthetic `series.attention`; (3) scaffold FastAPI serving this report schema.
- Note: `tribe/out/demo_sneaker/` is a demo artifact — do not commit (harness never-commit rule); delete or gitignore before sharing.
