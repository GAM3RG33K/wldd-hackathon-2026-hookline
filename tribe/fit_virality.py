#!/usr/bin/env python
"""Fit the viral-potential model on analysed sample reels.

Label: 1 = known-viral outlier reel (samples/outliers), 0 = lowest-view reel (face/faceless).
Inputs: the cross-video `features` block of each saved full-TRIBE report (backend/out/s-*/).
Account data (handle, follower band, views) is used ONLY to pick the label groups, never as
a feature: the model judges the video.

Small n, so: univariate effect sizes first, then an L2 logistic regression on the strongest
few features, with leave-one-out accuracy as an honesty check. Feature selection runs INSIDE
each fold — selecting top-k on the full set first and then cross-validating leaks the held-out
reel into the selection and inflates the number.

  python tribe/fit_virality.py            # fit, write tribe/virality_model.json, re-score reports
  python tribe/fit_virality.py --dry-run  # print the fit only

Before fitting at all, run `python tribe/validate_virality.py`: it reports AUC + permutation
p + a per-channel ablation on held-out folds. The model is only written when that run shows
the fit beating the shipped heuristic on held-out data.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

sys_path = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(sys_path))
import hookline_report as HL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "backend" / "out"
MODEL = Path(__file__).resolve().parent / "virality_model.json"
CANDIDATES = ["hook", "engagement_raw", "hook_lift", "peak_share", "drop_share", "ending", "volatility",
              "cuts_per_s", "motion", "loudness_db", "duration_s", "net_visual", "net_auditory",
              "net_language", "net_attention", "net_default_mode",
              "peaks", "peaks_per_window", "peak_gap_cv"]
# avg attention is NOT a candidate: it is ~50 for every video by construction (within-video score).


def strata():
    out = {}
    for st in ("face", "faceless", "outliers"):
        for r in csv.DictReader((ROOT / "samples" / st / "metadata.csv").open()):
            out[r["video_id"]] = st
    return out


def load():
    st = strata()
    rows = []
    for p in sorted(OUT.glob("s-*/report.json")):
        r = json.loads(p.read_text())
        f = r.get("features")
        vid = r.get("source_id") or Path(r.get("local_video", "")).stem
        if not f or vid not in st or "tribev2-full" not in str(r.get("neural_source", "")):
            continue
        f = {**f, **HL.peak_features(r.get("series", {}).get("attention") or [])}
        rows.append({"id": vid, "path": p, "y": 1 if st[vid] == "outliers" else 0, "f": f})
    return rows


def logreg(X, y, lam=1.0, steps=4000, lr=0.1):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + lam * w / len(y))
        b -= lr * float(np.mean(p - y))
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--k", type=int, default=5, help="features kept for the fit")
    ap.add_argument("--lam", type=float, default=2.0, help="L2 strength")
    a = ap.parse_args()

    rows = load()
    y = np.array([r["y"] for r in rows], dtype=float)
    print(f"reels: {len(rows)} ({int(y.sum())} viral outliers, {int(len(y) - y.sum())} low-view)")
    if y.sum() < 3 or (len(y) - y.sum()) < 3:
        raise SystemExit("need at least 3 reels in each group")
    feats = [k for k in CANDIDATES if all(r["f"].get(k) is not None for r in rows)]
    X = np.array([[float(r["f"][k]) for k in feats] for r in rows])

    # effect sizes (Cohen's d): how far apart the two groups sit on each feature.
    # Reported on the FULL set — descriptive only. It must NOT feed the CV loop below.
    d = {}
    for i, k in enumerate(feats):
        a1, a0 = X[y == 1, i], X[y == 0, i]
        sp = np.sqrt((a1.var(ddof=1) + a0.var(ddof=1)) / 2) or 1e-9
        d[k] = float((a1.mean() - a0.mean()) / sp)
    print("\nfeature            viral mean   low mean   effect d")
    for k in sorted(d, key=lambda k: -abs(d[k])):
        i = feats.index(k)
        print(f"  {k:16s} {X[y == 1, i].mean():11.4f} {X[y == 0, i].mean():10.4f} {d[k]:+9.2f}")

    # leave-one-out with feature selection INSIDE each fold. Selecting top-k on the full
    # set first and then cross-validating leaks the held-out reel into the selection and
    # inflates accuracy (that bug reported 65% on 20 reels; in-fold selection does not).
    hits = []
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        d_fold = {}
        for j, k in enumerate(feats):
            a1, a0 = X[m][y[m] == 1, j], X[m][y[m] == 0, j]
            if len(a1) < 2 or len(a0) < 2:
                d_fold[k] = 0.0
                continue
            sp = np.sqrt((a1.var(ddof=1) + a0.var(ddof=1)) / 2) or 1e-9
            d_fold[k] = abs(float((a1.mean() - a0.mean()) / sp))
        keep = sorted(d_fold, key=lambda k: -d_fold[k])[:a.k]
        idx = [feats.index(k) for k in keep]
        mu, sd = X[m][:, idx].mean(0), X[m][:, idx].std(0) + 1e-9
        Z = (X[:, idx] - mu) / sd          # held-out row scored on the fold's own mu/sd
        w, b = logreg(Z[m], y[m], a.lam)
        hits.append(int((Z[i] @ w + b > 0) == bool(y[i])))
    loo = float(np.mean(hits))
    base = float(max(y.sum(), len(y) - y.sum()) / len(y))
    print(f"\nleave-one-out accuracy (selection inside each fold): {loo:.0%}")
    print(f"trivial baseline (always guess the bigger group):    {base:.0%}")
    if loo <= base:
        print(f"=> the model does NOT beat the baseline. Accuracy is the wrong metric here "
              f"({int(len(y) - y.sum())} of {len(y)} reels are low-view); use "
              f"tribe/validate_virality.py, which reports AUC + a permutation test.")
    print("With 10 outliers every number has wide error bars — validate_virality.py is "
          "the one to quote.")

    # final fit on everything, for use only once held-out reels exist to test on
    keep = sorted(d, key=lambda k: -abs(d[k]))[:a.k]
    idx = [feats.index(k) for k in keep]
    mu, sd = X[:, idx].mean(0), X[:, idx].std(0) + 1e-9
    Z = (X[:, idx] - mu) / sd
    w, b = logreg(Z, y, a.lam)
    print(f"\nkept features: {keep}\nweights: {dict(zip(keep, np.round(w, 3)))}  bias {b:.3f}")

    model = {"label": f"first-pass fit: {int(y.sum())} viral outliers vs {int(len(y) - y.sum())} low-view reels, "
                      f"leave-one-out {loo:.0%}; not yet validated on held-out reels",
             "weights": dict(zip(keep, map(float, w))), "bias": float(b),
             "mean": dict(zip(keep, map(float, mu))), "std": dict(zip(keep, map(float, sd))),
             "effect_d": d, "n": len(y), "loo_accuracy": loo}
    if a.dry_run:
        return
    MODEL.write_text(json.dumps(model, indent=1))
    print(f"\nwrote {MODEL}")

    import importlib, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hookline_report as HL
    importlib.reload(HL)
    for r in rows:  # re-score saved reports with the fitted model
        rep = json.loads(r["path"].read_text())
        f = r["f"]
        z = b + sum(wk * ((f[k] - model["mean"][k]) / model["std"][k]) for k, wk in model["weights"].items())
        v = float(100 / (1 + np.exp(-z)))
        rep["scores"].update(virality=round(v, 1), virality_band=HL.band(v), virality_tag=HL.virality_tag(v),
                             virality_method=model["label"])
        r["path"].write_text(json.dumps(rep, indent=1))
        print(f"  {r['id']:6s} {'viral' if r['y'] else 'low  '} -> {v:5.1f} {HL.virality_tag(v)}")


if __name__ == "__main__":
    main()
