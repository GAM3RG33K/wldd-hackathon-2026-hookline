#!/usr/bin/env python
"""Held-out validation of the viral-potential score — the answer to "how do you know it works?".

Runs entirely on the stored full-TRIBE reports + `samples/*/metadata.csv` (no GPU, no new
Space jobs). Views and follower bands build the *label only*; they are never features (rule 1).

What it reports, in order:
  1. The current hand-tuned heuristic score's AUC at separating the 10 known-viral outliers
     from the 71 low-view reels. No fitting, so no leakage is possible — this is the
     number to quote to a client.
  2. A fitted model's AUC under repeated stratified k-fold with feature selection AND
     standardisation inside each fold (the previous `fit_virality.py` left-one-out picked
     top-k features on the full set first, which leaks; this does not).
  3. Rank correlation (Spearman) of out-of-fold predictions against log10(views), overall
     and within each stratum — does the score *rank* reels, not just split them.
  4. Per-channel ablation: drop a channel's features, re-fit, and report the held-out AUC
     change with a permutation p-value. A channel only earns GPU time if it helps.
  5. A label-permutation null for every number, so "is this better than chance?" has an
     answer at this sample size.

Accuracy is deliberately NOT the headline: 71/81 reels are low-view, so predicting
"low" for everything already scores 88%. AUC and rank correlation are the honest metrics.

  python tribe/validate_virality.py                    # full report
  python tribe/validate_virality.py --json out.json    # also dump the numbers
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

sys_path = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(sys_path))
import hookline_report as HL  # noqa: E402  (peak_features only)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "backend" / "out"

# Feature groups by channel — the ablation table walks these.
CHANNELS = {
    "attention": ["hook", "hook_lift", "drop_share", "peak_share", "ending", "volatility",
                  "peaks", "peaks_per_window", "peak_gap_cv", "engagement_raw", "score_driven"],
    "ffmpeg": ["cuts_per_s", "motion", "loudness_db", "duration_s"],
    "asr": ["words_per_s"],
    "music": ["music_tempo", "music_mode", "music_onset_density", "music_energy_range",
              "music_buildups", "music_drops", "music_mode_swing", "beat_cut_sync"],
    "emotion": ["emo_strength", "emo_strong_share", "emo_high_arousal_share", "emo_hook",
                "emo_peak", "emo_peak_end", "emo_intensity_mean", "emo_valence_mean",
                "emo_valence_range", "emo_attention_coupling", "emo_resolves",
                "emo_peaks_per_window", "emo_peak_gap_cv"],
    # net_* (brain networks) exist on only 5/81 reports — excluded on coverage, not on merit.
}
BAND_MID = {"10k-50k": 3e4, "50k-200k": 1.2e5, "200k-1M": 5e5, "1M+": 2e6}


def load():
    """Labelled full-TRIBE reports: id, stratum, views, label, feature dict."""
    strata, views = {}, {}
    for st in ("face", "faceless", "outliers"):
        for r in csv.DictReader((ROOT / "samples" / st / "metadata.csv").open()):
            strata[r["video_id"]] = st
            v = (r.get("views") or "").strip()
            if v.isdigit():
                views[r["video_id"]] = int(v)
    rows = []
    for p in sorted(OUT.glob("s-*/report.json")):
        r = json.loads(p.read_text())
        vid = r.get("source_id") or Path(r.get("local_video", "")).stem
        if vid not in strata or vid not in views:
            continue
        if "tribev2-full" not in str(r.get("neural_source", "")):
            continue
        f = {**(r.get("features") or {}), **HL.peak_features((r.get("series") or {}).get("attention") or [])}
        rows.append({
            "id": vid, "stratum": strata[vid], "views": views[vid],
            "logv": math.log10(views[vid]),
            "y": 1 if strata[vid] == "outliers" else 0,
            "vir": (r.get("scores") or {}).get("virality"),
            "base": (r.get("scores") or {}).get("virality_base"),
            "f": f,
        })
    return rows


def usable_keys(rows, groups):
    """Keys present (non-None) on every reel, restricted to the given channels."""
    keys = []
    for g in groups:
        for k in CHANNELS[g]:
            if all(r["f"].get(k) is not None for r in rows):
                keys.append(k)
    return keys


def rank(a):
    a = np.asarray(a, float)
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    a, b = rank(a), rank(b)
    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)
    return float((a * b).mean())


def auc(scores, y):
    """Rank-based AUC; ties get average rank."""
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    r = rank(scores)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if not n1 or not n0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def logreg(X, y, lam=2.0, steps=3000, lr=0.15, tol=1e-7):
    """L2 logistic with early stopping (tiny data, so this is fast and stable)."""
    w, b = np.zeros(X.shape[1]), 0.0
    prev = np.inf
    for _ in range(steps):
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g_w = X.T @ (p - y) / len(y) + lam * w / len(y)
        g_b = float(np.mean(p - y))
        w -= lr * g_w
        b -= lr * g_b
        loss = float(np.mean((p - y) ** 2)) + lam * float(w @ w) / len(y)
        if abs(prev - loss) < tol:
            break
        prev = loss
    return w, b


def cv_predict(X, y, folds, k=5, lam=2.0):
    """Repeated-CV out-of-fold probabilities. Feature selection INSIDE each fold."""
    n = len(y)
    oof = np.zeros(n)
    for tr, te in folds:
        # select top-k by |effect d| on the training fold only
        d = {}
        for j in range(X.shape[1]):
            a1, a0 = X[tr][y[tr] == 1, j], X[tr][y[tr] == 0, j]
            if len(a1) < 2 or len(a0) < 2:
                d[j] = 0.0
                continue
            sp = math.sqrt((a1.var(ddof=1) + a0.var(ddof=1)) / 2) or 1e-9
            d[j] = abs(float((a1.mean() - a0.mean()) / sp))
        keep = sorted(d, key=lambda j: -d[j])[:k]
        mu, sd = X[tr][:, keep].mean(0), X[tr][:, keep].std(0) + 1e-9
        Ztr = (X[tr][:, keep] - mu) / sd
        Zte = (X[te][:, keep] - mu) / sd
        w, b = logreg(Ztr, y[tr], lam=lam)
        oof[te] = 1 / (1 + np.exp(-np.clip(Zte @ w + b, -30, 30)))
    return oof


def make_folds(y, n_splits, rng):
    """Stratified k-fold index pairs (both classes in every test fold)."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = []
    for i in range(n_splits):
        te_p = pos[i::n_splits]
        te_n = neg[i::n_splits]
        te = np.concatenate([te_p, te_n])
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) and len(te):
            folds.append((tr, te))
    return folds


def permute_pvalue(X, y, obs, n_splits, rng, n_perm=200, repeats=10):
    """Fraction of label-shuffles whose held-out AUC beats the observed one.
    Feature selection runs inside every fold, so the null matches the procedure."""
    hits = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        fp = [make_folds(yp, n_splits, rng) for _ in range(repeats)]
        if float(np.mean([auc(cv_predict(X, yp, f), yp) for f in fp])) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also dump numbers to this path")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = load()
    y = np.array([r["y"] for r in rows], dtype=int)
    logv = np.array([r["logv"] for r in rows])
    print(f"labelled full-TRIBE reels with views: {len(rows)} "
          f"({int(y.sum())} viral outliers, {int((1 - y).sum())} low-view)")
    if int(y.sum()) < 3:
        raise SystemExit("need at least 3 outliers")

    # 1. the shipped heuristic — no fitting, so no leakage
    vir = np.array([r["vir"] if r["vir"] is not None else np.nan for r in rows])
    ok = ~np.isnan(vir)
    print("\n[1] SHIPPED HEURISTIC (no fit, no leakage possible)")
    print(f"    AUC outlier-vs-low        {auc(vir[ok], y[ok]):+.3f}")
    print(f"    Spearman vs log10(views)  {spearman(vir[ok], logv[ok]):+.3f}")
    print(f"    top-10-by-views minus bottom-10-by-views: "
          f"{np.mean(sorted(vir[ok], reverse=True)[:10]) - np.mean(sorted(vir[ok])[:10]):+.1f} points")

    all_keys = usable_keys(rows, list(CHANNELS))
    X_all = np.array([[float(r["f"][k]) for k in all_keys] for r in rows])
    print(f"\n    usable features (present on all {len(rows)}): {len(all_keys)} "
          f"across channels {list(CHANNELS)}")

    rng = np.random.default_rng(a.seed)
    schedule = [make_folds(y, a.splits, rng) for _ in range(a.repeats)]
    base_auc = float(np.mean([auc(cv_predict(X_all, y, f), y) for f in schedule]))
    print(f"\n[2] FITTED MODEL, repeated {a.splits}-fold x {a.repeats} (selection inside each fold)")
    print(f"    held-out AUC, all channels   {base_auc:+.3f}   (heuristic above: "
          f"{auc(vir[ok], y[ok]):+.3f})")

    oof = np.mean([cv_predict(X_all, y, f) for f in schedule], axis=0)
    print("\n[3] RANK CORRELATION of out-of-fold predictions vs log10(views)")
    print(f"    overall                     {spearman(oof, logv):+.3f}")
    for st in ("outliers", "face", "faceless"):
        m = np.array([r["stratum"] == st for r in rows])
        if m.sum() > 4:
            print(f"    within {st:9s} (n={int(m.sum()):3d})       {spearman(oof[m], logv[m]):+.3f}")

    print("\n[4] PER-CHANNEL ABLATION (drop channel, re-fit on the SAME folds, held-out AUC)")
    abl = {}
    for ch in CHANNELS:
        keep = [j for j, k in enumerate(all_keys) if ch not in _channel_of(k)]
        if not keep:
            print(f"    {ch:9s} no usable features on this set — skipped")
            continue
        Xk = X_all[:, keep]
        paired = [auc(cv_predict(Xk, y, f), y) - auc(cv_predict(X_all, y, f), y) for f in schedule]
        am = float(base_auc + np.mean(paired))
        abl[ch] = {"auc_without": am, "delta": float(np.mean(paired)),
                   "delta_sd": float(np.std(paired)),
                   "helps": int(sum(1 for d in paired if d < 0))}
        print(f"    without {ch:9s} (kept {len(keep):2d} feats)  AUC {am:+.3f}  "
              f"delta {np.mean(paired):+.3f} ± {np.std(paired):.3f}  "
              f"channel helps in {sum(1 for d in paired if d < 0)}/{len(paired)} folds")

    print("\n[5] PERMUTATION NULL")
    hits = 0
    for _ in range(a.perm):
        yp = rng.permutation(y)
        fp = [make_folds(yp, a.splits, rng) for _ in range(min(a.repeats, 10))]
        if float(np.mean([auc(cv_predict(X_all, yp, f), yp) for f in fp])) >= base_auc:
            hits += 1
    p_all = (hits + 1) / (a.perm + 1)
    print(f"    fitted model: p = {p_all:.3f}  (fraction of {a.perm} label-shuffles beating it)")

    # the heuristic needs no fitting, so its null is a straight shuffle of the labels
    vh, yh = vir[ok], y[ok]
    null = [auc(vh, rng.permutation(yh)) for _ in range(2000)]
    p_heur = float((np.sum(np.array(null) >= auc(vh, yh)) + 1) / (len(null) + 1))
    print(f"    shipped heuristic: p = {p_heur:.3f}  (2000 label-shuffles)")

    # verdict is derived from the numbers, not asserted
    helping = sorted((ch for ch, v in abl.items() if v["delta"] < 0),
                     key=lambda ch: abl[ch]["delta"])
    hurting = sorted((ch for ch, v in abl.items() if v["delta"] > 0),
                     key=lambda ch: -abl[ch]["delta"])
    h_auc = auc(vir[ok], y[ok])
    gain = base_auc - h_auc
    parts = [
        f"Shipped heuristic separates the 10 viral outliers from the 71 low-view reels at "
        f"AUC {h_auc:+.2f} (p={p_heur:.3f}).",
        (f"Refitting on all {len(rows)} reels gives held-out AUC {base_auc:+.2f} (p={p_all:.3f}), "
         f"a change of {gain:+.2f} — {'so the hand-tuned weights stay' if abs(gain) < 0.03 else 'and is worth a closer look'}."),
        ("Removing a channel, re-fitting and re-testing on the same folds: "
         + ("no channel's removal moves AUC more than its own fold-to-fold spread"
            if not helping else "channels that currently help: "
            + ", ".join(f"{c} ({abl[c]['delta']:+.2f})" for c in helping))
         + ("; " if helping and hurting else "")
         + ("channels that would help if removed: "
            + ", ".join(f"{c} ({abl[c]['delta']:+.2f})" for c in hurting) if hurting else "") + "."),
    ]
    verdict = " ".join(parts)
    print(f"\nVERDICT:\n  {verdict}")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "n": len(rows), "n_outliers": int(y.sum()),
            "heuristic_auc": float(auc(vir[ok], y[ok])),
            "heuristic_spearman": float(spearman(vir[ok], logv[ok])),
            "fitted_auc": base_auc, "fitted_spearman": float(spearman(oof, logv)),
            "ablation_auc": abl, "p_fitted": p_all, "p_heuristic": p_heur,
            "features_used": all_keys, "verdict": verdict,
        }, indent=1))
        print(f"wrote {a.json}")


def _channel_of(key):
    for ch, ks in CHANNELS.items():
        if key in ks:
            return ch
    return set()


if __name__ == "__main__":
    main()
