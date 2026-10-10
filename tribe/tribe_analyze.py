#!/usr/bin/env python
"""Run TRIBE v2 on one or more videos and save the predicted cortical response.

usage: python tribe_analyze.py VIDEO [VIDEO ...] [--max-seconds 0] [--out out/] [--device cuda]

Per video, writes to --out/<video-stem>/:
  preds.npy    (T, 20484) predicted cortical response, average subject
  times.npy    (T,) stimulus time (s) of each prediction
  meta.json    video path and timing
Scoring and the report are done by hookline_report.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
os.environ.setdefault("TRIBE_WHISPER_MODEL", "small")  # large-v3 is too slow on CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def trim(video: Path, seconds: float, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-t", str(seconds),
         "-c:v", "libx264", "-c:a", "aac", str(dest)],
        check=True,
    )
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path, nargs="+")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="analyse only the first N seconds (0 = whole video)")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    args = ap.parse_args()

    missing = [v for v in args.videos if not v.exists()]
    if missing:
        print(f"video not found: {missing}", file=sys.stderr)
        return 1

    import numpy as np
    from tribev2 import TribeModel

    dev = args.device
    overrides = {f"data.{m}.device": dev for m in
                 ("text_feature", "audio_feature", "video_feature.image", "image_feature.image")}
    t0 = time.time()
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(args.cache), device=dev, config_update=overrides
    )
    print(f"[timing] load model {time.time() - t0:.1f}s", flush=True)

    for src in args.videos:
        out = args.out / src.stem
        out.mkdir(parents=True, exist_ok=True)
        video = trim(src, args.max_seconds, out / "input_trimmed.mp4") if args.max_seconds > 0 else src

        t1 = time.time()
        events = model.get_events_dataframe(video_path=str(video))
        t2 = time.time()
        print(f"[timing] {src.name}: audio+transcript {t2 - t1:.1f}s", flush=True)
        preds, segments = model.predict(events=events)
        preds = np.asarray(preds)
        t3 = time.time()
        print(f"[timing] {src.name}: encoders+predict {t3 - t2:.1f}s", flush=True)

        times = np.array([float(getattr(s, "start", i)) for i, s in enumerate(segments)])
        times = times - times.min() if len(times) else times
        np.save(out / "preds.npy", preds)
        np.save(out / "times.npy", times)
        # TRIBE writes the transcript next to the extracted wav, i.e. next to the video
        tsv = Path(video).with_suffix(".tsv")
        tsv = tsv if tsv.exists() else None
        (out / "meta.json").write_text(json.dumps({
            "video": str(video), "source": str(src), "transcript": str(tsv) if tsv else None,
            "n_steps": int(preds.shape[0]), "seconds_total": round(t3 - t1, 1),
        }))
        print(f"{src.name}: preds {preds.shape} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
