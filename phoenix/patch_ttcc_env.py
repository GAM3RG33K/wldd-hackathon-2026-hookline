#!/usr/bin/env python3
"""Idempotent patch for the TTCC environment's transformers version checker.

transformers' `_compare_versions` raises `ValueError` instead of skipping when
a dependency's installed version cannot be detected (regex, tqdm, …). The
checkpoint stack doesn't actually need those exact versions, so when the
version is undetectable we skip the comparison instead of failing.

Run once after creating the TTCC venv:
    python phoenix/patch_ttcc_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttcc import TTCC_PY  # noqa: E402

OLD = """def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None or want_ver is None:
        raise ValueError("""
NEW = """def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None:  # PHOENIX_PATCH: version undetectable -> skip, don't fail
        return
    if got_ver is None or want_ver is None:
        raise ValueError("""

MARKER = "PHOENIX_PATCH"


OLD_AV = 'metadata_errors="ignore"'


def _patch_av_call_sites() -> int:
    """Newer PyAV (19.x) removed `metadata_errors`; older torchvision/faster_whisper
    still pass it. Remove the kwarg at the (few) call sites."""
    site = TTCC_PY.parent.parent
    patched = 0
    for rel in ("torchvision/io/video.py", "torchvision/io/video_reader.py",
                "faster_whisper/audio.py"):
        f = next(site.glob(f"lib/python3.*/site-packages/{rel}"), None)
        if f is None or not f.is_file():
            continue
        src = f.read_text()
        if OLD_AV in src:
            f.write_text(src.replace(f", {OLD_AV}", "").replace(f"({OLD_AV}", "(")
                         .replace(f"{OLD_AV}, ", ""))
            patched += 1
            print(f"av call-site patched: {f.name}")
    return patched


def main() -> int:
    site = TTCC_PY.parent.parent
    matches = sorted(site.glob("lib/python3.*/site-packages/transformers/utils/versions.py"))
    if not matches:
        print("transformers versions.py not found under", site)
        return 1
    for vfile in matches:
        src = vfile.read_text()
        if MARKER in src:
            print(f"already patched: {vfile}")
        elif OLD not in src:
            print(f"pattern not found in {vfile} — transformers version changed?")
            return 1
        else:
            vfile.write_text(src.replace(OLD, NEW, 1))
            print(f"patched: {vfile}")
    _patch_av_call_sites()
    return 0


if __name__ == "__main__":
    sys.exit(main())
