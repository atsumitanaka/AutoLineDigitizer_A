#!/usr/bin/env python3
"""Deduplicate a PyInstaller macOS .app bundle.

PyInstaller mirrors every bundled module/binary to BOTH `Contents/Resources` and
`Contents/Frameworks` so the app can be code-signed. For an unsigned internal
build that doubles the on-disk size for no benefit. This script replaces each
top-level entry under `Resources/` with a relative symlink to its twin under
`Frameworks/` when the sizes match.

A future signed/notarized build MUST skip this step (code signatures do not
survive symlinks); keep the DEDUPE guard in CI for that reason.

Usage:
  python scripts/dedupe_macos_bundle.py dist/AutoLineDigitizer.app
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _fmt_mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def dedupe(app_path: Path) -> int:
    resources = app_path / "Contents" / "Resources"
    frameworks = app_path / "Contents" / "Frameworks"
    if not resources.is_dir() or not frameworks.is_dir():
        raise SystemExit(
            f"Not a PyInstaller macOS bundle: missing {resources} or {frameworks}"
        )

    freed = 0
    skipped = 0
    linked = 0
    for res_entry in sorted(resources.iterdir()):
        name = res_entry.name
        if res_entry.is_symlink():
            continue
        fw_entry = frameworks / name
        if not fw_entry.exists():
            continue
        res_size = _size(res_entry)
        fw_size = _size(fw_entry)
        if res_size == 0 or fw_size == 0:
            continue
        if res_size != fw_size:
            print(f"  skip {name}: size differs ({_fmt_mb(res_size)} vs {_fmt_mb(fw_size)})")
            skipped += 1
            continue
        try:
            if res_entry.is_dir():
                shutil.rmtree(res_entry)
            else:
                res_entry.unlink()
            os.symlink(f"../Frameworks/{name}", res_entry)
        except OSError as e:
            print(f"  skip {name}: {e}")
            skipped += 1
            continue
        freed += res_size
        linked += 1
        print(f"  linked {name:40s}  {_fmt_mb(res_size)}")

    print(
        f"\nDone. Linked {linked} entries, freed {_fmt_mb(freed)}; "
        f"skipped {skipped}."
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(dedupe(Path(sys.argv[1]).resolve()))
