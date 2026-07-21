#!/usr/bin/env python3
"""
bench/preview.py — Generate stable SVG/SVBC previews for each image that has
a row in bench/results.jsonl, so the TUI can render thumbnails.

Why this exists: the bench harness tears down its /tmp/nvdr-bench-* outputs
after every run, so the only persisted artefact is the JSONL. To render
SVGs in the UI we need on-disk SVGs keyed by image name.

For every image present in results.jsonl, we resolve the original input on
disk (via NVDR_BENCH_IMAGES / /home/.../samples_images default), run
image_to_svg once, and write the outputs into bench/preview/ with
filenames = <image_stem>.svg and <image_stem>.svbc.

Idempotent: skipped images whose outputs already exist unless --force.

Usage:
  python3 scripts/bench/preview.py            # build missing previews
  python3 scripts/bench/preview.py --force    # rebuild all
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("NVDR_REPO", "/home/supremeduck008/NVDR"))
BINARY = REPO / "image_to_svg"
RESULTS = REPO / "bench" / "results.jsonl"
PREVIEW_DIR = REPO / "bench" / "preview"

DEFAULT_IMG_DIRS = [
    Path(os.environ.get("NVDR_BENCH_IMAGES", "/home/supremeduck008/share/code/samples_images")),
    Path("/home/supremeduck008/NVDR/samples"),
]


def resolve_image(name: str) -> Path | None:
    """Find the input image by filename across plausible dirs."""
    candidates = []
    env = os.environ.get("NVDR_BENCH_IMAGES")
    if env:
        candidates.append(Path(env))
    candidates.extend(DEFAULT_IMG_DIRS)
    for d in candidates:
        p = d / name
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not BINARY.exists():
        print(f"error: binary missing at {BINARY}; run `make` first", file=sys.stderr)
        return 2
    if not RESULTS.exists():
        print(f"error: {RESULTS} not found; run bench first", file=sys.stderr)
        return 2

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    with RESULTS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            images.append(r["image"])
    images = sorted(set(images))
    print(f"# {len(images)} unique images in {RESULTS}", file=sys.stderr)

    # image_to_svg writes <out_base> literally (no .svg suffix) and the .svbc
    # alongside. We expose them via manifest with .svg/.svbc suffixes for the
    # UI's convenience. Idempotent skip uses the literal outputs.
    for name in images:
        src = resolve_image(name)
        if not src:
            print(f"  skip {name!r}: input not found", file=sys.stderr)
            continue
        stem = src.stem
        out_base = PREVIEW_DIR / stem
        raw_svg = Path(str(out_base))            # no extension, as written
        raw_svbc = Path(str(out_base) + ".svbc")
        if not args.force and raw_svg.exists() and raw_svbc.exists():
            print(f"  ok   {name} (cached)", file=sys.stderr)
            continue

        cmd = [str(BINARY), str(src), str(out_base)]
        print(f"  run  {name}", file=sys.stderr)
        try:
            subprocess.run(cmd, capture_output=True, timeout=180, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  FAIL {name}: rc={e.returncode}", file=sys.stderr)
            continue
        except subprocess.TimeoutExpired:
            print(f"  FAIL {name}: timeout", file=sys.stderr)
            continue

    # Print a tiny manifest the TUI can rely on. image_to_svg writes the
    # SVG literally with no extension, so we resolve the literal file via
    # the matching .svbc sibling (since both share out_base).
    manifest = PREVIEW_DIR / "manifest.json"
    items = []
    for svbc in sorted(PREVIEW_DIR.glob("*.svbc")):
        stem = svbc.name[: -len(".svbc")]
        out_base = PREVIEW_DIR / stem
        svg = out_base                       # literal path, no suffix
        # Find the original input image by JSONL lookup
        items.append({
            "stem": stem,
            "input": str(svbc).rsplit("/", 1)[0] + "/" + stem,  # placeholder
            "svg": str(svg) if svg.exists() else None,
            "svg_kb": round(svg.stat().st_size / 1024.0, 1) if svg.exists() else None,
            "svbc_kb": round(svbc.stat().st_size / 1024.0, 1),
        })
    # Re-resolve input paths properly via the JSONL field name -> image_path
    image_map = {}
    for name in images:
        src = resolve_image(name)
        if src:
            image_map[src.stem] = str(src)
    for it in items:
        it["input"] = image_map.get(it["stem"])
        # strip placeholder
        if it["input"] is None:
            it.pop("input", None)
    manifest.write_text(json.dumps(items, indent=2))
    print(f"# manifest written: {manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
