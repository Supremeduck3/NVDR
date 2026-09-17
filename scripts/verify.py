#!/usr/bin/env python3
"""
verify.py — regression gate for the NVDR pipeline.

For every sample image it checks the three things that have to hold before
any optimisation work can be trusted:

  1. Determinism — the encoder is run twice and the two containers must be
     byte-identical. The parallel build used to make every run differ,
     which silently invalidated any A/B measurement.
  2. Structure — svbc_check decodes the container through the public
     reader and verifies codebook bounds, full canvas coverage and PSNR.
  3. Size — the gzipped container against the source file, so a change
     that trades fidelity for bytes is visible instead of implied.

Usage:
    python3 scripts/verify.py [--images DIR] [--min-psnr X] [--jobs N]

Exit code is non-zero if any check fails, so it works as a CI gate.
"""
import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".ppm"}


def find_binary(name):
    for candidate in (ROOT / name, ROOT / f"{name}.exe"):
        if candidate.exists():
            return candidate
    sys.exit(f"{name} not built — run `make` (and `make svbc_check`) first")


def encode(binary, image, out_base):
    result = subprocess.run(
        [str(binary), str(image), str(out_base), "--format", "svbc"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None, result.stderr.strip() or "encoder failed"
    svbc = out_base.with_suffix(".svbc")
    if not svbc.exists():
        return None, "encoder produced no .svbc"
    return svbc, None


def gzip_size(path):
    import gzip
    return len(gzip.compress(path.read_bytes(), 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "samples"))
    ap.add_argument("--min-psnr", type=float, default=0.0,
                    help="fail a sample whose reconstruction falls below this PSNR")
    args = ap.parse_args()

    encoder = find_binary("image_to_svg")
    checker = find_binary("svbc_check")

    images = sorted(
        p for p in pathlib.Path(args.images).iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        sys.exit(f"no images found in {args.images}")

    failures = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nvdr-verify-"))
    try:
        print(f"{'image':<28} {'gzip/source':>12} {'determinism':>12}  structure")
        print("-" * 78)
        for image in images:
            name = image.name

            first, err = encode(encoder, image, tmp / f"{image.stem}_a.svg")
            if err:
                failures.append(f"{name}: {err}")
                print(f"{name:<28} {'-':>12} {'-':>12}  ENCODE FAILED: {err}")
                continue
            second, err = encode(encoder, image, tmp / f"{image.stem}_b.svg")
            if err:
                failures.append(f"{name}: {err}")
                continue

            digest_a = hashlib.sha256(first.read_bytes()).hexdigest()
            digest_b = hashlib.sha256(second.read_bytes()).hexdigest()
            stable = digest_a == digest_b
            if not stable:
                failures.append(f"{name}: encoder is not deterministic")

            ratio = gzip_size(first) / image.stat().st_size

            check = subprocess.run(
                [str(checker), str(image), str(first),
                 "--min-psnr", str(args.min_psnr), "--quiet"],
                capture_output=True, text=True,
            )
            if check.returncode != 0:
                failures.append(f"{name}: {check.stdout.strip() or 'structural check failed'}")

            print(f"{name:<28} {ratio:>11.0%} {'stable' if stable else 'VARIES':>12}  "
                  f"{'ok' if check.returncode == 0 else check.stdout.strip()}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all {len(images)} samples passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
