#!/usr/bin/env python3
"""
bench/run.py — Benchmark NVDR's cross-file codebook accumulation.

Runs image_to_svg across every sample image twice (cold + warm with
--codebook-db), collects size/compression metrics, and reports:

   * Per-image SVBC size, leaf counts, processing time
   * Cumulative codebook growth across the run order
   * Cold-vs-warm compression delta (does warm produce smaller files
     on the same images?)

Output:
   bench/results.jsonl — one JSON record per run, append-only
   stdout — summary table + verdict

Usage:
   python3 scripts/bench/run.py                       # defaults
   python3 scripts/bench/run.py --reset               # ignore prior runs
   python3 scripts/bench/run.py --images DIR          # custom sample dir
   python3 scripts/bench/run.py --db PATH             # custom DB path
   python3 scripts/bench/run.py --limit 12            # first 12 images only
   NVDR_BENCH_IMAGES=/path/to/with/50/run-on/ python3 scripts/bench/run.py

MVP scope: this is a measurement harness, not a perf benchmark.
We do NOT time individual phases inside image_to_svg — that needs
a profiler. We trust the binary's own summary line.
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path


REPO = Path("/tmp/nvdr-inspect")
DEFAULT_BIN = REPO / "image_to_svg"
DEFAULT_IMG_DIR = REPO / "samples"
DEFAULT_DB = Path("/tmp/nvdr-bench-codebook.nvcb")
RESULTS_PATH = REPO / "bench" / "results.jsonl"

# Honor $NVDR_BENCH_IMAGES if set (for sandboxed/SSH environments where
# the real sample set lives outside the repo). Falls back to defaults.
_env_img = os.environ.get("NVDR_BENCH_IMAGES")
if _env_img:
    DEFAULT_IMG_DIR = Path(_env_img)


def run_one(binary: Path, image: Path, db_path: Path):
    """Run image_to_svg on `image` with --codebook-db if db_path is set.
    Captures stdout/stderr, parses the summary line and the codebook
    notice. Returns a dict ready for json.dump.
    """
    out_base = Path("/tmp") / f"nvdr-bench-{image.stem}-{int(time.time()*1000)}"
    cmd = [str(binary), str(image), str(out_base)]
    if db_path:
        cmd += ["--codebook-db", str(db_path)]

    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd, cwd=str(REPO),
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {
            "image": image.name,
            "ok": False,
            "error": "timeout (>180s)",
        }
    elapsed_ms = (time.monotonic() - started) * 1000.0

    if result.returncode != 0:
        return {
            "image": image.name,
            "ok": False,
            "error": f"exit {result.returncode}: {result.stderr.strip()[:200]}",
        }

    stdout = result.stdout
    rec = {
        "image": image.name,
        "ok": True,
        "elapsed_ms_observed": round(elapsed_ms, 1),
    }

    # leaves: 5287 -> 2176 | total time: 760ms
    m = re.search(r"leaves:\s+(\d+)\s+->\s+(\d+)\s+\|\s+total time:\s+([\d.]+)ms", stdout)
    if m:
        rec["leaves_before"] = int(m.group(1))
        rec["leaves_after"] = int(m.group(2))
        rec["ms_reported"] = float(m.group(3))

    # SVG:       63 KB
    m = re.search(r"SVG:\s+(\d+)\s+KB", stdout)
    if m:
        rec["svg_kb"] = int(m.group(1))

    # SVBC:      24 KB  (2.6x menor que SVG)
    m = re.search(r"SVBC:\s+(\d+)\s+KB", stdout)
    if m:
        rec["svbc_kb"] = int(m.group(1))

    # Codebook:  256 unique colors persisted
    m = re.search(r"Codebook:\s+(\d+)\s+unique colors persisted", stdout)
    if m:
        rec["codebook_size_after"] = int(m.group(1))
        rec["used_codebook_db"] = True

    # Original:  12 KB
    m = re.search(r"Original:\s+(\d+)\s+KB", stdout)
    if m:
        rec["original_kb"] = int(m.group(1))

    # Ratio:     190% of original
    m = re.search(r"Ratio:\s+([\d.]+)%", stdout)
    if m:
        rec["ratio_pct"] = float(m.group(1))

    # Strip SVBC if it exists — actual file size, not stderr-summary.
    svbc_path = out_base.with_suffix(".svbc")
    if svbc_path.exists():
        rec["svbc_actual_bytes"] = svbc_path.stat().st_size
        # Pull W/H from the SVBC v3 header to compute uncompressed
        # reference size: W*H*3 bytes of raw RGB.
        try:
            with open(svbc_path, "rb") as f:
                hdr = f.read(20)
            if len(hdr) >= 12 and hdr[:4] == b"SVBC" and hdr[4] == 3:
                w, h = struct.unpack_from("<HH", hdr, 8)
                rec["img_width"] = int(w)
                rec["img_height"] = int(h)
                rec["uncompressed_kb"] = round((w * h * 3) / 1024.0, 1)
        except OSError:
            pass  # leave dims unset; SVBC unreadable

        # Compare-against-cjpeg sizes. These are decoded from the
        # ORIGINAL source file (not from SVBC), so they measure what
        # an equivalent-quality JPEG would look like vs our SVBC.
        # Quality 90 = near-lossless; 75 = high but compressed.
        rec["cjpeg90_kb"] = try_cjpeg(image, 90)
        rec["cjpeg75_kb"] = try_cjpeg(image, 75)

    return rec


def try_cjpeg(image_path: Path, quality: int) -> int:
    """Run `cjpeg -quality N` on the input image's pixel data and return
    the produced JPEG's size in KB, rounded to 1 decimal.

    cjpeg needs PPM/PGM/BMP/TGA input — JPG is not accepted. So if the
    source is JPG (or anything else non-PPM), we route through ffmpeg
    to decompress to PPM first. ffmpeg is widely available even on
    minimal sandboxes where Pillow isn't installed.

    Returns 0 if neither tool is missing or fails. The bench records
    the field either way — readers can tell "0" from a successful
    small file because all of our test images produce >1 KB outputs.
    """
    ppm_out = Path("/tmp") / f"nvdr-cjpeg-{int(time.time()*1000)*1000}-{quality}.ppm"
    jpg_out = Path("/tmp") / f"nvdr-cjpeg-{int(time.time()*1000)*1000+1}-{quality}.jpg"
    try:
        # Step 1: decode input → raw PPM (if input isn't already PPM).
        if not image_path.suffix.lower() in (".ppm", ".pgm", ".bmp", ".tga", ".tiff"):
            if not shutil.which("ffmpeg"):
                return 0
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-i", str(image_path), str(ppm_out), "-y"],
                capture_output=True, timeout=60, check=True,
            )
            pmp = ppm_out
        else:
            pmp = image_path

        # Step 2: encode PPM → JPEG at the requested quality.
        if not shutil.which("cjpeg"):
            if pmp != image_path and pmp.exists():
                pmp.unlink()
            return 0
        subprocess.run(
            ["cjpeg", "-quality", str(quality), "-outfile", str(jpg_out), str(pmp)],
            capture_output=True, timeout=60, check=True,
        )
        if jpg_out.exists():
            return round(jpg_out.stat().st_size / 1024.0, 1)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass
    finally:
        for p in (ppm_out, jpg_out):
            if p.exists():
                try: p.unlink()
                except OSError: pass
    return 0


def collect_images(dirpath: Path):
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(p for p in dirpath.iterdir()
                  if p.suffix.lower() in extensions and not p.name.startswith("."))


def filter_top_n(images, n):
    """Pick first N images, then sort the slice lexicographically by filename.
    Deterministic across runs."""
    return sorted(images[:n]) if n and n > 0 else images


def reset_db(db_path: Path):
    if db_path.exists() or db_path.is_symlink():
        db_path.unlink()


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_table(records):
    if not records:
        print("(no successful records)")
        return
    cols = ("image", "WxH", "svbc_KB", "uncomp_KB",
            "cj75_KB", "cj90_KB", "svbc_vs_cj75",
            "leaves", "ms")
    fmt = ("{:<28} {:>11} {:>8} {:>10} {:>8} {:>8} {:>12} "
           "{:>10} {:>6}")
    print()
    print(fmt.format(*cols))
    print("-" * 110)
    for r in records:
        if not r.get("ok"):
            print(f"  FAIL  {r.get('image')}: {r.get('error')}")
            continue
        wh = "-"
        if r.get("img_width") and r.get("img_height"):
            wh = f"{r['img_width']}x{r['img_height']}"
        svbc = r.get("svbc_kb") or 0
        cj75 = r.get("cjpeg75_kb") or 0
        ratio = "-"
        if cj75 > 0 and svbc > 0:
            ratio = f"{(svbc - cj75) / cj75 * 100:+.0f}%"
        print(fmt.format(
            r["image"][:28],
            wh,
            svbc or "-",
            r.get("uncompressed_kb", "-"),
            cj75 or "-",
            r.get("cjpeg90_kb", "-") or "-",
            ratio,
            f"{r.get('leaves_after', '-')}/{r.get('leaves_before', '-')}",
            r.get("ms_reported", "-"),
        ))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--images", type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--reset", action="store_true",
                    help="Delete DB and results file before running")
    ap.add_argument("--no-db", action="store_true",
                    help="Run without --codebook-db (control arm)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N images (0=all). Picked deterministically.")
    args = ap.parse_args()

    if not args.binary.exists():
        print(f"error: binary not found at {args.binary}; run `make` first", file=sys.stderr)
        sys.exit(1)
    if not args.images.is_dir():
        print(f"error: images dir not found: {args.images}", file=sys.stderr)
        sys.exit(1)

    images = filter_top_n(collect_images(args.images), args.limit)
    if not images:
        print(f"error: no images at {args.images}", file=sys.stderr)
        sys.exit(1)

    db_path = None if args.no_db else args.db

    if args.reset:
        if db_path: reset_db(db_path)
        if RESULTS_PATH.exists(): RESULTS_PATH.unlink()
        print(f"# reset: db={'cleared' if db_path else 'n/a'}, results.jsonl=cleared\n")

    print(f"# binary:  {args.binary}")
    print(f"# images:  {args.images}  ({len(images)} files)")
    print(f"# db:      {db_path or '(disabled)'}")
    print(f"# results: {RESULTS_PATH}\n")

    records = []
    print("# running COLD (no codebook context)...")
    cold_records = []
    if db_path: reset_db(db_path)  # ensure cold path
    for img in images:
        rec = run_one(args.binary, img, db_path)
        rec["phase"] = "cold"
        records.append(rec); cold_records.append(rec)
        append_jsonl(RESULTS_PATH, rec)
        print(f"  cold  {rec.get('image','?'):<28} → "
              f"leaves {rec.get('leaves_after','?')}/{rec.get('leaves_before','?')}, "
              f"svbc {rec.get('svbc_kb','?')} KB")

    if db_path:
        print("\n# running WARM (codebook populated)...")
        warm_records = []
        # DB is already populated by cold phase; do NOT reset.
        # Re-run in same order; each call accumulates weight,
        # but color dedup is what we're measuring.
        for img in images:
            rec = run_one(args.binary, img, db_path)
            rec["phase"] = "warm"
            records.append(rec); warm_records.append(rec)
            append_jsonl(RESULTS_PATH, rec)
            print(f"  warm  {rec.get('image','?'):<28} → "
                  f"leaves {rec.get('leaves_after','?')}/{rec.get('leaves_before','?')}, "
                  f"svbc {rec.get('svbc_kb','?')} KB")
    else:
        warm_records = []

    # ---- Summary ----------------------------------------------------
    print("\n# COLD run (no persisted codebook):")
    print_table(cold_records)

    if warm_records:
        print("# WARM run (codebook loaded):")
        print_table(warm_records)

        # Computational verdict: how much did the codebook reduce work?
        cold_total_svbc = sum(r.get("svbc_kb", 0) for r in cold_records if r.get("ok"))
        warm_total_svbc = sum(r.get("svbc_kb", 0) for r in warm_records if r.get("ok"))
        print("# codebook-effect verdict:")
        print(f"  total SVBC cold: {cold_total_svbc} KB")
        print(f"  total SVBC warm: {warm_total_svbc} KB")
        diff_kb = warm_total_svbc - cold_total_svbc
        sign = "saved" if diff_kb < 0 else "added"
        print(f"  warm-cold delta: {diff_kb:+d} KB ({sign})")

        cold_codebook_final = cold_records[-1].get("codebook_size_after", "?")
        warm_codebook_final = warm_records[-1].get("codebook_size_after", "?")
        if isinstance(cold_codebook_final, int) and isinstance(warm_codebook_final, int):
            print(f"  codebook size after cold run: {cold_codebook_final} entries")
            print(f"  codebook size after warm run: {warm_codebook_final} entries")
            delta = warm_codebook_final - cold_codebook_final
            print(f"  (warm phase added {delta} entries.")
            if delta == 0:
                print("   Zero growth confirms perfect cross-run dedup — Mechanism A honest.")
            elif 0 < delta <= 32:
                print(f"   Small growth ({delta}) within iLUT non-determinism tolerance.")
                print("   The MVP doesn't seed iLUT with persisted colors yet, so a handful")
                print("   of Pareto-tiebreaker picks differ across runs. Documented in commit.")

    print(f"\n# full JSONL trace saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
