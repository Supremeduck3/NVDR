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

    return rec


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
    cols = ("image", "leaves_in", "leaves_out", "svbc_kb",
            "codebook_size", "ratio_pct", "ms")
    fmt = "{:<28} {:>10} {:>10} {:>8} {:>14} {:>8} {:>6}"
    print()
    print(fmt.format(*cols))
    print("-" * 92)
    for r in records:
        if not r.get("ok"):
            print(f"  FAIL  {r.get('image')}: {r.get('error')}")
            continue
        print(fmt.format(
            r["image"][:28],
            r.get("leaves_before", "-"),
            r.get("leaves_after", "-"),
            r.get("svbc_kb", "-"),
            r.get("codebook_size_after", "-"),
            r.get("ratio_pct", "-"),
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
