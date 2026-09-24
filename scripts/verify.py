#!/usr/bin/env python3
"""
verify.py — regression gate for NVDR.

Checks the things that have to hold before any measurement on top of them
can be trusted:

  1. Determinism — the encoder runs twice and the two containers must be
     byte-identical. Without this an A/B measurement is measuring noise.
  2. Quality — every sample decoded back and compared against a recorded
     baseline, per image. A global floor is too weak: the samples sit
     anywhere between 21 and 30 dB, so a drop that matters on one is
     invisible against the others.
  3. Size — container bytes against the same baseline, so a change that
     buys quality with bytes shows up as what it is rather than as a win.
  4. Truncation — the guarantee the format exists for. Cut at several
     points; every cut past the anchor must decode, and quality must not
     fall as bytes are added.
  5. C against JS — the two decoders must produce identical pixels, at
     every level and on cut files, or the viewer shows something the
     container does not contain.

Synthetic probes are generated here rather than committed. They exist
because the sample set is six photographs, and photographs hid a
regression that cost a flat-shape image 5.5 dB: a tree that saturates
reacts to the tolerance knob in the opposite direction to one that does
not. `blocos` saturates immediately, `circulos` is edge-dominated,
`degrade` is pure ramp.

Usage:
    python3 scripts/verify.py                # check against the baseline
    python3 scripts/verify.py --update       # record a new baseline
    python3 scripts/verify.py --psnr-slack 0.5 --size-slack 0.05

Exit code is non-zero if any check fails, so it works as a CI gate.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "baseline.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".ppm"}
PSNR_RE = re.compile(r"psnr\s+([0-9.]+)\s*dB")


def find_binary(name):
    for candidate in (ROOT / name, ROOT / f"{name}.exe"):
        if candidate.exists():
            return candidate
    sys.exit(f"{name} not built — run `make` first")


def write_ppm(path, width, height, fn):
    rows = bytearray()
    for y in range(height):
        for x in range(width):
            rows += bytes(fn(x, y))
    path.write_bytes(b"P6\n%d %d\n255\n" % (width, height) + bytes(rows))


def make_probes(directory):
    """The three failure modes six photographs do not cover."""
    size = 256

    palette = [(255, 255, 255), (20, 20, 20), (200, 40, 40), (40, 80, 200),
               (250, 220, 40), (30, 160, 90), (150, 60, 180), (240, 240, 240),
               (10, 10, 10), (90, 90, 90), (255, 140, 0), (0, 150, 160),
               (180, 180, 180), (60, 20, 20), (230, 230, 180), (35, 35, 70)]

    def blocos(x, y):
        return palette[(y * 4 // size) * 4 + (x * 4 // size)]

    circles = [(65, 65, 40, (220, 40, 40)), (190, 75, 30, (40, 90, 220)),
               (100, 190, 50, (250, 200, 30)), (200, 200, 25, (30, 160, 90))]

    def circulos(x, y):
        for cx, cy, r, colour in circles:
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                return colour
        return (245, 245, 245)

    def degrade(x, y):
        return (255 * x // (size - 1), 255 * y // (size - 1), 128)

    out = []
    for name, fn in (("blocos", blocos), ("circulos", circulos),
                     ("degrade", degrade)):
        path = directory / f"{name}.ppm"
        write_ppm(path, size, size, fn)
        out.append(path)
    return out


def make_sequence(directory, frames=12, size=128):
    """A frame sequence, for the same reason the still probes exist.

    A codec that predicts from the wrong reference looks correct on frame 1
    and walks away from the source by frame 12, so the sequence has to be
    long enough for drift to show and cheap enough to run every time. The
    background is textured so the prediction has something to be wrong
    about, and the block moves so the motion search has work to do."""
    paths = []
    for n in range(frames):
        bx, by = 20 + n * 4, 40
        def fn(x, y, bx=bx, by=by):
            if bx <= x < bx + 30 and by <= y < by + 30:
                return (230, 60, 40)
            v = 40 + ((x * 7 + y * 13) % 53) + ((x // 16 + y // 16) % 3) * 40
            return (v, min(255, v + 25), 255 - v)
        path = directory / f"seq{n:03d}.ppm"
        write_ppm(path, size, size, fn)
        paths.append(path)
    return paths


def check_sequence(tmp, psnr_slack):
    """Encode the sequence twice and report bytes, quality and drift.

    The loop run holds everything that trades quality over time still:
    equal quantisers for intra, P and B frames, no skipped blocks, and no
    look-ahead refining the intra frame.
    What it is after is a reference that walks away from the source, and
    by design the defaults settle a little below the intra frame. The
    default run is the encoder as it ships, held to the baseline on bytes
    and quality."""
    enc = find_binary("nvdrv_encode")
    dec = find_binary("nvdrv_decode")
    srcdir = tmp / "seq"
    srcdir.mkdir(exist_ok=True)
    make_sequence(srcdir)

    results, problems = {}, []
    for name, args in (("loop", ["--gop", "0", "--q", "24", "--pred-q", "24", "--b-q-step", "0",
                                 "--skip-k", "0", "--tpl", "0"]),
                       ("default", ["--gop", "0"])):
        out = tmp / f"seq_{name}.nvdrv"
        r = subprocess.run([str(enc), str(srcdir), str(out)] + args,
                           capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            return None, [r.stderr.strip() or f"{name} sequence encode failed"]

        # Determinism, same rule as the stills.
        out_b = tmp / f"seq_{name}_b.nvdrv"
        subprocess.run([str(enc), str(srcdir), str(out_b)] + args,
                       capture_output=True, text=True)
        if not (out_b.exists() and
                hashlib.sha256(out.read_bytes()).hexdigest()
                == hashlib.sha256(out_b.read_bytes()).hexdigest()):
            problems.append(f"{name} sequence encoder is not deterministic")

        r = subprocess.run([str(dec), str(out), "--compare", str(srcdir)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, [r.stderr.strip() or f"{name} sequence decode failed"]
        per_frame = [float(m) for m in
                     re.findall(r"(?:INTRA|pred|bi)\s+([0-9.]+) dB", r.stdout)]
        if len(per_frame) < 12:
            return None, [f"{name}: only {len(per_frame)} of 12 frames decoded"]

        # The browser player has to hold the same state as the C decoder
        # after every frame, or it drifts from the file while each frame
        # looks fine.
        cc = subprocess.run(["node", str(ROOT / "scripts" / "crosscheck_seq.mjs"), str(out)],
                            capture_output=True, text=True)
        if cc.returncode != 0:
            bad = [l for l in cc.stdout.splitlines() if "identical" not in l]
            problems.append(f"{name} C vs JS " + (bad[0] if bad else cc.stderr.strip()[:80]))

        drift = min(per_frame[1:]) - per_frame[0]
        # Drift: the last frames must not be worse than the first. Predicting
        # from the wrong reference shows up here and nowhere else.
        if name == "loop" and drift < -psnr_slack:
            problems.append(f"drifts {drift:+.2f} dB by the end")
        results[name] = {"bytes": out.stat().st_size,
                         "psnr": round(sum(per_frame) / len(per_frame), 2),
                         "drift": round(drift, 2)}
    return results, problems


def check_album(tmp, images):
    """Two albums. The photographs, which share nothing but statistics: the
    fluid context must never make them bigger. And the gate's frame
    sequence packed as stills, a burst: its images must come out predicted
    from each other, and at equal quality smaller than coded alone. C and
    JS must agree on every image of both, whole and cut."""
    tool = find_binary("nvdr_album")
    burst_dir = tmp / "burst"
    burst_dir.mkdir(exist_ok=True)
    burst = make_sequence(burst_dir)
    photos = [str(p) for p in images if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    results, problems = {}, []
    for name, files in (("fotos", photos), ("rajada", [str(p) for p in burst])):
        out = tmp / f"album_{name}.nvda"
        r = subprocess.run([str(tool), "pack", str(out)] + files, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            return None, [r.stderr.strip() or f"{name} album pack failed"]
        m = re.search(r"total\s+(\d+)\s+(\d+)", r.stdout)
        warm, cold = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        predicted = len(re.findall(r"\sprevista\s", r.stdout))
        if warm > cold:
            problems.append(f"{name}: album bigger than its images alone ({warm} > {cold})")
        if name == "rajada" and predicted < len(files) - 1:
            problems.append(f"rajada: only {predicted} of {len(files) - 1} images predicted")
        cc = subprocess.run(["node", str(ROOT / "scripts" / "crosscheck_album.mjs"), str(out)],
                            capture_output=True, text=True)
        if cc.returncode != 0:
            bad = [l for l in cc.stdout.splitlines() if "identical" not in l]
            problems.append(f"{name} C vs JS " + (bad[0] if bad else cc.stderr.strip()[:80]))
        results[name] = {"bytes": out.stat().st_size, "warm": warm, "cold": cold}
    return results, problems


def encode(encoder, image, out):
    r = subprocess.run([str(encoder), str(image), str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        return None, (r.stderr.strip() or "encoder failed")
    return out, None


def decode_psnr(decoder, container, image, tmp, level=None):
    """PSNR of the decoded container, or None when it refuses to decode."""
    args = [str(decoder), str(container), str(tmp / "out.ppm"),
            "--compare", str(image)]
    if level is not None:
        args += ["--level", str(level)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    m = PSNR_RE.search(r.stdout)
    return float(m.group(1)) if m else None


def check_truncation(decoder, container, image, tmp):
    """Every cut past the anchor decodes, and quality never falls."""
    data = container.read_bytes()
    cut = tmp / "cut.nvdr"
    previous, decoded, problems = None, 0, []
    for pct in (5, 12, 20, 30, 45, 60, 75, 90, 100):
        cut.write_bytes(data[: len(data) * pct // 100])
        psnr = decode_psnr(decoder, cut, image, tmp)
        if psnr is None:
            # Below the anchor stream there is no picture, and saying so is
            # the contract. Past a cut that decoded, it is a failure.
            if decoded:
                problems.append(f"{pct}% refused after {previous:.2f} dB decoded")
            continue
        decoded += 1
        if previous is not None and psnr < previous - 0.15:
            problems.append(f"{pct}% fell {previous:.2f} -> {psnr:.2f} dB")
        previous = psnr
    if not decoded:
        problems.append("no cut decoded at all")
    return problems


def check_crosscheck(container):
    script = ROOT / "scripts" / "crosscheck.mjs"
    r = subprocess.run(["node", str(script), str(container)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad = [l for l in r.stdout.splitlines()
               if "identical" not in l and "both refuse" not in l]
        return bad[:2] or [r.stderr.strip()[:120] or "crosscheck failed"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "samples"))
    ap.add_argument("--update", action="store_true",
                    help="record the current numbers as the new baseline")
    ap.add_argument("--psnr-slack", type=float, default=0.25,
                    help="dB a sample may lose before it fails")
    ap.add_argument("--size-slack", type=float, default=0.03,
                    help="fraction a container may grow before it fails")
    ap.add_argument("--skip-crosscheck", action="store_true")
    args = ap.parse_args()

    encoder = find_binary("nvdr_encode")
    decoder = find_binary("nvdr_decode")

    images = sorted(p for p in pathlib.Path(args.images).iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"no images found in {args.images}")

    baseline = {}
    if BASELINE.exists() and not args.update:
        baseline = json.loads(BASELINE.read_text())

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nvdr-verify-"))
    failures, recorded = [], {}
    try:
        images += make_probes(tmp)

        print(f"{'image':<26} {'bytes':>9} {'PSNR':>8} {'vs base':>16} "
              f"{'det':>7}  truncation / C-vs-JS")
        print("-" * 96)

        for image in images:
            name = image.name
            first, err = encode(encoder, image, tmp / "a.nvdr")
            if err:
                failures.append(f"{name}: {err}")
                print(f"{name:<26} ENCODE FAILED: {err}")
                continue
            payload = first.read_bytes()
            first = tmp / f"{image.stem}.nvdr"
            first.write_bytes(payload)

            second, err = encode(encoder, image, tmp / "b.nvdr")
            stable = not err and (
                hashlib.sha256(payload).hexdigest()
                == hashlib.sha256(second.read_bytes()).hexdigest())
            if not stable:
                failures.append(f"{name}: encoder is not deterministic")

            size = len(payload)
            psnr = decode_psnr(decoder, first, image, tmp)
            if psnr is None:
                failures.append(f"{name}: complete container does not decode")
                print(f"{name:<26} {size:>9} {'-':>8}  DECODE FAILED")
                continue
            recorded[name] = {"bytes": size, "psnr": round(psnr, 2)}

            delta = ""
            if name in baseline:
                was = baseline[name]
                d_psnr = psnr - was["psnr"]
                d_size = (size - was["bytes"]) / was["bytes"]
                delta = f"{d_psnr:+.2f}dB {d_size:+.1%}"
                if d_psnr < -args.psnr_slack:
                    failures.append(
                        f"{name}: PSNR {was['psnr']:.2f} -> {psnr:.2f} dB")
                if d_size > args.size_slack:
                    failures.append(
                        f"{name}: {was['bytes']} -> {size} bytes ({d_size:+.1%})")
            elif baseline:
                delta = "new"

            notes = check_truncation(decoder, first, image, tmp)
            if notes:
                failures += [f"{name}: {n}" for n in notes]
            if not args.skip_crosscheck:
                cc = check_crosscheck(first)
                if cc:
                    failures += [f"{name}: C vs JS {c}" for c in cc]
                    notes += cc

            print(f"{name:<26} {size:>9} {psnr:>7.2f}dB {delta:>16} "
                  f"{'stable' if stable else 'VARIES':>7}  "
                  f"{'ok' if not notes else '; '.join(notes)[:34]}")

        albums, album_problems = check_album(tmp, images)
        failures += [f"album: {p}" for p in album_problems]
        for name, album in (albums or {}).items():
            key = f"<album {name}>"
            recorded[key] = {"bytes": album["bytes"], "psnr": 0.0}
            delta = ""
            if key in baseline:
                was = baseline[key]
                d_size = (album["bytes"] - was["bytes"]) / was["bytes"]
                delta = f"{d_size:+.1%}"
                if d_size > args.size_slack:
                    failures.append(f"{key}: {was['bytes']} -> {album['bytes']} bytes ({d_size:+.1%})")
            elif baseline:
                delta = "new"
            gain = (album["warm"] - album["cold"]) / album["cold"] if album["cold"] else 0
            print(f"{key:<26} {album['bytes']:>9} {'':>8} {delta:>16} {'stable':>7}  "
                  f"{'ok' if not album_problems else '; '.join(album_problems)[:34]} "
                  f"(contra sozinhas {gain:+.1%})")

        seqs, seq_problems = check_sequence(tmp, args.psnr_slack)
        failures += [f"sequence: {p}" for p in seq_problems]
        for name, seq in (seqs or {}).items():
            key = "<sequence>" if name == "loop" else "<sequence default>"
            recorded[key] = {k: seq[k] for k in ("bytes", "psnr")}
            delta = ""
            if key in baseline:
                was = baseline[key]
                d_psnr = seq["psnr"] - was["psnr"]
                d_size = (seq["bytes"] - was["bytes"]) / was["bytes"]
                delta = f"{d_psnr:+.2f}dB {d_size:+.1%}"
                if d_psnr < -args.psnr_slack:
                    failures.append(
                        f"{key}: PSNR {was['psnr']:.2f} -> {seq['psnr']:.2f} dB")
                if d_size > args.size_slack:
                    failures.append(
                        f"{key}: {was['bytes']} -> {seq['bytes']} bytes ({d_size:+.1%})")
            elif baseline:
                delta = "new"
            label = f"<sequence {name}> 12 fr"
            print(f"{label:<26} {seq['bytes']:>9} {seq['psnr']:>7.2f}dB "
                  f"{delta:>16} {'stable':>7}  "
                  f"{'ok' if not seq_problems else '; '.join(seq_problems)[:34]} "
                  f"(deriva {seq['drift']:+.2f} dB)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if args.update:
        BASELINE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"\nbaseline written to {BASELINE.relative_to(ROOT)} "
              f"({len(recorded)} entries)")
        return 0

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all {len(recorded)} samples passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
