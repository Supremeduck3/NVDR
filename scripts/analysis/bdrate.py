#!/usr/bin/env python3
"""BD-rate of one set of rdcurve outputs against another.

    bdrate.py <ref_dir> <test_dir>

Each directory holds one file per image, lines of `q bytes psnrY psnrRGB
seconds` as rdcurve prints them. Prints the BD-rate per image and the
mean, on PSNR-Y and on PSNR-RGB (negative: the test needs fewer bytes),
the way scripts/video_bench_lib.mjs computes it: a cubic through log
bytes against PSNR, integrated over the overlap.
"""
import math
import pathlib
import sys


def solve(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def cubic_fit(xs, ys):
    a = [[sum(x ** (i + j) for x in xs) for j in range(4)] for i in range(4)]
    b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(4)]
    return solve(a, b)


def integral(c, lo, hi):
    f = lambda x: sum(c[i] * x ** (i + 1) / (i + 1) for i in range(4))
    return f(hi) - f(lo)


def bd(ref, test, key):
    lo = max(min(p[key] for p in ref), min(p[key] for p in test))
    hi = min(max(p[key] for p in ref), max(p[key] for p in test))
    if hi - lo < 0.3:
        return None
    cr = cubic_fit([p[key] for p in ref], [math.log(p["bytes"]) for p in ref])
    ct = cubic_fit([p[key] for p in test], [math.log(p["bytes"]) for p in test])
    return (math.exp((integral(ct, lo, hi) - integral(cr, lo, hi)) / (hi - lo)) - 1) * 100


def load(path):
    pts = []
    for line in path.read_text().split("\n"):
        f = line.split()
        if len(f) >= 5:
            pts.append({"bytes": int(f[1]), "y": float(f[2]), "rgb": float(f[3]), "s": float(f[4])})
    return pts


ref_dir, test_dir = map(pathlib.Path, sys.argv[1:3])
ys, rgbs = [], []
tr = tt = 0.0
for rf in sorted(ref_dir.glob("*.txt")):
    tf = test_dir / rf.name
    if not tf.exists():
        continue
    r, t = load(rf), load(tf)
    y, c = bd(r, t, "y"), bd(r, t, "rgb")
    tr += sum(p["s"] for p in r)
    tt += sum(p["s"] for p in t)
    ys.append(y)
    rgbs.append(c)
    print(f"{rf.name:28s} PSNR-Y {y:+6.2f}%   PSNR-RGB {c:+6.2f}%")
print(f"{'mean':28s} PSNR-Y {sum(ys) / len(ys):+6.2f}%   PSNR-RGB {sum(rgbs) / len(rgbs):+6.2f}%")
print(f"encode time: {tr:.1f}s -> {tt:.1f}s ({tt / tr:.2f}x)")
