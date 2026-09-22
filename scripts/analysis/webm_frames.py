#!/usr/bin/env python3
"""Per-frame sizes and keyframe flags straight out of a WebM, no ffprobe.

Matroska is EBML: every element is an ID and a size, both variable-length
integers. Frames live in SimpleBlocks (0xA3) inside Clusters (0x1F43B675);
after the track number and a 16-bit timecode comes a flags byte whose top
bit marks a keyframe. Containers that are only walked into, not read, are
listed so their contents are visited."""
import sys

MASTERS = {0x18538067, 0x1F43B675, 0xA0}      # Segment, Cluster, BlockGroup

def vint(d, i, keep_marker):
    b = d[i]; n = 1; m = 0x80
    while n <= 8 and not (b & m): n += 1; m >>= 1
    v = b if keep_marker else b & (m - 1)
    for k in range(1, n): v = (v << 8) | d[i + k]
    return v, n

def walk(d, start, end, out):
    i = start
    while i < end:
        eid, n = vint(d, i, True); i += n
        size, n = vint(d, i, False); i += n
        if size == (1 << (7 * n)) - 1: size = end - i          # unknown size
        if eid in MASTERS: walk(d, i, min(end, i + size), out)
        elif eid == 0xA3:
            _, tn = vint(d, i, False)
            flags = d[i + tn + 2]
            out.append((size - tn - 3, bool(flags & 0x80)))
        i += size

d = open(sys.argv[1], 'rb').read()
frames = []
walk(d, 0, len(d), frames)
key = [s for s, k in frames if k]; inter = [s for s, k in frames if not k]
print(f"{len(frames)} quadros: {len(key)} chave (media {sum(key)/max(1,len(key)):.0f} B), "
      f"{len(inter)} inter (media {sum(inter)/max(1,len(inter)):.0f} B)")
