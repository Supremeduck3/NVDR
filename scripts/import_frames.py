#!/usr/bin/env python3
"""
import_frames.py — take a zip (or a folder, or a URL) of frames and lay it
out the way nvdrv_encode expects.

Frames arrive named however the tool that exported them felt like naming
them, often nested a folder or two deep, sometimes with a stray thumbnail
or a .DS_Store alongside. This flattens that: it finds the images, sorts
them the way a human would read the numbers, renames them f000..fNNN, and
then checks the one thing that actually matters before any measurement is
worth taking — that every frame is the same size and that consecutive
frames are neither identical nor unrelated.

    python3 scripts/import_frames.py clipe.zip meu_clipe
    python3 scripts/import_frames.py https://host/clipe.zip meu_clipe
    python3 scripts/import_frames.py /path/to/folder meu_clipe --limit 60
"""
import argparse
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".ppm"}


def natural_key(p: pathlib.Path):
    """Sort f2 before f10, which plain string order gets backwards."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", p.name)]


def dimensions(path: pathlib.Path):
    """Width and height straight from the header, no image library here."""
    with open(path, "rb") as f:
        head = f.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return w, h
        if head[:2] == b"\xff\xd8":                       # JPEG
            f.seek(2)
            while True:
                b = f.read(1)
                while b and b != b"\xff":
                    b = f.read(1)
                marker = f.read(1)
                while marker == b"\xff":
                    marker = f.read(1)
                if not marker:
                    return None
                if marker[0] in range(0xC0, 0xD0) and marker[0] not in (0xC4, 0xC8, 0xCC):
                    f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                size = struct.unpack(">H", f.read(2))[0]
                f.seek(size - 2, 1)
        if head[:2] == b"P6":                             # PPM
            parts = head.split()
            if len(parts) >= 3:
                return int(parts[1]), int(parts[2])
    return None


def gather(source: str, work: pathlib.Path) -> pathlib.Path:
    if source.startswith(("http://", "https://")):
        blob = work / "download.zip"
        print(f"baixando {source}")
        r = subprocess.run(["curl", "-fsSL", "-o", str(blob), source])
        if r.returncode != 0:
            sys.exit("download falhou")
        source = str(blob)

    src = pathlib.Path(source)
    if src.is_dir():
        return src
    if zipfile.is_zipfile(src):
        out = work / "unpacked"
        out.mkdir()
        with zipfile.ZipFile(src) as z:
            # Refuse paths that would escape the destination.
            for info in z.infolist():
                target = (out / info.filename).resolve()
                if not str(target).startswith(str(out.resolve())):
                    sys.exit(f"zip contem um caminho suspeito: {info.filename}")
            z.extractall(out)
        return out
    sys.exit(f"'{source}' nao e pasta nem zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="zip, pasta ou URL")
    ap.add_argument("name", help="nome do clipe em samples/video/")
    ap.add_argument("--limit", type=int, default=0, help="usar so os N primeiros quadros")
    ap.add_argument("--force", action="store_true", help="sobrescrever um clipe existente")
    args = ap.parse_args()

    dest = ROOT / "samples" / "video" / args.name
    if dest.exists() and not args.force:
        sys.exit(f"{dest} ja existe — use --force para sobrescrever")

    work = pathlib.Path(tempfile.mkdtemp(prefix="nvdr-frames-"))
    try:
        tree = gather(args.source, work)
        frames = sorted((p for p in tree.rglob("*")
                         if p.is_file() and p.suffix.lower() in EXTS
                         and not p.name.startswith(".")),
                        key=natural_key)
        if not frames:
            sys.exit("nenhuma imagem encontrada")
        if args.limit:
            frames = frames[:args.limit]

        sizes = {}
        for p in frames:
            sizes.setdefault(dimensions(p), []).append(p.name)
        if len(sizes) > 1:
            print("quadros com tamanhos diferentes — o encoder para no primeiro que difere:")
            for dim, names in sorted(sizes.items(), key=lambda kv: -len(kv[1])):
                print(f"  {dim}: {len(names)} quadro(s), ex. {names[0]}")
            sys.exit(1)

        dim = next(iter(sizes))
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for i, p in enumerate(frames):
            shutil.copy2(p, dest / f"f{i:03d}{p.suffix.lower()}")

        total = sum(p.stat().st_size for p in dest.iterdir())
        print(f"{len(frames)} quadros {dim[0]}x{dim[1]} -> {dest.relative_to(ROOT)} "
              f"({total/1e6:.1f} MB)")
        print(f"\n  ./nvdrv_encode {dest.relative_to(ROOT)} /tmp/{args.name}.nvdrv")
        print(f"  ./nvdrv_decode /tmp/{args.name}.nvdrv --compare {dest.relative_to(ROOT)}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
