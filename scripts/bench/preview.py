#!/usr/bin/env python3
"""
bench/preview.py — Gera pré-visualizações estáveis em SVG/SVBC.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("NVDR_REPO", str(Path(__file__).resolve().parents[2])))

BINARY_BASE = REPO / "image_to_svg"
BINARY = BINARY_BASE.with_suffix(".exe") if BINARY_BASE.with_suffix(".exe").exists() else BINARY_BASE

RESULTS = REPO / "bench" / "results.jsonl"
PREVIEW_DIR = REPO / "bench" / "preview"

DEFAULT_IMG_DIRS = [
    Path(os.environ.get("NVDR_BENCH_IMAGES", str(REPO / "samples_images"))),
    Path(r"Z:\samples_images"),
    REPO / "samples",
    REPO / "samples_images",
]


def resolve_image(name: str, custom_dir: Path | None = None) -> Path | None:
    """Procura a imagem original em várias pastas conhecidas."""
    p_name = Path(name).name
    candidates = []
    
    if custom_dir:
        candidates.append(custom_dir)
        
    env = os.environ.get("NVDR_BENCH_IMAGES")
    if env:
        candidates.append(Path(env))
    
    candidates.extend(DEFAULT_IMG_DIRS)
    
    for d in candidates:
        if not d.exists():
            continue
        p1 = d / name
        p2 = d / p_name
        if p1.exists():
            return p1
        if p2.exists():
            return p2
        # Tenta buscar ignorando maiúsculas/minúsculas ou em subpastas
        found = list(d.rglob(p_name))
        if found:
            return found[0]
            
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Reconstrói todos os previews")
    ap.add_argument("--images-dir", default=None, help="Diretório onde estão as imagens originais")
    args = ap.parse_args()

    custom_dir = Path(args.images_dir) if args.images_dir else None

    if not BINARY.exists():
        print(f"erro: binário não encontrado em {BINARY}; compile primeiro", file=sys.stderr)
        return 2
    if not RESULTS.exists():
        print(f"erro: {RESULTS} não encontrado; rode o bench primeiro", file=sys.stderr)
        return 2

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    with RESULTS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "image" in r:
                    images.append(r["image"])
            except json.JSONDecodeError:
                continue
                
    images = sorted(set(images))
    print(f"# {len(images)} imagens únicas em {RESULTS}", file=sys.stderr)

    image_map = {}
    for name in images:
        src = resolve_image(name, custom_dir)
        if not src:
            print(f"  skip {name!r}: imagem original não encontrada", file=sys.stderr)
            continue
            
        stem = Path(name).stem
        image_map[stem] = str(src)
        out_base = PREVIEW_DIR / Path(name).stem
        raw_svg = Path(str(out_base))
        raw_svbc = Path(str(out_base) + ".svbc")

        if not args.force and raw_svg.exists() and raw_svbc.exists():
            print(f"  ok   {name} (cache)", file=sys.stderr)
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

    manifest_path = PREVIEW_DIR / "manifest.json"
    items = []
    for svbc in sorted(PREVIEW_DIR.glob("*.svbc")):
        stem = svbc.name[:-len(".svbc")]
        out_base = PREVIEW_DIR / stem
        svg = out_base
        
        input_file = image_map.get(stem)
        
        items.append({
            "stem": stem,
            "input": input_file,
            "svg": str(svg) if svg.exists() else None,
            "svg_kb": round(svg.stat().st_size / 1024.0, 1) if svg.exists() else None,
            "svbc_kb": round(svbc.stat().st_size / 1024.0, 1),
        })

    manifest_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"# manifest gerado com sucesso: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())