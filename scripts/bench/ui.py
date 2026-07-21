
#!/usr/bin/env python3
"""bench/ui.py — Textual TUI for browsing bench/results.jsonl."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

REPO = Path(os.environ.get("NVDR_REPO", "/home/supremeduck008/NVDR"))
RESULTS = REPO / "bench" / "results.jsonl"
MANIFEST = REPO / "bench" / "preview" / "manifest.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {it["stem"]: it for it in data}


def jpg_to_halfblock(path: str, cols: int, rows: int) -> str:
    """Render a JPG into ANSI half-block characters (true colors)."""
    if Image is None:
        return "[red]Pillow not installed; cannot preview[/red]"
    if not Path(path).exists():
        return f"[red]missing: {path}[/red]"

    img = Image.open(path)
    target_w = cols * 2
    target_h = rows
    img = img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)

    pixels = img.load()
    assert pixels is not None
    lines: list[str] = []
    for y in range(0, target_h, 2):
        row: list[str] = []
        for x in range(cols):
            top = pixels[x, y]
            bot = pixels[x, y + 1] if y + 1 < target_h else top
            r1, g1, b1 = top
            r2, g2, b2 = bot
            row.append(
                f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m▀\x1b[0m"
            )
        lines.append("".join(row))
    return "\n".join(lines)


class BenchApp(App):
    CSS = """
    Screen { layout: vertical; }
    #dt { height: 40%; }
    #preview { height: 1fr; padding: 1 2; border: solid $accent; }
    """

    BINDINGS = [
        Binding("up", "cursor_up", "up", show=False),
        Binding("down", "cursor_down", "down", show=False),
        Binding("j", "cursor_down", "j", show=False),
        Binding("k", "cursor_up", "k", show=False),
        Binding("r", "reload", "reload"),
        Binding("q", "quit", "quit"),
    ]

    rows: reactive[list[dict[str, Any]]] = reactive(list, recompose=False)
    manifest: reactive[dict[str, dict[str, Any]]] = reactive(dict, recompose=False)

    def __init__(self, rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]):
        super().__init__()
        self.rows = rows
        self.manifest = manifest

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        dt = DataTable(id="dt", cursor_type="row", zebra_stripes=True)
        yield dt
        yield Static("select a row", id="preview")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "NVDR bench"
        self.sub_title = f"{len(self.rows)} rows · manifest={len(self.manifest)}"
        self._rebuild_table()
        self._refresh_preview()

    def _rebuild_table(self) -> None:
        dt = self.query_one("#dt", DataTable)
        dt.clear(columns=True)
        dt.add_columns(
            "image", "phase", "svbc_kb", "svg_kb",
            "cjpeg75", "cjpeg90", "ratio %", "codebook",
        )
        for r in self.rows:
            stem = Path(r["image"]).stem
            cb = r.get("codebook_size_after") or 0
            dt.add_row(
                stem,
                r.get("phase", "?"),
                f"{r.get('svbc_kb', 0):.1f}",
                f"{r.get('svg_kb', 0):.1f}",
                f"{r.get('cjpeg75_kb', 0):.1f}",
                f"{r.get('cjpeg90_kb', 0):.1f}",
                f"{r.get('ratio_pct', 0):.1f}",
                str(cb),
            )

    def _current_row(self) -> dict[str, Any] | None:
        dt = self.query_one("#dt", DataTable)
        if dt.row_count == 0:
            return None
        try:
            row_idx = dt.cursor_row
        except Exception:
            return None
        if row_idx is None or row_idx < 0:
            row_idx = 0
        if row_idx >= len(self.rows):
            return None
        return self.rows[row_idx]

    def _refresh_preview(self) -> None:
        row = self._current_row()
        prev = self.query_one("#preview", Static)
        if row is None:
            prev.update("[dim]no rows[/dim]")
            return
        stem = Path(row["image"]).stem
        m = self.manifest.get(stem, {})
        jpg = m.get("input")
        if not jpg:
            prev.update(f"[red]no preview manifest entry for stem={stem!r}[/red]")
            return
        # size = preview widget area minus padding/border. get from app size.
        size = self.size  # (cols, rows)
        cols = max(20, size.width - 4)
        rows = max(8, size.height - max(8, int(size.height * 0.4)) - 4)
        ascii_art = jpg_to_halfblock(jpg, cols, rows)
        meta = (
            f"[bold]{stem}[/bold]\n"
            f"phase={row.get('phase')}  svbc={row.get('svbc_kb')}kb  "
            f"svg={row.get('svg_kb')}kb  cjpeg75={row.get('cjpeg75_kb')}kb  "
            f"cjpeg90={row.get('cjpeg90_kb')}kb\n"
            f"ratio={row.get('ratio_pct')}%  codebook={row.get('codebook_size_after')}  "
            f"mtf_n8={row.get('mtf_hit_rate_n8')}%\n\n"
        )
        prev.update(meta + ascii_art)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._refresh_preview()

    def action_cursor_up(self) -> None:
        self.query_one("#dt", DataTable).action_cursor_up()
        self._refresh_preview()

    def action_cursor_down(self) -> None:
        self.query_one("#dt", DataTable).action_cursor_down()
        self._refresh_preview()

    def action_reload(self) -> None:
        self.rows = _load_jsonl(RESULTS)
        self.manifest = _load_manifest(MANIFEST)
        self.on_mount()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO),
                    help="NVDR repo root (also honors NVDR_REPO env)")
    args = ap.parse_args()
    repo_path = Path(args.repo)
    results_path = repo_path / "bench" / "results.jsonl"
    manifest_path = repo_path / "bench" / "preview" / "manifest.json"

    rows = _load_jsonl(results_path)
    manifest = _load_manifest(manifest_path)
    if not rows:
        print(f"no rows in {results_path}; run bench first", file=sys.stderr)
        return 2
    if not manifest:
        print(f"warning: manifest missing at {manifest_path}; previews disabled",
              file=sys.stderr)

    app = BenchApp(rows, manifest)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
