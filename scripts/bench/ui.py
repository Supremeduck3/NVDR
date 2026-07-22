#!/usr/bin/env python3
"""bench/ui.py — Textual TUI com suporte a renderização ANSI, troca de pasta e visualização em Tela Cheia."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Static

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

REPO = Path(os.environ.get("NVDR_REPO", str(Path(__file__).resolve().parents[2])))
RESULTS = REPO / "bench" / "results.jsonl"
MANIFEST = REPO / "bench" / "preview" / "manifest.json"
IMAGES_DIR = Path(os.environ.get("NVDR_IMAGES_DIR", r"Z:\samples_images"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
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
    data = json.loads(path.read_text(encoding="utf-8"))
    return {it["stem"]: it for it in data}


def _resolve_image_path(raw: str) -> str:
    """Procura a imagem dentro do IMAGES_DIR com suporte a caminhos do Windows."""
    if not raw:
        return ""
    p = Path(raw)
    if p.exists():
        return str(p)

    filename = p.name
    if IMAGES_DIR.exists():
        candidates = list(IMAGES_DIR.rglob(filename))
        if candidates:
            return str(candidates[0])

        if not p.suffix:
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidates = list(IMAGES_DIR.rglob(f"{filename}{ext}"))
                if candidates:
                    return str(candidates[0])

    return raw


def jpg_to_halfblock(path: str, cols: int, rows: int) -> str:
    """Renderiza um JPG/PNG em caracteres half-block ANSI (TrueColor)."""
    if Image is None:
        return "[bold red]Pillow não instalado; pré-visualização indisponível.[/bold red]"
    if not Path(path).exists():
        return f"[bold red]Arquivo ausente:[/bold red]\n[dim]{path}[/dim]"

    if cols < 2 or rows < 2:
        return ""

    try:
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
    except Exception as e:
        return f"[bold red]Erro ao abrir imagem:[/bold red] {e}"


class ImageModalScreen(ModalScreen):
    """Tela em modo Fullscreen/Modal para exibir a imagem expandida."""

    BINDINGS = [
        Binding("escape", "dismiss", "Voltar"),
        Binding("f", "dismiss", "Sair do Fullscreen"),
    ]

    CSS = """
    ImageModalScreen {
        align: center middle;
        background: $background 95%;
    }

    #modal-preview {
        width: 100%;
        height: 100%;
        border: heavy $accent;
        content-align: center middle;
    }
    """

    def __init__(self, ascii_art_text: Text | str):
        super().__init__()
        self.ascii_art_text = ascii_art_text

    def compose(self) -> ComposeResult:
        yield Static(self.ascii_art_text, id="modal-preview")


class BenchApp(App):
    TITLE = "NVDR Bench Viewer"

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #main-container {
        height: 1fr;
        width: 100%;
    }

    #table-container {
        width: 60%;
        height: 100%;
        border-right: heavy $accent;
    }

    #details-container {
        width: 40%;
        height: 100%;
        padding: 0 1;
    }

    DataTable {
        height: 100%;
        background: $surface;
    }

    DataTable > .datatable--header {
        text-style: bold;
        background: $accent-darken-2;
        color: $text;
    }

    #path-input {
        margin-bottom: 1;
        border: tall $accent;
    }

    #meta-card {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: round $primary;
        background: $panel;
    }

    #preview-card {
        height: 1fr;
        border: round $accent;
        background: $surface;
        content-align: center middle;
        overflow: hidden;
    }
    """

    BINDINGS = [
        Binding("up", "cursor_up", "Cima", show=False),
        Binding("down", "cursor_down", "Baixo", show=False),
        Binding("j", "cursor_down", "Navegar ↓", show=True),
        Binding("k", "cursor_up", "Navegar ↑", show=True),
        Binding("f", "toggle_fullscreen", "Tela Cheia", show=True),
        Binding("i", "focus_path_input", "Mudar Pasta Imagens", show=True),
        Binding("r", "reload", "Recarregar", show=True),
        Binding("q", "quit", "Sair", show=True),
    ]

    rows: reactive[list[dict[str, Any]]] = reactive(list, recompose=False)
    manifest: reactive[dict[str, dict[str, Any]]] = reactive(dict, recompose=False)

    def __init__(self, rows: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]):
        super().__init__()
        self.rows = rows
        self.manifest = manifest

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="table-container"):
                yield DataTable(id="dt", cursor_type="row", zebra_stripes=True)
            with Vertical(id="details-container"):
                yield Input(
                    value=str(IMAGES_DIR),
                    placeholder="Cole o caminho da pasta de imagens e pressione Enter...",
                    id="path-input"
                )
                yield Static("[dim]Selecione uma linha para ver detalhes[/dim]", id="meta-card")
                yield Static("", id="preview-card")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{len(self.rows)} itens · Manifest ({len(self.manifest)})"
        self._rebuild_table()
        self._refresh_selection()

    def _rebuild_table(self) -> None:
        dt = self.query_one("#dt", DataTable)
        dt.clear(columns=True)
        dt.add_columns(
            "Imagem", "Fase", "SVBC (KB)", "SVG (KB)",
            "JPEG 75", "JPEG 90", "Ratio %", "Codebook"
        )
        for r in self.rows:
            stem = Path(r["image"]).stem
            cb = r.get("codebook_size_after") or 0
            dt.add_row(
                stem,
                str(r.get("phase", "?")),
                f"{r.get('svbc_kb', 0):.1f}",
                f"{r.get('svg_kb', 0):.1f}",
                f"{r.get('cjpeg75_kb', 0):.1f}",
                f"{r.get('cjpeg90_kb', 0):.1f}",
                f"{r.get('ratio_pct', 0):.1f}%",
                str(cb),
            )

    def _current_row(self) -> dict[str, Any] | None:
        dt = self.query_one("#dt", DataTable)
        if dt.row_count == 0:
            return None
        row_idx = dt.cursor_row
        if row_idx is None or row_idx < 0 or row_idx >= len(self.rows):
            return None
        return self.rows[row_idx]

    def _refresh_selection(self) -> None:
        row = self._current_row()
        meta_widget = self.query_one("#meta-card", Static)
        prev_widget = self.query_one("#preview-card", Static)

        if row is None:
            meta_widget.update("[dim]Nenhum registro selecionado[/dim]")
            prev_widget.update("")
            return

        raw_image_path = row.get("image", "")
        stem = Path(raw_image_path).stem
        m = self.manifest.get(stem, {})

        meta_text = (
            f"[bold primary]{stem}[/bold primary]\n"
            f"[dim]Fase:[/dim] [bold]{row.get('phase', '?')}[/bold]\n\n"
            f"[b]SVBC:[/b] {row.get('svbc_kb', 0):.1f} KB   | [b]SVG:[/b] {row.get('svg_kb', 0):.1f} KB\n"
            f"[b]JPEG 75:[/b] {row.get('cjpeg75_kb', 0):.1f} KB| [b]JPEG 90:[/b] {row.get('cjpeg90_kb', 0):.1f} KB\n"
            f"[b]Razão:[/b] [green]{row.get('ratio_pct', 0):.1f}%[/green]   | "
            f"[b]Codebook:[/b] {row.get('codebook_size_after', 0)} | "
            f"[b]MTF n8:[/b] {row.get('mtf_hit_rate_n8', 0)}%"
        )
        meta_widget.update(meta_text)

        jpg = m.get("input")
        if not jpg or not Path(jpg).exists():
            jpg = _resolve_image_path(raw_image_path)

        if not jpg or not Path(jpg).exists():
            prev_widget.update(f"[bold red]Sem entrada para:[/bold red]\n[yellow]{stem}[/yellow]")
            return

        region = prev_widget.content_region
        cols = max(10, region.width)
        rows = max(5, region.height)

        ascii_art = jpg_to_halfblock(jpg, cols, rows)
        
        if ascii_art.startswith("[bold red]"):
            prev_widget.update(ascii_art)
        else:
            prev_widget.update(Text.from_ansi(ascii_art))

    def action_toggle_fullscreen(self) -> None:
        """Ação da tecla 'f' para expandir a imagem em tela cheia."""
        row = self._current_row()
        if not row:
            return

        raw_image_path = row.get("image", "")
        stem = Path(raw_image_path).stem
        m = self.manifest.get(stem, {})

        jpg = m.get("input")
        if not jpg or not Path(jpg).exists():
            jpg = _resolve_image_path(raw_image_path)

        if not jpg or not Path(jpg).exists():
            self.notify("Não há imagem para expandir.", severity="warning")
            return

        cols = max(20, self.size.width)
        rows = max(10, self.size.height - 2)

        ascii_art = jpg_to_halfblock(jpg, cols, rows)
        if ascii_art.startswith("[bold red]"):
            content = ascii_art
        else:
            content = Text.from_ansi(ascii_art)

        self.push_screen(ImageModalScreen(content))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "path-input":
            new_path = Path(event.value.strip())
            if new_path.exists() and new_path.is_dir():
                global IMAGES_DIR
                IMAGES_DIR = new_path
                self.notify(f"Pasta alterada para: {new_path}", title="Sucesso")
                self.query_one("#dt", DataTable).focus()
                self._refresh_selection()
            else:
                self.notify(f"Diretório inválido ou inexistente: {new_path}", severity="error", title="Erro")

    def action_focus_path_input(self) -> None:
        self.query_one("#path-input", Input).focus()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._refresh_selection()

    def on_resize(self) -> None:
        try:
            self._refresh_selection()
        except Exception:
            pass

    def action_cursor_up(self) -> None:
        self.query_one("#dt", DataTable).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#dt", DataTable).action_cursor_down()

    def action_reload(self) -> None:
        self.rows = _load_jsonl(RESULTS)
        self.manifest = _load_manifest(MANIFEST)
        self.sub_title = f"{len(self.rows)} itens · Manifest ({len(self.manifest)})"
        self._rebuild_table()
        self._refresh_selection()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO),
                    help="Raiz do repositório NVDR")
    ap.add_argument("--images-dir", default=None,
                    help="Diretório de pré-visualização de imagens")
    args = ap.parse_args()
    repo_path = Path(args.repo)
    results_path = repo_path / "bench" / "results.jsonl"
    manifest_path = repo_path / "bench" / "preview" / "manifest.json"

    global IMAGES_DIR
    if args.images_dir:
        IMAGES_DIR = Path(args.images_dir)

    rows = _load_jsonl(results_path)
    manifest = _load_manifest(manifest_path)
    if not rows:
        print(f"Nenhum resultado encontrado em {results_path}; execute o benchmark primeiro.", file=sys.stderr)
        return 2

    app = BenchApp(rows, manifest)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())