"""Mermaid-lite flowchart → ASCII box-drawing for TUI."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rich.text import Text

from textual.containers import Vertical
from textual.widgets import Static


_NODE_RE = re.compile(
    r"^\s*(?P<id>[A-Za-z0-9_]+)"
    r"(?:\[(?P<bracket>[^\]]*)\]|\((?P<paren>[^)]*)\))?"
    r"\s*$",
)


def _parse_node(token: str) -> Tuple[str, str]:
    token = (token or "").strip()
    m = _NODE_RE.match(token)
    if not m:
        return token, token
    nid = m.group("id") or token
    label = m.group("bracket") or m.group("paren") or nid
    return nid, label.strip() or nid


def _split_chain(line: str) -> List[Tuple[str, str]]:
    """Split ``A --> B --> C`` into node pairs."""
    parts = re.split(r"\s*(?:-->|---)\s*", line.strip())
    nodes: List[Tuple[str, str]] = []
    for part in parts:
        if part:
            nodes.append(_parse_node(part))
    return nodes


def render_mermaid_flowchart(source: str) -> str:
    """Render a minimal mermaid flowchart (LR/TB chains) to ASCII."""
    lines = [ln.strip() for ln in (source or "").splitlines() if ln.strip()]
    if not lines:
        return "▤ diagram: empty"

    direction = "LR"
    body_lines = lines
    first = lines[0].lower()
    if first.startswith("flowchart") or first.startswith("graph"):
        bits = lines[0].split()
        if len(bits) >= 2:
            direction = bits[1].upper()
        body_lines = lines[1:]

    edges: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    node_map: Dict[str, str] = {}
    for raw in body_lines:
        if "-->" not in raw and "---" not in raw:
            continue
        chain = _split_chain(raw)
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            node_map[a[0]] = a[1]
            node_map[b[0]] = b[1]
            edges.append((a, b))

    if not edges:
        return "▤ diagram: no edges (use flowchart LR + A --> B)"

    ordered: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for a, b in edges:
        if a[0] not in seen:
            ordered.append(a)
            seen.add(a[0])
        if b[0] not in seen:
            ordered.append(b)
            seen.add(b[0])

    if direction in ("TB", "BT"):
        out: List[str] = []
        for i, (_nid, label) in enumerate(ordered):
            box = _box(label)
            out.extend(box)
            if i < len(ordered) - 1:
                out.append("      │")
                out.append("      ▼")
        return "\n".join(out)

    rendered = [_box(n[1]) for n in ordered]
    return _join_horizontal(rendered)


def _box(label: str, width: int = 0) -> List[str]:
    text = (label or "?").replace("\n", " ").strip()
    inner_w = max(len(text), width, 4)
    top = "┌" + "─" * (inner_w + 2) + "┐"
    mid = "│ " + text.center(inner_w) + " │"
    bot = "└" + "─" * (inner_w + 2) + "┘"
    return [top, mid, bot]


def _join_horizontal(boxes: List[List[str]]) -> str:
    if not boxes:
        return ""
    heights = [len(b) for b in boxes]
    h = max(heights)
    padded = []
    for box in boxes:
        w = len(box[0]) if box else 0
        pad_top = (h - len(box)) // 2
        pad_bot = h - len(box) - pad_top
        lines = ["".ljust(w)] * pad_top + box + ["".ljust(w)] * pad_bot
        padded.append(lines)
    rows: List[str] = []
    for row_idx in range(h):
        parts = []
        for col_idx, col in enumerate(padded):
            parts.append(col[row_idx])
            if col_idx < len(padded) - 1:
                arrow_row = h // 2
                parts.append("──▶" if row_idx == arrow_row else "   ")
        rows.append("".join(parts))
    return "\n".join(rows)


class DiagramBlock(Vertical):
    DEFAULT_CSS = """
    DiagramBlock {
        height: auto;
        margin: 0 0 1 0;
        padding: 1 1;
        background: #12121A;
        border: round #2D2D3D;
    }
    DiagramBlock .diagram-title {
        height: auto;
        text-style: bold;
        margin: 0 0 1 0;
    }
    DiagramBlock .diagram-body {
        height: auto;
        color: #E5E7EB;
    }
    """

    def __init__(self, mermaid_source: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._source = mermaid_source or ""

    def compose(self):
        accent = "#8B5CF6"
        try:
            from Interface.ui_prefs import load_prefs
            accent = str(load_prefs().get("accent_color") or accent)
        except Exception:
            pass
        body = render_mermaid_flowchart(self._source)
        yield Static(Text.assemble(("◇ ", accent), ("Диаграмма", f"bold {accent}")), classes="diagram-title")
        yield Static(body, classes="diagram-body")
