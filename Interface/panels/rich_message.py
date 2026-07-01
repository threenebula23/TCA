"""Parse assistant markdown into text / chart / diagram segments."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

try:
    from textual.widgets import Markdown as MarkdownWidget
except ImportError:  # pragma: no cover
    MarkdownWidget = None  # type: ignore[misc, assignment]

from rich.markdown import Markdown

from Interface.panels.blocks.chart_block import ChartBlock
from Interface.panels.blocks.diagram_block import DiagramBlock
from Interface.panels.ai_chat._helpers import _syntax_theme

_FENCE_RE = re.compile(
    r"```(?P<lang>lorne-chart|mermaid)\s*\n(?P<body>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def parse_rich_segments(text: str) -> List[Tuple[str, Any]]:
    """Split *text* into ``('text'|'chart'|'diagram', payload)`` segments."""
    src = text or ""
    if not src.strip():
        return []

    segments: List[Tuple[str, Any]] = []
    last = 0
    for match in _FENCE_RE.finditer(src):
        if match.start() > last:
            chunk = src[last:match.start()].strip()
            if chunk:
                segments.append(("text", chunk))
        lang = (match.group("lang") or "").lower()
        body = (match.group("body") or "").strip()
        if lang == "lorne-chart":
            try:
                segments.append(("chart", json.loads(body)))
            except json.JSONDecodeError:
                segments.append(("text", match.group(0)))
        elif lang == "mermaid":
            segments.append(("diagram", body))
        last = match.end()

    if last < len(src):
        chunk = src[last:].strip()
        if chunk:
            segments.append(("text", chunk))

    if not segments and src.strip():
        segments.append(("text", src.strip()))
    return segments


def compose_rich_body(text: str) -> ComposeResult:
    """Yield widgets for parsed assistant body segments."""
    theme = _syntax_theme()
    for kind, payload in parse_rich_segments(text):
        if kind == "chart" and isinstance(payload, dict):
            yield ChartBlock(payload)
        elif kind == "diagram" and isinstance(payload, str):
            yield DiagramBlock(payload)
        elif kind == "text" and str(payload).strip():
            body = str(payload).strip()[:120_000]
            if MarkdownWidget is not None:
                yield MarkdownWidget(body, classes="assistant-md")
            else:
                yield Static(Markdown(body, code_theme=theme), classes="assistant-md")


class RichAssistantBody(Vertical):
    """Vertical stack of markdown + chart/diagram blocks."""

    DEFAULT_CSS = """
    RichAssistantBody {
        height: auto;
        width: 100%;
    }
    RichAssistantBody .assistant-md {
        height: auto;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text or ""

    def compose(self) -> ComposeResult:
        yield from compose_rich_body(self._text)
