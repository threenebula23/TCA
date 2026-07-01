"""Collapsible widget for model reasoning / thought segments.

Shown in the chat stream whenever the model emits <think>…</think> content.
Collapsed by default so it doesn't dominate the conversation; expands on click.
"""
from __future__ import annotations

from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Collapsible, Static


def _accent_color() -> str:
    try:
        from Interface.ui_prefs import load_prefs
        from Interface.themes import get_theme
        prefs = load_prefs()
        theme = get_theme(str(prefs.get("theme", "Purple Dark")))
        return str(prefs.get("accent_color") or theme.get("accent") or "#8B5CF6")
    except Exception:
        return "#8B5CF6"


class ThinkingBlock(Vertical):
    """Collapsible card displaying one model thought segment.

    Header shows "· Мысль модели" with a char-count badge.
    Body renders the full thought text, trimmed to 2000 chars.
    Collapsed by default.
    """

    DEFAULT_CSS = """
    ThinkingBlock {
        height: auto;
        margin: 0 0 1 0;
        background: #0D0D12;
        border-left: thick #4B5563;
    }
    ThinkingBlock .tb-header {
        height: auto;
        layout: horizontal;
        padding: 0 1;
        background: #0D0D12;
    }
    ThinkingBlock .tb-label {
        width: 1fr;
        height: auto;
        color: #6B7280;
        text-style: italic;
        content-align: left middle;
    }
    ThinkingBlock .tb-badge {
        width: auto;
        height: auto;
        color: #4B5563;
        content-align: right middle;
    }
    ThinkingBlock .tb-toggle {
        width: auto;
        min-width: 5;
        height: 1;
        background: transparent;
        border: none;
        color: #4B5563;
        padding: 0 0;
        margin: 0 0;
    }
    ThinkingBlock .tb-toggle:hover {
        background: #1C1C26;
        color: #8B5CF6;
    }
    ThinkingBlock .tb-body {
        display: none;
        height: auto;
        padding: 0 2 1 2;
        color: #6B7280;
        text-style: italic;
    }
    ThinkingBlock .tb-body.visible {
        display: block;
    }
    """

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = (text or "").strip()[:2000]
        self._expanded = False

    def compose(self) -> ComposeResult:
        char_count = len(self._text)
        with Horizontal(classes="tb-header"):
            yield Static(
                Text("· Мысль модели", style="italic #6B7280"),
                classes="tb-label",
            )
            yield Button(
                f"▶ развернуть ({char_count} симв.)",
                classes="tb-toggle",
            )
        yield Static(
            Text(self._text, style="#6B7280"),
            classes="tb-body",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._expanded = not self._expanded
        try:
            body = self.query_one(".tb-body", Static)
            btn = self.query_one(".tb-toggle", Button)
            char_count = len(self._text)
            if self._expanded:
                body.add_class("visible")
                btn.label = f"▼ свернуть ({char_count} симв.)"
            else:
                body.remove_class("visible")
                btn.label = f"▶ развернуть ({char_count} симв.)"
        except Exception:
            pass
