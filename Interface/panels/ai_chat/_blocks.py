"""Assistant / user bubble widgets."""
from __future__ import annotations

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.markdown import Markdown

from rich.text import Text

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Checkbox, Collapsible, DirectoryTree, Input, Label, RichLog, Select,
    Static, TextArea,
)

try:
    from textual.widgets import Markdown as MarkdownWidget
except ImportError:  # pragma: no cover
    MarkdownWidget = None  # type: ignore[misc, assignment]

try:
    from Interface.panels.creator_progress import CreatorProgressBlock
except Exception:  # pragma: no cover
    CreatorProgressBlock = None  # type: ignore[misc, assignment]

try:
    from Interface.panels.diff_block import (
        CodeDiffBlock,
        diff_stats as _diff_stats,
        read_before_after_texts as _read_before_after_texts,
    )
except Exception:  # pragma: no cover
    CodeDiffBlock = None  # type: ignore[misc, assignment]
    def _diff_stats(before: str, after: str) -> tuple[int, int]:  # type: ignore[misc]
        return 0, 0
    def _read_before_after_texts(path: str, snapshot_id):  # type: ignore[misc]
        return "", ""

try:
    from Interface.panels.deep_checkpoint import DeepCheckpointBlock
except Exception:  # pragma: no cover
    DeepCheckpointBlock = None  # type: ignore[misc, assignment]

try:
    from Interface.panels.tool_card import ToolCardBlock, PRETTY_TOOL_NAMES
except Exception:  # pragma: no cover
    ToolCardBlock = None  # type: ignore[misc, assignment]
    PRETTY_TOOL_NAMES = frozenset()  # type: ignore[assignment]

try:
    from Interface.panels.download_block import DownloadProgressBlock
except Exception:  # pragma: no cover
    DownloadProgressBlock = None  # type: ignore[misc, assignment]

from rich.markdown import Markdown
from rich.text import Text

from ._constants import PURPLE, PURPLE_LIGHT, GRAY
from ._helpers import _syntax_theme
from ._messages import RollbackRequested

try:
    from Interface.panels.rich_message import RichAssistantBody
except Exception:  # pragma: no cover
    RichAssistantBody = None  # type: ignore[misc, assignment]

class AssistantMessageBlock(Vertical):
    """Ответ ассистента: Markdown + кнопка копирования + футер."""

    DEFAULT_CSS = """
    AssistantMessageBlock {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1 1 1;
        background: #12121a;
        border: round #2D2D3D;
        border-title-color: #A78BFA;
        border-title-style: bold;
    }
    AssistantMessageBlock .assistant-header {
        height: auto;
        margin: 0 0 1 0;
        color: #A78BFA;
        text-style: bold;
    }
    AssistantMessageBlock .assistant-md {
        height: auto;
        margin: 0 0 1 0;
    }
    AssistantMessageBlock .assistant-footer {
        height: auto;
        color: #6B7280;
        margin-top: 1;
        border-top: solid #1F1F2C;
        padding-top: 1;
    }
    AssistantMessageBlock .copy-row {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    AssistantMessageBlock .copy-row Button {
        min-width: 18;
        height: 3;
        content-align: center middle;
        border: round #2D2D3D;
    }
    """

    def __init__(self, plain_copy: str, footer: str, copy_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._plain = plain_copy
        self._footer = footer or ""
        self._copy_id = copy_id

    def _accent(self) -> str:
        try:
            from Interface.ui_prefs import load_prefs
            from Interface.themes import get_theme
            prefs = load_prefs()
            theme = get_theme(str(prefs.get("theme", "Purple Dark")))
            return str(prefs.get("accent_color") or theme.get("accent") or PURPLE)
        except Exception:
            return PURPLE

    def compose(self) -> ComposeResult:
        body = (self._plain or "").strip()[:120_000]
        accent = self._accent()
        yield Static(
            Text.assemble(("● ", ""), ("Lorne", f"bold {accent}"), ("  ·  ", "dim"), (time.strftime("%H:%M"), "dim")),
            classes="assistant-header",
        )
        if RichAssistantBody is not None:
            yield RichAssistantBody(body)
        elif MarkdownWidget is not None:
            yield MarkdownWidget(body, classes="assistant-md")
        else:
            theme = _syntax_theme()
            yield Static(Markdown(body, code_theme=theme), classes="assistant-md")
        with Horizontal(classes="copy-row"):
            yield Button("▤ Копировать ответ", id=f"copy-assistant-{self._copy_id}", variant="default")
        if self._footer.strip():
            yield Static(self._footer, classes="assistant-footer")

    @on(Button.Pressed)
    def _copy_local(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid != f"copy-assistant-{self._copy_id}":
            return
        text = self._plain or ""
        try:
            fn = getattr(self.app, "copy_to_clipboard", None)
            if callable(fn):
                fn(text)
                self.notify("Скопировано в буфер")
                return
        except Exception:
            pass
        self.notify("Буфер недоступен в этом терминале", severity="warning")


class UserMessageBlock(Vertical):
    DEFAULT_CSS = """
    UserMessageBlock {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1 1 1;
        background: #1a1528;
        border: round #3D2D5C;
        border-left: outer #8B5CF6;
    }
    UserMessageBlock .user-header {
        height: auto;
        margin: 0 0 1 0;
        text-style: bold;
    }
    UserMessageBlock .user-rollback-row {
        height: auto;
        margin-top: 1;
    }
    UserMessageBlock .user-rollback-row Button {
        min-width: 28;
        background: #1F2430;
        color: #9CA3AF;
        border: round #2D2D3D;
    }
    UserMessageBlock .user-rollback-row Button:hover {
        background: #272D3A;
        color: #D1D5DB;
    }
    """

    def __init__(self, text: str, turn_index: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text
        self._turn_index = int(turn_index)
        self._name_static: Optional[Static] = None
        self._timestamp = time.strftime("%H:%M")

    def _accent(self) -> str:
        try:
            from Interface.ui_prefs import load_prefs
            from Interface.themes import get_theme
            prefs = load_prefs()
            theme = get_theme(str(prefs.get("theme", "Purple Dark")))
            return str(prefs.get("accent_color") or theme.get("accent") or PURPLE)
        except Exception:
            return PURPLE

    def compose(self) -> ComposeResult:
        accent = self._accent()
        self._name_static = Static(
            Text.assemble(("○ ", ""), ("Вы", f"bold {accent}"), ("  ·  ", "dim"), (self._timestamp, "dim")),
            classes="user-header",
        )
        yield self._name_static
        yield Static(Text(self._text, style="#E5E7EB"))
        if self._turn_index >= 0:
            with Horizontal(classes="user-rollback-row"):
                yield Button(
                    "↺ Откат к состоянию до этого запроса",
                    id=f"rollback-btn-{self._turn_index}",
                    variant="default",
                )

    def on_mount(self) -> None:
        self.refresh_accent()

    def refresh_accent(self) -> None:
        accent = self._accent()
        try:
            self.styles.border_left = ("outer", accent)
        except Exception:
            pass
        if self._name_static is not None:
            try:
                self._name_static.update(
                    Text.assemble(("○ ", ""), ("Вы", f"bold {accent}"), ("  ·  ", "dim"), (self._timestamp, "dim")),
                )
            except Exception:
                pass

    @on(Button.Pressed)
    def _on_rollback_press(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("rollback-btn-"):
            try:
                idx = int(bid.replace("rollback-btn-", "", 1))
            except ValueError:
                return
            try:
                self.app.post_message(RollbackRequested(idx))
            except Exception:
                self.post_message(RollbackRequested(idx))

