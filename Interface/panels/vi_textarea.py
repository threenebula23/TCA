"""Vi-like modal text editor widget for Lorne TCA.

Wraps Textual's TextArea with a full modal editing state machine:
  NORMAL   — navigation and editing commands (hjkl, w/b/e, dd/yy/p, etc.)
  INSERT   — text entry; Esc → NORMAL
  VISUAL   — character/line/block selection with vi motions
  COMMAND  — : command line (w, q, :%s, /{pat}, etc.)
  WIDGET   — UI navigation without mouse (Space+w from NORMAL)

The widget emits a ``ViModeChanged`` message on every mode transition so
the parent app can update the status bar.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Static, TextArea
from textual.containers import Vertical


ViMode = Literal["normal", "insert", "visual", "command", "widget"]

_MODE_COLORS: dict[str, str] = {
    "normal":  "#8B5CF6",
    "insert":  "#10B981",
    "visual":  "#3B82F6",
    "command": "#F59E0B",
    "widget":  "#06B6D4",
}

_MODE_LABELS: dict[str, str] = {
    "normal":  "NORMAL",
    "insert":  "INSERT",
    "visual":  "VISUAL",
    "command": "COMMAND",
    "widget":  "WIDGET",
}

_STATUS_HINTS: dict[str, str] = {
    "normal":  "h/j/k/l · i:вставка · v:выделение · ::команда · Space+w:widget",
    "insert":  "-- ВСТАВКА -- · Esc:normal",
    "visual":  "движение · y:копировать · d:вырезать · >/<:отступ · Esc:normal",
    "command": ":команда · Enter:выполнить · Esc:отмена",
    "widget":  "Space+c:чат · Space+e:файлы · [/]:вкладки · j/k:прокрутка · Esc:normal",
}


class ViModeChanged(Message):
    """Emitted when the vi mode changes."""

    def __init__(self, mode: ViMode, hint: str, color: str) -> None:
        super().__init__()
        self.mode = mode
        self.hint = hint
        self.color = color


class ViEditorArea(TextArea):
    """A ``TextArea`` whose key handling is driven by the parent vi state machine.

    The plain ``TextArea`` is the focused widget, so it receives every key
    *before* the surrounding ``ViTextArea`` container. Its base ``_on_key``
    inserts printable characters and then stops the event, which means the
    container's ``on_key`` never runs and vi commands (``h j k l i v :`` …)
    silently turn into typed text. By overriding ``_on_key`` here we let the
    owner consume keys first in every mode except INSERT, where the normal
    editing behaviour is preserved.
    """

    def __init__(self, *args, owner: "ViTextArea | None" = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vi_owner = owner

    async def _on_key(self, event: events.Key) -> None:
        owner = self._vi_owner
        if owner is None:
            await super()._on_key(event)
            return

        if owner._vi_mode == "insert":
            owner._handle_insert_key(event, event.key, event.character or "")
            if not event._stop_propagation:
                await super()._on_key(event)
            return

        # NORMAL / VISUAL / COMMAND / WIDGET: the vi layer owns the key. Never
        # let the base TextArea insert text or run its own bindings.
        owner.on_key(event)
        event.stop()
        event.prevent_default()


class ViTextArea(Vertical):
    """A Vertical container wrapping TextArea with Vi-like modal editing.

    Usage: drop a ViTextArea anywhere you'd use a TextArea. The widget
    shows the file content with full vi keybindings and emits ViModeChanged
    so the app status line stays in sync.
    """

    DEFAULT_CSS = """
    ViTextArea {
        height: 1fr;
    }
    ViTextArea TextArea {
        height: 1fr;
    }
    ViTextArea #vi-command-input {
        display: none;
        height: 1;
        border: none;
        padding: 0 0;
        margin: 0 0;
        background: #0D0D12;
        color: #E5E7EB;
    }
    ViTextArea #vi-command-input.visible {
        display: block;
    }
    """

    BINDINGS = []

    def __init__(self, content: str = "", language: str = "python",
                 filepath: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content
        self._language = language
        self._filepath = filepath
        self._vi_mode: ViMode = "normal"
        self._pending_op: str = ""
        self._count_str: str = ""
        self._yank_buf: str = ""
        self._yank_line: bool = False
        self._search_pattern: str = ""
        self._search_dir: int = 1
        self._ft_char: str = ""
        self._ft_dir: int = 1
        self._ft_inclusive: bool = True
        self._last_change: Optional[str] = None

    def compose(self) -> ComposeResult:
        ta = ViEditorArea(
            self._content, language=self._language, id="vi-editor-area", owner=self,
        )
        ta.can_focus = True
        yield ta
        yield Input(placeholder=":", id="vi-command-input")

    def on_mount(self) -> None:
        self._set_mode("normal")
        try:
            self.query_one("#vi-editor-area", TextArea).focus()
        except Exception:
            pass

    def _set_mode(self, mode: ViMode) -> None:
        prev = self._vi_mode
        self._vi_mode = mode
        self._pending_op = ""
        self._count_str = ""
        try:
            cmd_input = self.query_one("#vi-command-input", Input)
            if mode == "command":
                cmd_input.add_class("visible")
                cmd_input.value = ":"
                cmd_input.focus()
            else:
                cmd_input.remove_class("visible")
                if prev == "command":
                    try:
                        self.query_one("#vi-editor-area", TextArea).focus()
                    except Exception:
                        pass
        except Exception:
            pass

        color = _MODE_COLORS.get(mode, "#8B5CF6")
        hint = _STATUS_HINTS.get(mode, "")
        self.post_message(ViModeChanged(mode, hint, color))

    @property
    def _textarea(self) -> Optional[TextArea]:
        try:
            return self.query_one("#vi-editor-area", TextArea)
        except Exception:
            return None

    @property
    def text(self) -> str:
        ta = self._textarea
        return ta.text if ta is not None else self._content

    @text.setter
    def text(self, value: str) -> None:
        self._content = value
        ta = self._textarea
        if ta is not None:
            ta.load_text(value)

    @property
    def show_line_numbers(self) -> bool:
        ta = self._textarea
        return ta.show_line_numbers if ta is not None else True

    @show_line_numbers.setter
    def show_line_numbers(self, value: bool) -> None:
        ta = self._textarea
        if ta is not None:
            ta.show_line_numbers = value

    @property
    def theme(self) -> str:
        ta = self._textarea
        return getattr(ta, "theme", "monokai") if ta is not None else "monokai"

    @theme.setter
    def theme(self, value: str) -> None:
        ta = self._textarea
        if ta is not None:
            try:
                ta.theme = value
            except Exception:
                pass

    @property
    def _count(self) -> int:
        try:
            return max(1, int(self._count_str)) if self._count_str else 1
        except ValueError:
            return 1

    def on_key(self, event: Key) -> None:
        mode = self._vi_mode
        key = event.key
        char = event.character or ""

        if mode == "insert":
            self._handle_insert_key(event, key, char)
        elif mode == "normal":
            self._handle_normal_key(event, key, char)
        elif mode == "visual":
            self._handle_visual_key(event, key, char)
        elif mode == "command":
            self._handle_command_key(event, key, char)
        elif mode == "widget":
            self._handle_widget_key(event, key, char)

    def _handle_insert_key(self, event: Key, key: str, char: str) -> None:
        ta = self._textarea
        if not ta:
            return
        if key in ("escape", "ctrl+["):
            event.stop()
            self._set_mode("normal")
        elif key == "ctrl+w":
            event.stop()
            self._delete_word_before(ta)
        elif key == "ctrl+u":
            event.stop()
            self._delete_to_line_start(ta)

    def _handle_normal_key(self, event: Key, key: str, char: str) -> None:
        ta = self._textarea
        if not ta:
            return

        if char.isdigit() and not (char == "0" and not self._count_str) and not self._pending_op:
            event.stop()
            self._count_str += char
            return

        count = self._count

        if self._pending_op:
            self._resolve_operator(event, key, char, ta, count)
            return

        event.stop()

        if char == "i":
            self._set_mode("insert")
        elif char == "I":
            self._move_first_nonblank(ta)
            self._set_mode("insert")
        elif char == "a":
            self._move_right(ta, 1)
            self._set_mode("insert")
        elif char == "A":
            self._move_line_end(ta)
            self._set_mode("insert")
        elif char == "o":
            self._open_line_below(ta)
            self._set_mode("insert")
        elif char == "O":
            self._open_line_above(ta)
            self._set_mode("insert")
        elif char == "v":
            try:
                from textual.widgets.text_area import Selection
                loc = ta.cursor_location
                ta.selection = Selection(loc, loc)
            except Exception:
                pass
            self._set_mode("visual")
        elif char == "V":
            self._set_mode("visual")
        elif char == ":":
            self._set_mode("command")
        elif char == " " and self._pending_op == "":
            self._pending_op = "space"
        elif key == "ctrl+leftbracket":
            self._set_mode("normal")

        elif char in ("h", "") or key == "left":
            self._move_left(ta, count)
        elif char in ("l", "") or key == "right":
            self._move_right(ta, count)
        elif char in ("j", "") or key == "down":
            self._move_down(ta, count)
        elif char in ("k", "") or key == "up":
            self._move_up(ta, count)
        elif char == "w":
            self._move_word_forward(ta, count)
        elif char == "W":
            self._move_word_forward(ta, count, big=True)
        elif char == "b":
            self._move_word_backward(ta, count)
        elif char == "B":
            self._move_word_backward(ta, count, big=True)
        elif char == "e":
            self._move_word_end(ta, count)
        elif char == "0":
            self._move_line_start(ta)
        elif char == "^":
            self._move_first_nonblank(ta)
        elif char == "$":
            self._move_line_end(ta)
        elif char == "G":
            if self._count_str:
                self._goto_line(ta, count)
            else:
                self._goto_file_end(ta)
        elif key == "ctrl+f":
            self._page_down(ta, count)
        elif key == "ctrl+b":
            self._page_up(ta, count)
        elif key == "ctrl+d":
            self._half_page_down(ta, count)
        elif key == "ctrl+u":
            self._half_page_up(ta, count)

        elif char == "x":
            self._delete_char(ta, count)
        elif char == "X":
            self._delete_char_before(ta, count)
        elif char == "r":
            self._pending_op = "r"
        elif char == "R":
            self._set_mode("insert")
        elif char == "s":
            self._delete_char(ta, 1)
            self._set_mode("insert")
        elif char == "S":
            self._delete_line_content(ta)
            self._set_mode("insert")
        elif char == "d":
            self._pending_op = "d"
        elif char == "D":
            self._delete_to_line_end(ta)
        elif char == "c":
            self._pending_op = "c"
        elif char == "C":
            self._delete_to_line_end(ta)
            self._set_mode("insert")
        elif char == "y":
            self._pending_op = "y"
        elif char == "Y":
            self._yank_line_full(ta, count)
        elif char == "p":
            self._paste_after(ta)
        elif char == "P":
            self._paste_before(ta)
        elif key == "ctrl+r":
            ta.redo()
        elif char == "u":
            ta.undo()
        elif char == ".":
            pass  # TODO: повторить последнее изменение
        elif char == "~":
            self._toggle_case(ta)
        elif char == "J":
            self._join_line(ta)
        elif char in (">",):
            self._pending_op = ">"
        elif char in ("<",):
            self._pending_op = "<"

        elif char == ";":
            if self._ft_char:
                self._move_to_char(ta, self._ft_char, self._ft_dir,
                                   self._ft_inclusive, count)
        elif char == ",":
            if self._ft_char:
                self._move_to_char(ta, self._ft_char, -self._ft_dir,
                                   self._ft_inclusive, count)
        elif char == "%":
            self._match_bracket(ta)

        elif char == "/" or char == "?":
            self._set_mode("command")
        elif char == "n":
            self._search_next(ta)
        elif char == "N":
            self._search_prev(ta)
        elif char == "*":
            self._search_word_under_cursor(ta, forward=True)
        elif char == "#":
            self._search_word_under_cursor(ta, forward=False)

        elif char in ("f", "F", "t", "T"):
            self._pending_op = char

        elif char == "g":
            self._pending_op = "g"

        elif char == "z":
            self._pending_op = "z"

        self._count_str = ""

    def _resolve_operator(self, event: Key, key: str, char: str, ta: TextArea, count: int) -> None:
        op = self._pending_op
        event.stop()
        self._pending_op = ""
        self._count_str = ""

        if op == "space":
            if char == "w":
                self._set_mode("widget")
            elif char in ("c", "e", "a", "s", "t"):
                self._set_mode("widget")
                self._dispatch_widget_action(char)
            return

        if op == "g":
            if char == "g":
                self._goto_file_start(ta)
            elif char == "u":
                self._pending_op = "gu"
            elif char == "U":
                self._pending_op = "gU"
            return

        if op in ("gu", "gU"):
            return

        if op == "z":
            if char == "z":
                ta.scroll_cursor_visible()
            return

        if op == "r":
            if char:
                self._replace_char(ta, char)
            return

        if op in ("f", "F", "t", "T"):
            if char:
                self._ft_char = char
                self._ft_dir = 1 if op in ("f", "t") else -1
                self._ft_inclusive = op in ("f", "F")
                self._move_to_char(ta, char, self._ft_dir, self._ft_inclusive, count)
            return

        if op in (">", "<"):
            if char == op:
                self._indent_line(ta, count, op == ">")
            return

        if op in ("d", "c", "y"):
            motion = char
            if motion == op:
                self._op_whole_line(ta, op, count)
            elif motion == "w":
                self._op_motion_word(ta, op, count, big=False)
            elif motion == "W":
                self._op_motion_word(ta, op, count, big=True)
            elif motion in ("j", "down"):
                self._op_lines(ta, op, count + 1)
            elif motion in ("k", "up"):
                self._op_lines_before(ta, op, count + 1)
            elif motion == "$":
                self._op_to_line_end(ta, op)
            elif motion == "0":
                self._op_to_line_start(ta, op)
            if op == "c":
                self._set_mode("insert")

    def _handle_visual_key(self, event: Key, key: str, char: str) -> None:
        ta = self._textarea
        if not ta:
            return
        event.stop()

        if key == "escape":
            self._set_mode("normal")
            return

        count = self._count

        if char in ("h", "") or key == "left":
            self._move_left(ta, count)
        elif char in ("l", "") or key == "right":
            self._move_right(ta, count)
        elif char in ("j", "") or key == "down":
            self._move_down(ta, count)
        elif char in ("k", "") or key == "up":
            self._move_up(ta, count)
        elif char == "w":
            self._move_word_forward(ta, count)
        elif char == "b":
            self._move_word_backward(ta, count)
        elif char == "$":
            self._move_line_end(ta)
        elif char == "0":
            self._move_line_start(ta)
        elif char == "y":
            self._yank_selection(ta)
            self._set_mode("normal")
        elif char in ("d", "x"):
            self._delete_selection(ta)
            self._set_mode("normal")
        elif char == "c":
            self._delete_selection(ta)
            self._set_mode("insert")
        elif char == ">":
            self._indent_selection(ta, True)
        elif char == "<":
            self._indent_selection(ta, False)
        elif char == "~":
            self._toggle_case_selection(ta)
        elif char == "u":
            self._lower_selection(ta)
        elif char == "U":
            self._upper_selection(ta)
        elif char == "J":
            self._join_selection(ta)

        self._count_str = ""

    def _handle_command_key(self, event: Key, key: str, char: str) -> None:
        if key == "escape":
            event.stop()
            self._set_mode("normal")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Execute command when user presses Enter in command input."""
        if event.input.id != "vi-command-input":
            return
        cmd = str(event.value).lstrip(":")
        event.input.value = ""
        self._execute_command(cmd)
        self._set_mode("normal")

    def _execute_command(self, cmd: str) -> None:
        ta = self._textarea
        if not ta:
            return
        cmd = cmd.strip()
        if not cmd:
            return

        if cmd in ("w", "write"):
            self._save_file()
        elif cmd in ("q", "quit"):
            self._close_tab()
        elif cmd in ("q!", "quit!"):
            self._close_tab(force=True)
        elif cmd in ("wq", "x", "ZZ"):
            self._save_file()
            self._close_tab()
        elif re.match(r"^\d+$", cmd):
            line = int(cmd)
            self._goto_line(ta, line)
        elif cmd.startswith("e "):
            path = cmd[2:].strip()
            self._open_file(path)
        elif cmd in ("bn", "bnext"):
            self._next_tab()
        elif cmd in ("bp", "bprev"):
            self._prev_tab()
        elif cmd.startswith("set "):
            self._handle_set(cmd[4:])
        elif m := re.match(r"^%s/(.+?)/(.*?)/(g|gc)?$", cmd):
            find, replace, flags = m.group(1), m.group(2), m.group(3) or ""
            self._global_replace(ta, find, replace, "g" in flags)
        elif cmd.startswith("/"):
            self._search_pattern = cmd[1:]
            self._search_dir = 1
            self._do_search(ta, self._search_pattern, 1)
        elif cmd.startswith("?"):
            self._search_pattern = cmd[1:]
            self._search_dir = -1
            self._do_search(ta, self._search_pattern, -1)
        elif cmd.startswith("!"):
            self._run_shell(cmd[1:])

    def _handle_widget_key(self, event: Key, key: str, char: str) -> None:
        event.stop()
        if key == "escape":
            self._set_mode("normal")
            return
        self._dispatch_widget_action(char, key)

    def _dispatch_widget_action(self, char: str, key: str = "") -> None:
        """Dispatch widget-mode actions to the app."""
        try:
            app = self.app
        except Exception:
            return

        if char == "c":
            try:
                app.action_focus_chat()
            except Exception:
                pass
        elif char == "e":
            try:
                app.action_focus_file_explorer()
            except Exception:
                pass
        elif char == "a":
            try:
                app.action_focus_active_agents()
            except Exception:
                pass
        elif char == "s":
            try:
                app.action_open_settings()
            except Exception:
                pass
        elif char == "t":
            try:
                app.action_new_terminal()
            except Exception:
                pass
        elif char in ("1", "2", "3", "4", "5", "6"):
            try:
                app.action_focus_tab(int(char))
            except Exception:
                pass
        elif char in ("[", "h") or key == "shift+tab":
            try:
                app.action_prev_tab()
            except Exception:
                pass
        elif char in ("]", "l") or key == "tab":
            try:
                app.action_next_tab()
            except Exception:
                pass
        elif char == "?" or key == "f1":
            try:
                app.action_open_keybindings()
            except Exception:
                pass
        elif char == "+":
            try:
                app.action_sidebar_grow()
            except Exception:
                pass
        elif char == "-":
            try:
                app.action_sidebar_shrink()
            except Exception:
                pass
        elif char == "=":
            try:
                app.action_sidebar_reset()
            except Exception:
                pass
        elif char == "m":
            try:
                app.action_cycle_mode()
            except Exception:
                pass
        elif char == "M":
            try:
                app.action_cycle_model()
            except Exception:
                pass

    def _move_left(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_cursor_left()

    def _move_right(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_cursor_right()

    def _move_up(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_cursor_up()

    def _move_down(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_cursor_down()

    def _move_word_forward(self, ta: TextArea, n: int = 1, big: bool = False) -> None:
        for _ in range(n):
            ta.action_cursor_word_right()

    def _move_word_backward(self, ta: TextArea, n: int = 1, big: bool = False) -> None:
        for _ in range(n):
            ta.action_cursor_word_left()

    def _move_word_end(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_cursor_word_right()

    def _move_line_start(self, ta: TextArea) -> None:
        ta.action_cursor_line_start()

    def _move_line_end(self, ta: TextArea) -> None:
        ta.action_cursor_line_end()

    def _move_first_nonblank(self, ta: TextArea) -> None:
        ta.action_cursor_line_start()
        text = ta.text
        row = ta.cursor_location[0]
        lines = text.splitlines()
        if row < len(lines):
            line = lines[row]
            col = len(line) - len(line.lstrip())
            ta.move_cursor((row, col))

    def _goto_line(self, ta: TextArea, line: int) -> None:
        lines = ta.text.splitlines()
        row = max(0, min(line - 1, len(lines) - 1))
        ta.move_cursor((row, 0))

    def _goto_file_start(self, ta: TextArea) -> None:
        ta.move_cursor((0, 0))

    def _goto_file_end(self, ta: TextArea) -> None:
        lines = ta.text.splitlines()
        row = max(0, len(lines) - 1)
        ta.move_cursor((row, 0))

    def _page_down(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_scroll_page_down()

    def _page_up(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_scroll_page_up()

    def _half_page_down(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n * 10):
            ta.action_cursor_down()

    def _half_page_up(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n * 10):
            ta.action_cursor_up()

    def _move_to_char(self, ta: TextArea, char: str, direction: int,
                      inclusive: bool, count: int = 1) -> None:
        row, col = ta.cursor_location
        lines = ta.text.splitlines()
        if row >= len(lines):
            return
        line = lines[row]
        found = 0
        if direction == 1:
            for i in range(col + 1, len(line)):
                if line[i] == char:
                    found += 1
                    if found >= count:
                        ta.move_cursor((row, i if inclusive else i - 1))
                        return
        else:
            for i in range(col - 1, -1, -1):
                if line[i] == char:
                    found += 1
                    if found >= count:
                        ta.move_cursor((row, i if inclusive else i + 1))
                        return

    def _match_bracket(self, ta: TextArea) -> None:
        """Jump to the bracket matching the one under the cursor (vi ``%``)."""
        pairs = {"(": ")", "[": "]", "{": "}"}
        rpairs = {v: k for k, v in pairs.items()}
        row, col = ta.cursor_location
        lines = ta.text.splitlines()
        if row >= len(lines) or col >= len(lines[row]):
            return
        ch = lines[row][col]
        if ch in pairs:
            target, depth, step, fwd = pairs[ch], 0, 1, True
        elif ch in rpairs:
            target, depth, step, fwd = rpairs[ch], 0, -1, False
        else:
            return
        r, c = row, col
        while 0 <= r < len(lines):
            line = lines[r]
            while 0 <= c < len(line):
                cur = line[c]
                if cur == ch:
                    depth += 1
                elif cur == target:
                    depth -= 1
                    if depth == 0:
                        ta.move_cursor((r, c))
                        return
                c += step
            r += step
            if 0 <= r < len(lines):
                c = 0 if fwd else len(lines[r]) - 1

    def _delete_char(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_delete_right()

    def _delete_char_before(self, ta: TextArea, n: int = 1) -> None:
        for _ in range(n):
            ta.action_delete_left()

    def _delete_word_before(self, ta: TextArea) -> None:
        ta.action_delete_word_left()

    def _delete_to_line_start(self, ta: TextArea) -> None:
        ta.action_delete_to_start_of_line()

    def _delete_to_line_end(self, ta: TextArea) -> None:
        ta.action_delete_to_end_of_line()

    def _delete_line_content(self, ta: TextArea) -> None:
        ta.action_cursor_line_start()
        ta.action_delete_to_end_of_line()

    def _replace_char(self, ta: TextArea, char: str) -> None:
        ta.action_delete_right()
        ta.insert(char)
        ta.action_cursor_left()

    def _toggle_case(self, ta: TextArea) -> None:
        row, col = ta.cursor_location
        lines = ta.text.splitlines()
        if row < len(lines) and col < len(lines[row]):
            c = lines[row][col]
            replacement = c.lower() if c.isupper() else c.upper()
            ta.action_delete_right()
            ta.insert(replacement)

    def _join_line(self, ta: TextArea) -> None:
        row, col = ta.cursor_location
        lines = ta.text.splitlines()
        if row < len(lines) - 1:
            ta.move_cursor((row, len(lines[row])))
            ta.action_delete_right()
            ta.insert(" ")

    def _open_line_below(self, ta: TextArea) -> None:
        ta.action_cursor_line_end()
        ta.action_cursor_right()
        ta.insert("\n")

    def _open_line_above(self, ta: TextArea) -> None:
        row, _ = ta.cursor_location
        if row == 0:
            ta.move_cursor((0, 0))
            ta.insert("\n")
            ta.move_cursor((0, 0))
        else:
            ta.move_cursor((row - 1, 0))
            ta.action_cursor_line_end()
            ta.insert("\n")

    def _indent_line(self, ta: TextArea, count: int = 1, right: bool = True) -> None:
        row, _ = ta.cursor_location
        lines = ta.text.splitlines()
        for _ in range(count):
            if row < len(lines):
                if right:
                    ta.move_cursor((row, 0))
                    ta.insert("    ")
                else:
                    line = lines[row]
                    strip = min(4, len(line) - len(line.lstrip()))
                    if strip > 0:
                        ta.move_cursor((row, 0))
                        for _ in range(strip):
                            ta.action_delete_right()

    def _op_whole_line(self, ta: TextArea, op: str, count: int) -> None:
        for _ in range(count):
            if op == "d":
                ta.action_delete_line()
            elif op == "y":
                row, _ = ta.cursor_location
                lines = ta.text.splitlines()
                if row < len(lines):
                    self._yank_buf = "\n".join(lines[row:row + count])
                    self._yank_line = True
                break

    def _op_motion_word(self, ta: TextArea, op: str, count: int, big: bool) -> None:
        for _ in range(count):
            if op == "d":
                ta.action_delete_word_right()
            elif op == "c":
                ta.action_delete_word_right()

    def _op_lines(self, ta: TextArea, op: str, count: int) -> None:
        for _ in range(count):
            if op == "d":
                ta.action_delete_line()

    def _op_lines_before(self, ta: TextArea, op: str, count: int) -> None:
        for _ in range(count):
            if op == "d":
                ta.action_delete_line()

    def _op_to_line_end(self, ta: TextArea, op: str) -> None:
        if op == "d":
            ta.action_delete_to_end_of_line()

    def _op_to_line_start(self, ta: TextArea, op: str) -> None:
        if op == "d":
            ta.action_delete_to_start_of_line()

    def _yank_line_full(self, ta: TextArea, count: int = 1) -> None:
        row, _ = ta.cursor_location
        lines = ta.text.splitlines()
        self._yank_buf = "\n".join(lines[row:row + count])
        self._yank_line = True

    def _paste_after(self, ta: TextArea) -> None:
        if not self._yank_buf:
            return
        if self._yank_line:
            row, _ = ta.cursor_location
            lines = ta.text.splitlines()
            ta.move_cursor((row, len(lines[row]) if row < len(lines) else 0))
            ta.insert("\n" + self._yank_buf)
        else:
            ta.action_cursor_right()
            ta.insert(self._yank_buf)

    def _paste_before(self, ta: TextArea) -> None:
        if not self._yank_buf:
            return
        if self._yank_line:
            row, _ = ta.cursor_location
            ta.move_cursor((row, 0))
            ta.insert(self._yank_buf + "\n")
        else:
            ta.insert(self._yank_buf)

    def _yank_selection(self, ta: TextArea) -> None:
        try:
            self._yank_buf = ta.selected_text
            self._yank_line = False
        except Exception:
            pass

    def _delete_selection(self, ta: TextArea) -> None:
        try:
            self._yank_buf = ta.selected_text
            ta.action_delete_left()
        except Exception:
            pass

    def _indent_selection(self, ta: TextArea, right: bool) -> None:
        pass  # Сложная операция; делегируем будущей реализации

    def _toggle_case_selection(self, ta: TextArea) -> None:
        pass

    def _lower_selection(self, ta: TextArea) -> None:
        try:
            sel = ta.selected_text
            ta.action_delete_left()
            ta.insert(sel.lower())
        except Exception:
            pass

    def _upper_selection(self, ta: TextArea) -> None:
        try:
            sel = ta.selected_text
            ta.action_delete_left()
            ta.insert(sel.upper())
        except Exception:
            pass

    def _join_selection(self, ta: TextArea) -> None:
        pass

    def _search_next(self, ta: TextArea) -> None:
        if self._search_pattern:
            self._do_search(ta, self._search_pattern, self._search_dir)

    def _search_prev(self, ta: TextArea) -> None:
        if self._search_pattern:
            self._do_search(ta, self._search_pattern, -self._search_dir)

    def _do_search(self, ta: TextArea, pattern: str, direction: int) -> None:
        try:
            text = ta.text
            row, col = ta.cursor_location
            lines = text.splitlines()
            total = len(lines)
            start = row * 10000 + col
            flat = text
            if direction == 1:
                m = re.search(pattern, flat[start + 1:], re.IGNORECASE)
                if m:
                    abs_pos = start + 1 + m.start()
                    chars = 0
                    for r, line in enumerate(lines):
                        if chars + len(line) + 1 > abs_pos:
                            ta.move_cursor((r, abs_pos - chars))
                            return
                        chars += len(line) + 1
            else:
                m = re.search(pattern, flat[:start], re.IGNORECASE)
                if m:
                    abs_pos = m.start()
                    chars = 0
                    for r, line in enumerate(lines):
                        if chars + len(line) + 1 > abs_pos:
                            ta.move_cursor((r, abs_pos - chars))
                            return
                        chars += len(line) + 1
        except Exception:
            pass

    def _search_word_under_cursor(self, ta: TextArea, forward: bool) -> None:
        try:
            row, col = ta.cursor_location
            lines = ta.text.splitlines()
            if row < len(lines):
                line = lines[row]
                start = col
                while start > 0 and line[start - 1].isalnum() or (start > 0 and line[start - 1] == "_"):
                    start -= 1
                end = col
                while end < len(line) and (line[end].isalnum() or line[end] == "_"):
                    end += 1
                word = line[start:end]
                if word:
                    self._search_pattern = re.escape(word)
                    self._search_dir = 1 if forward else -1
                    self._do_search(ta, self._search_pattern, self._search_dir)
        except Exception:
            pass

    def _global_replace(self, ta: TextArea, find: str, replace: str, global_: bool) -> None:
        try:
            flags = re.IGNORECASE
            new_text = re.sub(find, replace, ta.text, flags=flags) if global_ else \
                       re.sub(find, replace, ta.text, count=1, flags=flags)
            ta.load_text(new_text)
        except Exception:
            pass

    def _save_file(self) -> None:
        try:
            if self._filepath:
                from pathlib import Path
                Path(self._filepath).write_text(
                    self._textarea.text if self._textarea else "",
                    encoding="utf-8",
                )
        except Exception:
            pass

    def _close_tab(self, force: bool = False) -> None:
        try:
            self.app.action_close_tab()
        except Exception:
            pass

    def _open_file(self, path: str) -> None:
        try:
            self.app.action_open_file(path)
        except Exception:
            pass

    def _next_tab(self) -> None:
        try:
            self.app.action_next_tab()
        except Exception:
            pass

    def _prev_tab(self) -> None:
        try:
            self.app.action_prev_tab()
        except Exception:
            pass

    def _handle_set(self, opts: str) -> None:
        pass

    def _run_shell(self, cmd: str) -> None:
        try:
            import subprocess
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            output = (result.stdout + result.stderr)[:500]
            try:
                from Interface.tui_bridge import get_bridge
                bridge = get_bridge()
                if bridge:
                    bridge.on_info(f"$ {cmd}\n{output}")
            except Exception:
                pass
        except Exception:
            pass
