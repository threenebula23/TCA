"""IDE Lorne (Textual): чат по центру, файлы и агенты слева, вкладки рабочей области."""
from __future__ import annotations

import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from textual import on
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Static, Button, TextArea, Select, Checkbox

from Interface.branding import APP_DISPLAY_NAME, APP_FULL_VERSION_LABEL
from Interface.ui_prefs import load_prefs
from Interface.themes import apply_theme

from .panels.file_explorer import (
    FileExplorerPanel, FileSelected, AddToContext, RunFileRequested, OpenChatSettings,
)
from .panels.code_editor import FileEditorTabPane, FileSaved, CloseWorkspaceTab
from .panels.workspace_center import WorkspaceCenter, CHAT_TAB_ID
from .panels.active_agents_panel import (
    ActiveAgentsPanel, AgentWorkerSelected, AgentMainChatSelected,
)
from .panels.ai_chat import (
    AIChatPanel, ChatSubmitted, ModelChanged, ModeToggled, StopRequested,
    RollbackRequested, DeepCheckpointAction,
)
from .session_picker_screen import SessionPickerScreen
from .panels.vi_textarea import ViModeChanged


class LorneApp(App):
    """Полноэкранное приложение IDE (Textual)."""

    CSS_PATH = "tui_app.tcss"
    TITLE = f"{APP_DISPLAY_NAME} — Terminal coding assistant · {APP_FULL_VERSION_LABEL}"
    LAYERS = ["base", "overlay"]

    BINDINGS = [
        Binding("ctrl+q", "quit", "Exit", show=True, priority=True),
        Binding("ctrl+s", "save_file", "Save", show=False, priority=True),
        Binding("f1", "open_keybindings", "Keybindings", show=False, priority=True),
        # priority=True is required so these fire even while the chat input
        # (a TextArea) has focus and would otherwise swallow the keystroke.
        Binding("f2", "cycle_mode", "Mode", show=True, priority=True),
        Binding("f3", "cycle_model", "Model", show=True, priority=True),
        Binding("f5", "run_current_file", "Run File", show=False, priority=True),
        Binding("ctrl+shift+x", "stop_agent", "Stop Agent", show=False, priority=True),
        Binding("ctrl+f", "toggle_find", "Find", show=False),
        Binding("ctrl+g", "goto_line", "Go to Line", show=False),
        Binding("ctrl+w", "close_tab", "Close Tab", show=False),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", show=False),
        Binding("escape", "focus_chat", "Chat", show=False),
        Binding("f6", "resize_left_smaller", "Left -", show=False),
        Binding("f7", "resize_left_larger", "Left +", show=False),
    ]

    def __init__(
        self,
        model_name: str = "",
        branch: str = "",
        models: Optional[List[Dict]] = None,
        on_chat_submit: Optional[Callable[..., None]] = None,
        on_model_change: Optional[Callable[[str], None]] = None,
        on_mode_toggle: Optional[Callable[[str], None]] = None,
        on_session_resolved: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_chat_rollback: Optional[Callable[[int], None]] = None,
        on_app_close: Optional[Callable[[], None]] = None,
        on_deep_checkpoint: Optional[Callable[[str, str], None]] = None,
        require_session_picker: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._branch = branch
        self._models = models or []
        self._on_chat_submit = on_chat_submit
        self._on_model_change = on_model_change
        self._on_mode_toggle = on_mode_toggle
        self._on_session_resolved = on_session_resolved
        self._on_chat_rollback = on_chat_rollback
        self._on_app_close = on_app_close
        self._on_deep_checkpoint = on_deep_checkpoint
        self._require_session_picker = require_session_picker
        self._bridge = None
        self._left_width = 28
        self._mode_name = "agent"
        self._shutdown_done = False

    def on_mount(self) -> None:
        prefs = load_prefs()
        try:
            apply_theme(self, str(prefs.get("theme", "Purple Dark")))
        except Exception:
            pass
        dens = str(prefs.get("density", "normal"))
        if dens not in ("compact", "normal", "spacious"):
            dens = "normal"
        for d in ("compact", "normal", "spacious"):
            self.remove_class(f"density-{d}")
        self.add_class(f"density-{dens}")
        if self._require_session_picker and self._on_session_resolved:
            self.set_timer(0.05, self._push_session_picker)

    def _push_session_picker(self) -> None:
        try:
            from Agent.checkpoint import list_sessions

            rows = list_sessions(limit=80)
        except Exception:
            rows = []
        self.push_screen(SessionPickerScreen(rows), self._on_session_picker_result)

    def _on_session_picker_result(self, result: Optional[Dict[str, Any]]) -> None:
        if result is None:
            self._run_shutdown_hooks()
            self.exit()
            return
        act = str(result.get("action", ""))
        if act == "delete":
            try:
                from Agent.checkpoint import delete_session, list_sessions

                sid = str(result.get("session_id", ""))
                if sid:
                    delete_session(sid)
            except Exception:
                pass
            self._push_session_picker()
            return
        if self._on_session_resolved:
            self._on_session_resolved(result)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-bar"):
            yield Button("✕ Exit", id="app-exit-btn")
            yield Button("» Chat", id="app-chat-btn", variant="primary")
            yield Static(f"{APP_DISPLAY_NAME} · {APP_FULL_VERSION_LABEL}", id="top-bar-badge")
        with Horizontal(id="main"):
            with Vertical(id="col-left"):
                yield FileExplorerPanel(id="file-explorer")
                yield ActiveAgentsPanel(id="active-agents")
            yield WorkspaceCenter(
                models=self._models,
                current_model=self._model_name,
                id="workspace-center",
            )
        status = f" {self._model_name}"
        if self._branch:
            status += f"  ⎇ {self._branch}"
        status += "  │  Esc: чат  M: меню"
        with Horizontal(id="bottom-bar"):
            yield Static(status, id="status-bar")
            yield Static("[NORMAL] h/j/k/l · i:вставка · v:выделение · ::команда · Space+w:widget  F1:справка",
                         id="vi-mode-indicator")

    @property
    def file_explorer(self) -> FileExplorerPanel:
        return self.query_one("#file-explorer", FileExplorerPanel)

    @property
    def workspace(self) -> WorkspaceCenter:
        return self.query_one("#workspace-center", WorkspaceCenter)

    @property
    def chat(self) -> AIChatPanel:
        return self.workspace.chat

    @property
    def active_agents(self) -> ActiveAgentsPanel:
        return self.query_one("#active-agents", ActiveAgentsPanel)

    @property
    def status_bar(self) -> Static:
        return self.query_one("#status-bar", Static)

    @on(FileSelected)
    def on_file_open(self, event: FileSelected) -> None:
        self.workspace.open_path(event.path)

    @on(AddToContext)
    def on_add_to_context(self, event: AddToContext) -> None:
        try:
            self.chat.register_context_hint(event.path)
            self.notify(f"Контекст: {event.path.name}")
        except Exception as e:
            self.chat.add_error(f"Контекст: {e}")

    @on(RunFileRequested)
    def on_run_file(self, event: RunFileRequested) -> None:
        p = event.path
        if p.suffix == ".py":
            cmd = [sys.executable, str(p)]
        elif p.suffix == ".sh":
            cmd = ["bash", str(p)]
        elif p.suffix in (".js", ".ts"):
            cmd = ["node", str(p)]
        else:
            cmd = [sys.executable, str(p)]

        def _run() -> None:
            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=180, cwd=str(Path.cwd()),
                )
                lines = []
                if r.stdout:
                    lines.append(r.stdout[:4000])
                if r.stderr:
                    lines.append("stderr:\n" + r.stderr[:2000])
                if r.returncode != 0:
                    lines.append(f"(код выхода {r.returncode})")
                msg = "\n".join(lines) if lines else "(нет вывода)"
                if self._bridge:
                    self._bridge.on_info(f"▶ {' '.join(shlex.quote(x) for x in cmd)}\n{msg}")
            except Exception as e:
                if self._bridge:
                    self._bridge.on_error(f"Run: {e}")

        self.chat.add_info(f"Запуск: {' '.join(shlex.quote(x) for x in cmd)}")
        threading.Thread(target=_run, daemon=True).start()

    @on(OpenChatSettings)
    def on_open_chat_settings(self, event: OpenChatSettings) -> None:
        try:
            sec = (event.section or "").strip().lower()
            if sec in {"close", "closed", "off", "none"}:
                self.workspace.close_active_settings_tab()
                return
            self.workspace.focus_chat_tab()
            self.workspace.open_settings_tab(sec)
        except Exception as e:
            self.chat.add_error(f"Settings open: {e}")

    # Settings widgets live in workspace tabs (not under #ai-chat); bind on App.
    @on(Button.Pressed, "#sp-apply-accent")
    def _app_on_apply_accent(self) -> None:
        self.chat.on_apply_accent()

    @on(Button.Pressed, "#sp-open-palette")
    def _app_on_open_palette(self) -> None:
        self.chat.on_open_palette()

    @on(Select.Changed, "#sp-theme")
    def _app_on_sp_theme(self, event: Select.Changed) -> None:
        self.chat.on_sp_theme(event)

    @on(Select.Changed, "#sp-density")
    def _app_on_sp_density(self, event: Select.Changed) -> None:
        self.chat.on_sp_density(event)

    @on(Select.Changed, "#sp-syntax")
    def _app_on_sp_syntax(self, event: Select.Changed) -> None:
        self.chat.on_sp_syntax(event)

    @on(Select.Changed, "#sa-profile")
    def _app_on_sa_profile(self, event: Select.Changed) -> None:
        self.chat.on_sa_profile(event)

    @on(Checkbox.Changed, "#sa-browser")
    def _app_on_sa_browser(self, event: Checkbox.Changed) -> None:
        self.chat.on_sa_browser(event)

    @on(Checkbox.Changed, "#sa-playwright")
    def _app_on_sa_playwright(self, event: Checkbox.Changed) -> None:
        self.chat.on_sa_playwright(event)

    @on(Checkbox.Changed, "#sa-custom-tools")
    def _app_on_sa_custom_tools(self, event: Checkbox.Changed) -> None:
        self.chat.on_sa_custom_tools(event)

    @on(Checkbox.Changed, "#sa-research-deep-fetch")
    def _app_on_sa_research_deep_fetch(self, event: Checkbox.Changed) -> None:
        self.chat.on_sa_research_deep_fetch(event)

    @on(Select.Changed, "#sa-orch-mode")
    def _app_on_sa_orch_mode(self, event: Select.Changed) -> None:
        self.chat.on_sa_orch_mode(event)

    @on(Button.Pressed, "#sa-apply")
    def _app_on_sa_apply(self) -> None:
        self.chat.on_sa_apply()

    @on(Button.Pressed, "#sor-save-key")
    def _app_on_sor_save_key(self) -> None:
        self.chat.on_sor_save_key()

    @on(Button.Pressed, "#sor-check-balance")
    def _app_on_sor_check_balance(self) -> None:
        self.chat.on_sor_check_balance()

    @on(Button.Pressed, "#sor-add-model")
    def _app_on_sor_add_model(self) -> None:
        self.chat.on_sor_add_model()

    @on(Button.Pressed, "#sol-save-conn")
    def _app_on_sol_save_conn(self) -> None:
        self.chat.on_sol_save_conn()

    @on(Select.Changed, "#sol-preset-select")
    def _app_on_sol_preset_changed(self, event: Select.Changed) -> None:
        self.chat.on_sol_preset_changed(event)

    @on(Button.Pressed, "#sol-save-preset")
    def _app_on_sol_save_preset(self) -> None:
        self.chat.on_sol_save_preset()

    @on(Button.Pressed, "#sol-apply-model-settings")
    def _app_on_sol_apply_model_settings(self) -> None:
        self.chat.on_sol_apply_model_settings()

    @on(Button.Pressed, "#sol-refresh")
    def _app_on_sol_refresh(self) -> None:
        self.chat.on_sol_refresh()

    @on(Select.Changed, "#sol-model-select")
    def _app_on_sol_model_select(self, event: Select.Changed) -> None:
        self.chat.on_sol_model_select(event)

    @on(Button.Pressed, "#sol-add")
    def _app_on_sol_add(self) -> None:
        self.chat.on_sol_add()

    @on(Button.Pressed, "#slm-save-conn")
    def _app_on_slm_save_conn(self) -> None:
        self.chat.on_slm_save_conn()

    @on(Button.Pressed, "#slm-refresh")
    def _app_on_slm_refresh(self) -> None:
        self.chat.on_slm_refresh()

    @on(Button.Pressed, "#slm-apply-model")
    def _app_on_slm_apply_model(self) -> None:
        self.chat.on_slm_apply_model()

    @on(FileSaved)
    def on_file_saved(self, event: FileSaved) -> None:
        self.file_explorer.refresh_tree()

    @on(RollbackRequested)
    def on_rollback_requested(self, event: RollbackRequested) -> None:
        if self._on_chat_rollback:
            self._on_chat_rollback(event.turn_index)

    @on(DeepCheckpointAction)
    def on_deep_checkpoint_action(self, event: DeepCheckpointAction) -> None:
        """Forward a Deep Solver checkpoint button click to the agent side.

        The agent owns the ``apply_checkpoint_action`` handler because it
        needs live references to ``messages`` and the enhanced system
        prompt; the TUI just relays the intent. On 'continue' the agent
        also mounts a context chip via the bridge.
        """
        handler = getattr(self, "_on_deep_checkpoint", None)
        if handler:
            try:
                handler(event.cp_id, event.action)
            except Exception:
                pass

    @on(ChatSubmitted)
    def on_chat_message(self, event: ChatSubmitted) -> None:
        text = event.text
        bubble = getattr(event, "bubble_text", None) or event.text
        paths = list(event.image_paths or [])
        if paths:
            block = "\n".join(f"[Image file: {p}]" for p in paths)
            text = block + "\n\n" + text
        if self._on_chat_submit:
            hints = self.chat.get_context_hints()
            if hints:
                text = (
                    "[Pinned paths — check these files if relevant]\n"
                    + "\n".join(hints)
                    + "\n\n---\n"
                    + text
                )
            try:
                self._on_chat_submit(text, bubble)
            except TypeError:
                self._on_chat_submit(text)

    @on(ModelChanged)
    def on_model_changed(self, event: ModelChanged) -> None:
        self._model_name = event.model_id
        self._update_status()
        if self._on_model_change:
            self._on_model_change(event.model_id)

    @on(ModeToggled)
    def on_mode_toggled(self, event: ModeToggled) -> None:
        self._mode_name = str(event.mode or "agent")
        self._update_status()
        if self._on_mode_toggle:
            self._on_mode_toggle(event.mode)

    @on(StopRequested)
    def on_stop_requested(self, event: StopRequested) -> None:
        if self._bridge:
            self._bridge.request_stop()
            self.chat.add_warning("Остановка — агент завершит текущую операцию")

    @on(Button.Pressed, "#app-exit-btn")
    def on_exit_click(self) -> None:
        self._run_shutdown_hooks()
        self.exit()

    @on(Button.Pressed, "#app-chat-btn")
    def on_chat_picker_click(self) -> None:
        """Open the existing session picker screen (same one used at startup)."""
        try:
            from Agent.checkpoint import list_sessions

            rows = list_sessions(limit=80)
        except Exception:
            rows = []
        self.push_screen(SessionPickerScreen(rows), self._on_session_picker_result)

    def action_quit(self) -> None:
        self._run_shutdown_hooks()
        self.exit()

    def _run_shutdown_hooks(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._on_app_close:
            try:
                self._on_app_close()
            except Exception:
                pass

    def on_shutdown(self, event: events.Shutdown) -> None:
        self._run_shutdown_hooks()

    @on(AgentWorkerSelected)
    def on_agent_worker_selected(self, event: AgentWorkerSelected) -> None:
        self.workspace.focus_chat_tab()
        self.chat.set_view_worker(event.worker_id)

    @on(AgentMainChatSelected)
    def on_agent_main_chat(self) -> None:
        self.workspace.focus_chat_tab()
        self.chat.set_view_worker(None)

    @on(CloseWorkspaceTab)
    def on_close_workspace_tab(self, event: CloseWorkspaceTab) -> None:
        self.workspace.close_tab_by_id(event.tab_id)

    def action_focus_chat(self) -> None:
        try:
            self.workspace.focus_chat_tab()
            self.query_one("#chat-input", TextArea).focus()
        except Exception:
            pass

    def action_toggle_sidebar(self) -> None:
        try:
            col = self.query_one("#col-left", Vertical)
            col.display = not col.display
        except Exception:
            pass

    def action_save_file(self) -> None:
        try:
            tabs = self.workspace._tabs()
            aid = tabs.active
            if not aid or aid == CHAT_TAB_ID:
                return
            pane = tabs.get_pane(aid)
            editor = pane.query_one(FileEditorTabPane)
            editor._save_to_disk()
        except Exception:
            pass

    def action_toggle_find(self) -> None:
        self.notify("Поиск: откройте файл во вкладке и используйте Ctrl+F в редакторе")

    def action_goto_line(self) -> None:
        self.notify("Переход к строке из вкладки файла")

    def action_close_tab(self) -> None:
        self.workspace.close_active_if_not_chat()

    def action_open_file(self, path: str) -> None:
        try:
            self.workspace.open_path(Path(path).expanduser())
        except Exception as e:
            self.notify(f"Не удалось открыть: {e}", severity="warning")

    def action_run_current_file(self) -> None:
        try:
            tabs = self.workspace._tabs()
            aid = tabs.active
            if not aid or aid == CHAT_TAB_ID:
                self.notify("Откройте файл во вкладке", severity="warning")
                return
            pane = tabs.get_pane(aid)
            ed = pane.query_one(FileEditorTabPane)
            self.post_message(RunFileRequested(ed._path))
        except Exception:
            self.notify("Нет активного файла", severity="warning")

    def action_stop_agent(self) -> None:
        if self._bridge:
            self._bridge.request_stop()
            self.chat.add_warning("Остановка запрошена")

    def _set_width(self, widget_id: str, width: int) -> None:
        try:
            widget = self.query_one(widget_id)
            widget.styles.width = width
        except Exception:
            pass

    def action_resize_left_smaller(self) -> None:
        self._left_width = max(14, self._left_width - 4)
        self._set_width("#col-left", self._left_width)

    def action_resize_left_larger(self) -> None:
        self._left_width = min(50, self._left_width + 4)
        self._set_width("#col-left", self._left_width)

    def set_bridge(self, bridge) -> None:
        self._bridge = bridge

    def update_status(self, model: str = "", branch: str = "",
                      tokens: str = "", rag: str = "") -> None:
        from rich.text import Text
        from Interface.panels.ai_chat._constants import MODE_ICONS, MODE_COLORS, PURPLE

        if model:
            self._model_name = model
        if branch:
            self._branch = branch
        mode_key = (self._mode_name or "agent").lower()
        mode_icon = MODE_ICONS.get(mode_key, "•")
        mode_color = MODE_COLORS.get(mode_key, PURPLE)

        out = Text()
        if self._model_name:
            out.append(f" {self._model_name}", style="#9CA3AF")
        if self._branch:
            if len(out) > 0:
                out.append("  │  ", style="dim")
            out.append(f"⎇ {self._branch}", style="#9CA3AF")
        if tokens:
            if len(out) > 0:
                out.append("  │  ", style="dim")
            out.append(tokens, style="#9CA3AF")
        if rag:
            if len(out) > 0:
                out.append("  │  ", style="dim")
            out.append(f"RAG: {rag}", style="#9CA3AF")
        if len(out) > 0:
            out.append("  │  ", style="dim")
        out.append(f" {mode_icon} {mode_key.upper()} ", style=f"bold white on {mode_color}")
        out.append("   F2:режим  F3:модель  Esc:чат", style="dim")
        self.status_bar.update(out)

    def _update_status(self) -> None:
        self.update_status(model=self._model_name, branch=self._branch)

    # ─── Vi editor integration ────────────────────────

    def on_vi_mode_changed(self, event: "ViModeChanged") -> None:
        """Update the vi mode indicator in the status bar."""
        try:
            indicator = self.query_one("#vi-mode-indicator", Static)
            color = event.color
            label = event.mode.upper()
            hint = event.hint
            indicator.update(f"[{label}] {hint}  F1:справка")
            indicator.styles.color = color
        except Exception:
            pass

    def on_descendant_focus(self, event) -> None:
        """Dim the vi-mode indicator when focus leaves a vi-enabled editor,
        so the status bar never implies vi keys are active in chat/settings/
        the file explorer (those widgets don't run the vi state machine).
        """
        try:
            from Interface.panels.vi_textarea import ViEditorArea
            indicator = self.query_one("#vi-mode-indicator", Static)
            focused = event.widget
            in_vi_editor = isinstance(focused, ViEditorArea) or any(
                isinstance(a, ViEditorArea) for a in focused.ancestors
            )
            if in_vi_editor:
                indicator.remove_class("vi-inactive")
            else:
                indicator.add_class("vi-inactive")
        except Exception:
            pass

    def action_open_keybindings(self) -> None:
        """Open the keybindings reference tab."""
        try:
            self.workspace.open_settings_tab(section="keybindings")
        except Exception:
            pass

    def action_sidebar_grow(self) -> None:
        self._left_width = min(50, self._left_width + 2)
        self._set_width("#col-left", self._left_width)

    def action_sidebar_shrink(self) -> None:
        self._left_width = max(14, self._left_width - 2)
        self._set_width("#col-left", self._left_width)

    def action_sidebar_reset(self) -> None:
        self._left_width = 28
        self._set_width("#col-left", self._left_width)

    def action_focus_file_explorer(self) -> None:
        try:
            self.file_explorer.focus()
        except Exception:
            pass

    def action_focus_active_agents(self) -> None:
        try:
            self.active_agents.focus()
        except Exception:
            pass

    def action_open_settings(self) -> None:
        try:
            self.workspace.open_settings_tab()
        except Exception:
            pass

    def action_new_terminal(self) -> None:
        try:
            self.workspace.open_terminal_tab()
        except Exception as e:
            self.notify(f"Терминал недоступен: {e}", severity="warning")

    def _cycle_select(self, select_id: str, step: int = 1) -> None:
        """Move a Select widget to its next/previous option and fire its handler."""
        from textual.widgets import Select

        sel = self.query_one(select_id, Select)
        options = [val for _, val in sel._options if val is not Select.BLANK]
        if not options:
            return
        try:
            idx = options.index(sel.value)
        except ValueError:
            idx = 0
        sel.value = options[(idx + step) % len(options)]

    def action_cycle_mode(self) -> None:
        try:
            self.workspace.focus_chat_tab()
            self._cycle_select("#mode-select", 1)
        except Exception:
            pass

    def action_cycle_model(self) -> None:
        try:
            self.workspace.focus_chat_tab()
            self._cycle_select("#model-select", 1)
        except Exception:
            pass

    def action_next_tab(self) -> None:
        try:
            self.workspace.action_next_tab()
        except Exception:
            pass

    def action_prev_tab(self) -> None:
        try:
            self.workspace.action_prev_tab()
        except Exception:
            pass

    def action_focus_tab(self, index: int) -> None:
        try:
            self.workspace.action_focus_tab(index)
        except Exception:
            pass


