MODULE: Interface.tui_app

PURPOSE:
IDE Lorne (Textual): чат по центру, файлы и агенты слева, вкладки рабочей области.

PUBLIC_API:


DEPENDENCIES:
- Agent.checkpoint
- Interface.branding
- Interface.panels.ai_chat._constants
- Interface.panels.vi_textarea
- Interface.themes
- Interface.ui_prefs
- __future__
- panels.active_agents_panel
- panels.ai_chat
- panels.code_editor
- panels.file_explorer
- panels.vi_textarea
- panels.workspace_center
- pathlib
- rich.text
- session_picker_screen
- shlex
- subprocess
- sys
- textual
- textual.app
- textual.binding
- textual.containers
- textual.widgets
- threading
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_tui.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
