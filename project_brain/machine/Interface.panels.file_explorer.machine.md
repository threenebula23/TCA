MODULE: Interface.panels.file_explorer

PURPOSE:
File Explorer panel — файлы и настройки.

PUBLIC_API:


DEPENDENCIES:
- Agent.agent
- Agent.llm_provider
- Interface.modal_style
- Interface.panels.workspace_center
- Interface.themes
- Interface.ui_prefs
- __future__
- os
- pathlib
- rich.markdown
- rich.text
- shutil
- textual
- textual.app
- textual.binding
- textual.containers
- textual.message
- textual.screen
- textual.widgets
- threading
- time
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Interface/panels/code_editor/_part_b.py
- Interface/panels/version_control.py
- Interface/tui_bridge.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
