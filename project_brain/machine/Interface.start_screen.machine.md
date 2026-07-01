MODULE: Interface.start_screen

PURPOSE:
Initial TUI screen for selecting a project before launching the main app.

PUBLIC_API:
|name|description|
|---|---|
|load_recent_projects||
|save_recent_project||
|select_project_path||

DEPENDENCIES:
- Agent.runtime_paths
- Interface.branding
- Interface.modal_style
- Interface.ui_prefs
- __future__
- collections
- json
- math
- pathlib
- pyfiglet
- rich.align
- rich.style
- rich.text
- textual
- textual.app
- textual.containers
- textual.screen
- textual.widget
- textual.widgets
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_tui.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
