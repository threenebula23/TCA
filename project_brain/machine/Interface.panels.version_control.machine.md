MODULE: Interface.panels.version_control

PURPOSE:
Version Control panel — branch management + commit history with file selection.

PUBLIC_API:


DEPENDENCIES:
- Agent.git_integration
- Interface.panels.file_explorer
- Interface.themes
- Interface.ui_prefs
- __future__
- hashlib
- pathlib
- rich.text
- textual
- textual.app
- textual.containers
- textual.message
- textual.widgets
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
