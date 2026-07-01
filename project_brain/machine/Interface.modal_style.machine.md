MODULE: Interface.modal_style

PURPOSE:
Shared styling helpers for ModalScreens (dialogs / popups).

PUBLIC_API:
|name|description|
|---|---|
|current_accent||
|apply_accent_to|Paint the accent colour onto a modal's container border and title label.

This is called from ``on_mount`` so the popup matches the user's current
theme immediately (no app restart needed).|

DEPENDENCIES:
- Interface.themes
- Interface.ui_prefs
- __future__
- rich.text
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Interface/panels/ai_chat/_accent_dialog.py
- Interface/panels/ai_chat/_messages.py
- Interface/panels/file_explorer.py
- Interface/session_picker_screen.py
- Interface/start_screen.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
