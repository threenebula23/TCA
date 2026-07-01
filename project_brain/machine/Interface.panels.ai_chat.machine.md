MODULE: Interface.panels.ai_chat

PURPOSE:
Панель чата Lorne (пакет). Публичный API — :class:`AIChatPanel` и сообщения.

PUBLIC_API:


DEPENDENCIES:
- Interface.panels.creator_progress
- Interface.panels.deep_checkpoint
- Interface.panels.diff_block
- Interface.panels.download_block
- Interface.panels.tool_card
- __future__
- _accent_dialog
- _blocks
- _css
- _helpers
- _messages
- _mixin_events
- _mixin_setup
- _mixin_stream
- os
- pathlib
- re
- rich.markdown
- rich.text
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
- Interface/tui_app.py
- Interface/tui_bridge.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
