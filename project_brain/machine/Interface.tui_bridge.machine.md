MODULE: Interface.tui_bridge

PURPOSE:
Bridge between Agent loop and TUI panels.

PUBLIC_API:
|name|description|
|---|---|
|get_bridge||
|set_bridge||

DEPENDENCIES:
- Interface.panels.ai_chat._constants
- Interface.panels.file_explorer
- __future__
- pathlib
- sys
- threading
- time
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_tui.py
- Agent/background_agent_runner.py
- Agent/creator_mode.py
- Agent/graph_runner.py
- Agent/spinner.py
- Agent/tools/download_tool.py
- Agent/tools/interactive.py
- Agent/tools/terminal_tool.py
- Agent/tools/thinking_tool.py
- Interface/panels/vi_textarea.py
- Interface/visualization.py
- tests/test_branding.py
- tests/test_package_imports.py

RISKS:
