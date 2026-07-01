MODULE: Agent.tools.download_tool

PURPOSE:
`download_file` — pull an image / archive / dataset from an HTTP(S) URL.

PUBLIC_API:
|name|description|
|---|---|
|cancel_download|UI-facing: mark a download as cancelled. Returns True if a live
download was found, False otherwise.|
|download_file|HTTP(S) загрузка в workspace; dest пустой → ./downloads/<basename>; max_bytes 0 = 200 МиБ; прогресс/отмена в TUI.|

DEPENDENCIES:
- Interface.branding
- Interface.tui_bridge
- __future__
- langchain_core.tools
- os
- path_utils
- pathlib
- threading
- time
- typing
- urllib.error
- urllib.parse
- urllib.request
- uuid

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Agent/tools/__init__.py
- Interface/panels/ai_chat/_mixin_events.py
- tests/test_file_ops.py
- tests/test_latest_fixes.py
- tests/test_ollama_provider.py

RISKS:
