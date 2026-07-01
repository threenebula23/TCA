MODULE: Agent.tools.terminal_tool

PURPOSE:
Инструмент выполнения команд в терминале (Windows/Unix).

PUBLIC_API:
|name|description|
|---|---|
|run_command|Shell; stdin закрыт (`-y`/pipe). cwd пусто = проект. background=True — отдельный процесс, лог в каталоге данных проекта (``.lorne`` / legacy ``.tca``) / ``background_cmd.log`` (dev-серверы).|

DEPENDENCIES:
- Agent.runtime_paths
- Interface.graph_display
- Interface.tui_bridge
- Terminal.runner
- __future__
- contextlib
- langchain_core.tools
- os
- path_utils
- pathlib
- runtime_paths
- sys
- time
- typing

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/creator_mode.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- Agent/tools/__init__.py
- tests/test_file_ops.py
- tests/test_latest_fixes.py
- tests/test_ollama_provider.py

RISKS:
