MODULE: Terminal.runner

PURPOSE:
Cross-platform command execution for Lorne.

PUBLIC_API:
|name|description|
|---|---|
|run_command|Execute a shell command. Returns (stdout, stderr, returncode).|
|run_command_safe|Safe command execution for agent tools. Limits output length and catches errors.|
|run_command_detached|Start command in background; append stdout/stderr to каталог данных проекта (``.lorne`` / legacy ``.tca``) / ``<log_name>``.

Use for long-running dev servers (``npm run dev``) so the agent is not blocked.|

DEPENDENCIES:
- Agent.path_utils
- Agent.runtime_paths
- pathlib
- subprocess
- sys
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/command_router/_main.py
- Agent/tools/qa_tool.py
- Agent/tools/terminal_tool.py
- Terminal/__init__.py
- tests/test_latest_fixes.py

RISKS:
