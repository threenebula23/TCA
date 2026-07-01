MODULE: Agent.git_integration

PURPOSE:
Git integration for Lorne — automatic snapshots and rollback.

PUBLIC_API:
|name|description|
|---|---|
|get_git_manager|Get or create the singleton GitManager.|

DEPENDENCIES:
- __future__
- git
- pathlib
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_tui.py
- Agent/command_router/_mixin_handlers.py
- Agent/tools/file_ops.py
- Agent/tools/git_tool.py
- Interface/panels/version_control.py
- tests/test_ollama_provider.py

RISKS:
