MODULE: Agent.command_router._mixin_handlers

PURPOSE:
Маршрутизатор slash-команд (classic CLI и воркер TUI). Возврат: обработано / не команда / exit.

PUBLIC_API:


DEPENDENCIES:
- Agent.creator_summary
- Agent.git_integration
- Agent.llm_provider
- Agent.multiagent
- Agent.rag
- Agent.tool_registry
- Interface.cli_theme
- Interface.ui_prefs
- Interface.visualization
- creator_summary
- json
- langchain_core.messages
- os
- pathlib
- rich
- rich.panel
- rich.syntax
- rich.table
- tool_registry
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- tests/test_ollama_provider.py

RISKS:
