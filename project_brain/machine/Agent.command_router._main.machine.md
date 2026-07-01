MODULE: Agent.command_router._main

PURPOSE:
Маршрутизатор slash-команд (classic CLI и воркер TUI). Возврат: обработано / не команда / exit.

PUBLIC_API:


DEPENDENCIES:
- Agent.agent._impl_prepare
- Agent.creator_summary
- Agent.llm_provider
- Agent.message_utils
- Agent.rag
- Agent.tool_registry
- Interface.panels.usage_calendar
- Interface.ui_prefs
- Interface.visualization
- Terminal.runner
- _mixin_handlers
- creator_summary
- json
- langchain_core.messages
- message_utils
- os
- pathlib
- rich
- rich.console
- rich.panel
- rich.rule
- rich.table
- rich.text
- simple_term_menu
- tool_registry
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- tests/test_ollama_provider.py

RISKS:
