MODULE: Agent.agent._impl_tui

PURPOSE:
Точка входа TUI для агента Lorne.

PUBLIC_API:
|name|description|
|---|---|
|run_tui_mode|Запуск полноэкранного TUI (Textual).|

DEPENDENCIES:
- Agent.agent._impl_prepare
- Agent.command_router
- Agent.command_router._main
- Agent.deep_solver
- Agent.git_integration
- Agent.message_utils
- Agent.path_utils
- Agent.project_brain
- Agent.project_brain.agent_architecture
- Agent.prompts
- Agent.rag
- Agent.runtime_paths
- Agent.stream_chat_mode
- Interface.start_screen
- Interface.tui_app
- Interface.tui_bridge
- __future__
- _impl_classic
- _impl_prepare
- langchain_core.messages
- os
- sys
- threading
- traceback

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Interface/panels/file_explorer.py
- lorne.py
- tests/test_ollama_provider.py
- tests/test_prompt_budget.py

RISKS:
