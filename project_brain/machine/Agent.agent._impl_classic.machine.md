MODULE: Agent.agent._impl_classic

PURPOSE:
Classic CLI agent loop.

PUBLIC_API:
|name|description|
|---|---|
|run_coding_agent_loop||

DEPENDENCIES:
- Agent.command_router._main
- Agent.deep_solver
- Agent.path_utils
- Agent.project_brain
- Agent.project_brain.agent_architecture
- Agent.prompts
- Agent.rag
- Agent.stream_chat_mode
- Agent.tool_registry
- Interface.input_widget
- Interface.ui_prefs
- Interface.visualization
- __future__
- _impl_prepare
- langchain_core.messages
- os
- questionary
- simple_term_menu
- tool_registry

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Interface/panels/file_explorer.py
- lorne.py
- tests/test_ollama_provider.py
- tests/test_prompt_budget.py

RISKS:
