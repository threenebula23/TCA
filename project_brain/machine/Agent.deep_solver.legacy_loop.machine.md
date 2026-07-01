MODULE: Agent.deep_solver.legacy_loop

PURPOSE:
Python module `Agent.deep_solver.legacy_loop`.

PUBLIC_API:
|name|description|
|---|---|
|is_running||
|submit_user_message||
|register_checkpoint||
|get_checkpoint||
|clear_checkpoint||
|list_checkpoints||
|run_deep_solver||
|apply_checkpoint_action||

DEPENDENCIES:
- Agent.checkpoint
- Agent.creator_mode
- Agent.creator_provider
- Agent.creator_summary
- Agent.llm_provider
- Agent.message_utils
- Agent.path_utils
- Agent.runtime_paths
- Agent.tool_registry
- Agent.tool_schemas
- Agent.tools
- Interface.visualization
- __future__
- checkpoint
- creator_mode
- creator_provider
- creator_summary
- json
- langchain_core.messages
- langchain_core.tools
- llm_provider
- message_utils
- os
- pathlib
- runtime_paths
- threading
- time
- tool_registry
- tool_schemas
- typing
- uuid

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_tui.py
- tests/test_deep_solver.py
- tests/test_ollama_provider.py
- tests/test_package_imports.py

RISKS:
