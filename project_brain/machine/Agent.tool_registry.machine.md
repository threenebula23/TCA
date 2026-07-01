MODULE: Agent.tool_registry

PURPOSE:
Реестр инструментов Lorne: сборка списка тулов для LLM и карты dispatch.

PUBLIC_API:
|name|description|
|---|---|
|set_tool_session_prefs||
|build_tools|Return ``(tools, custom_list)`` for the session; flags control browser / ask.|
|get_agent_mode_tools||
|build_tool_map||
|bind_tools_safe||
|reload_tools||

DEPENDENCIES:
- Agent.llm_provider
- Agent.message_utils
- Agent.rag
- Agent.tools
- Agent.tools.compact_tools
- Agent.tools.office_document_tool
- Agent.tools.parallel_helper_tool
- Agent.tools.planning_tool
- Agent.tools.qa_tool
- Agent.tools.versioning_tool
- langchain_core.tools
- llm_provider
- message_utils
- rag
- sys
- tools
- tools.ast_tool
- tools.batch_replace_tool
- tools.browser_tool
- tools.compact_tools
- tools.decompose_tool
- tools.env_tool
- tools.git_tool
- tools.lint_tool
- tools.memory_tool
- tools.multi_read_tool
- tools.notes_tool
- tools.office_document_tool
- tools.parallel_helper_tool
- tools.planning_tool
- tools.playwright_sync_tool
- tools.qa_tool
- tools.verify_tool
- tools.versioning_tool
- typing

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_prepare.py
- Agent/command_router/_main.py
- Agent/command_router/_mixin_handlers.py
- Agent/creator_mode.py
- Agent/deep_solver/_impl_a.py
- Agent/deep_solver/legacy_loop.py
- Agent/tools/parallel_helper_tool.py
- tests/test_latest_fixes.py
- tests/test_ollama_provider.py
- tests/test_prompt_budget.py

RISKS:
