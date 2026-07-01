MODULE: Agent.background_agent_runner

PURPOSE:
Short-lived background agent: LLM + tools in a worker thread (testing while main blocks).

PUBLIC_API:
|name|description|
|---|---|
|run_short_agent_loop|Synchronous mini-loop; used from worker thread.|
|start_background_job||
|get_job_status||
|wait_for_job||

DEPENDENCIES:
- Agent.message_utils
- Agent.tool_schemas
- Agent.tools
- Interface.tui_bridge
- __future__
- json
- langchain_core.messages
- os
- threading
- time
- typing
- uuid

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/tools/parallel_helper_tool.py
- tests/test_ollama_provider.py

RISKS:
