MODULE: Agent.tools.memory_tool

PURPOSE:
Persistent in-session key-value memory for the agent.

PUBLIC_API:
|name|description|
|---|---|
|structured_memory|Сессионная KV-память: set/get/list/delete/clear.

action: set | get | list | delete | clear
key: ключ записи (для set/get/delete)
value: значение (для set)
namespace: пространство имён (по умолчанию 'default'; воркеры Creator используют wor…|

DEPENDENCIES:
- __future__
- json
- langchain_core.tools
- pathlib
- typing

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
