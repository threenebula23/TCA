MODULE: Agent.tools.env_tool

PURPOSE:
Environment information tool.

PUBLIC_API:
|name|description|
|---|---|
|env_info|Информация об окружении: Python, пакеты, команды, ОС.

check_packages: ['django','numpy'] — проверить версии конкретных пакетов.
check_commands: ['git','node','npm'] — проверить наличие команд в PATH.
Вызывай в начале сессии или перед устан…|

DEPENDENCIES:
- __future__
- importlib.metadata
- importlib.util
- langchain_core.tools
- os
- platform
- shutil
- sys
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
