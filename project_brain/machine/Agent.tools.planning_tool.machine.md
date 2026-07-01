MODULE: Agent.tools.planning_tool

PURPOSE:
Инструменты плана: модель может создать план, обновлять статусы и продолжать выполнение.

PUBLIC_API:
|name|description|
|---|---|
|save_plan|Сохраняет план (список шагов) для текущей задачи. Используй перед выполнением большой задачи.|
|load_plan|Загружает текущий план (если есть).|
|update_plan|Обновляет статус шага плана. status: pending | in_progress | completed | blocked.|
|clear_plan|Удаляет текущий план.|

DEPENDENCIES:
- Agent.runtime_paths
- __future__
- datetime
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
- Agent/tools/__init__.py
- Agent/tools/compact_tools.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
