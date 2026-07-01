MODULE: Agent.tool_schemas

PURPOSE:
Pydantic-схемы аргументов инструментов: валидация, сжатие лишних полей, подсказки модели.

PUBLIC_API:
|name|description|
|---|---|
|validate_tool_arguments|Возвращает (нормализованные аргументы, текст ошибки или None).|

DEPENDENCIES:
- __future__
- json
- pydantic
- typing

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/creator_mode.py
- Agent/deep_solver/_impl_a.py
- Agent/deep_solver/legacy_loop.py
- Agent/graph_runner.py
- tests/test_latest_fixes.py
- tests/test_ollama_provider.py
- tests/test_project_brain_tool.py

RISKS:
