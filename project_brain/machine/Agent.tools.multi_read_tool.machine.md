MODULE: Agent.tools.multi_read_tool

PURPOSE:
Read multiple files in a single tool call.

PUBLIC_API:
|name|description|
|---|---|
|multi_read|Читает несколько файлов одним вызовом (до 8 штук, max_lines_each строк каждый).

paths: список путей к файлам (максимум 8)
max_lines_each: максимум строк на файл (по умолчанию 200)
Возвращает {path: {content, total_lines, truncated}} для ка…|

DEPENDENCIES:
- __future__
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
