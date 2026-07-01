MODULE: Agent.tools.ast_tool

PURPOSE:
AST-based code structure analysis tool.

PUBLIC_API:
|name|description|
|---|---|
|ast_analyze|AST-анализ Python/JS/TS/MD файла: классы, функции, импорты, сигнатуры.

path: путь к файлу
query: опционально фильтрует вывод (например 'class Auth' или 'def handle').
Возвращает структуру без чтения всего файла целиком — экономит токены.
П…|

DEPENDENCIES:
- __future__
- ast
- langchain_core.tools
- pathlib
- re
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
