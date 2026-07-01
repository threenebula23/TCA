MODULE: Agent.tools.lint_tool

PURPOSE:
Run linters and return structured error reports.

PUBLIC_API:
|name|description|
|---|---|
|lint_check|Запускает линтер на path и возвращает ошибки структурировано.

path: путь к файлу или директории
fix: если True — применить автоисправления (ruff --fix)
linter: auto | ruff | pylint | eslint | tsc
Всегда вызывай после правки кода перед тем …|

DEPENDENCIES:
- __future__
- json
- langchain_core.tools
- pathlib
- subprocess
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
