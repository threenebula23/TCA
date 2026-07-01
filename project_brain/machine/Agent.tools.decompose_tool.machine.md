MODULE: Agent.tools.decompose_tool

PURPOSE:
Heuristic task decomposition tool for weak models.

PUBLIC_API:
|name|description|
|---|---|
|task_decompose|Декомпозиция задачи на атомарные шаги (без LLM, на эвристиках).

task: описание задачи
context_files: список файлов контекста (опционально)
mode: code | research | docs | refactor | auto
Возвращает список шагов с типом и рекомендуемыми инст…|

DEPENDENCIES:
- __future__
- langchain_core.tools
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
