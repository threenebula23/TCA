MODULE: Agent.tools.versioning_tool

PURPOSE:
Инструменты отката/версий файлов для агента.

PUBLIC_API:
|name|description|
|---|---|
|list_file_versions|Показывает последние сохранённые версии файла для отката.|
|rollback_file|Откатывает файл. Если version_id пустой — откат к последней сохранённой версии.|

DEPENDENCIES:
- Agent.path_utils
- Agent.versioning
- __future__
- langchain_core.tools
- path_utils
- pathlib
- typing
- versioning

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
