# Module: Agent.tools.versioning_tool

## Purpose

Инструменты отката/версий файлов для агента.

---

## Responsibilities

- Инструменты отката/версий файлов для агента.

---

## Public API

|name|description|
|---|---|
|list_file_versions|Показывает последние сохранённые версии файла для отката.|
|rollback_file|Откатывает файл. Если version_id пустой — откат к последней сохранённой версии.|

---

## Dependencies

- `Agent.path_utils`
- `Agent.versioning`
- `__future__`
- `langchain_core.tools`
- `path_utils`
- `pathlib`
- `typing`
- `versioning`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/__init__.py`
- `Agent/tools/compact_tools.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/versioning_tool.py`

---

## Entry Points


---

## API / route hints

