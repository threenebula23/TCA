# Module: Agent.tools.git_tool

## Purpose

Git tools for the Lorne agent — log, diff, rollback via LangChain @tool.

---

## Responsibilities

- Git tools for the Lorne agent — log, diff, rollback via LangChain @tool.

---

## Public API

|name|description|
|---|---|
|git_log|Показать историю Git-коммитов. path — фильтр по файлу (пусто = весь проект), limit — макс. число коммитов.|
|git_diff|Показать diff. commit — хеш коммита (пусто = текущие изменения).|
|git_rollback_file|Откатить файл к указанному коммиту. path — путь к файлу, commit — хеш (пусто = последний коммит).|
|git_status|Показать текущий статус Git-репозитория: ветка, изменения, staged файлы.|

---

## Dependencies

- `Agent.git_integration`
- `langchain_core.tools`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
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

- `Agent/tools/git_tool.py`

---

## Entry Points


---

## API / route hints

