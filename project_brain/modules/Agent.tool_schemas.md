# Module: Agent.tool_schemas

## Purpose

Pydantic-схемы аргументов инструментов: валидация, сжатие лишних полей, подсказки модели.

---

## Responsibilities

- Pydantic-схемы аргументов инструментов: валидация, сжатие лишних полей, подсказки модели.

---

## Public API

|name|description|
|---|---|
|validate_tool_arguments|Возвращает (нормализованные аргументы, текст ошибки или None).|

---

## Dependencies

- `__future__`
- `json`
- `pydantic`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/creator_mode.py`
- `Agent/deep_solver/_impl_a.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/graph_runner.py`
- `tests/test_latest_fixes.py`
- `tests/test_ollama_provider.py`
- `tests/test_project_brain_tool.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tool_schemas.py`

---

## Entry Points


---

## API / route hints

