# Module: Agent.tools.batch_replace_tool

## Purpose

Apply multiple find-and-replace operations to a file in one call.

---

## Responsibilities

- Apply multiple find-and-replace operations to a file in one call.

---

## Public API

|name|description|
|---|---|
|batch_replace|Применяет несколько замен к файлу за один вызов.

path: путь к файлу
replacements: [{"from": "old_name", "to": "new_name"}, ...]
regex: если True — 'from' интерпретируется как regex
Идеально для рефакторинга: переименование символа, замена …|

---

## Dependencies

- `__future__`
- `langchain_core.tools`
- `pathlib`
- `re`
- `shutil`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/batch_replace_tool.py`

---

## Entry Points


---

## API / route hints

