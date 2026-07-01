# Module: Agent.tools.decompose_tool

## Purpose

Heuristic task decomposition tool for weak models.

---

## Responsibilities

- Heuristic task decomposition tool for weak models.

---

## Public API

|name|description|
|---|---|
|task_decompose|Декомпозиция задачи на атомарные шаги (без LLM, на эвристиках).

task: описание задачи
context_files: список файлов контекста (опционально)
mode: code | research | docs | refactor | auto
Возвращает список шагов с типом и рекомендуемыми инст…|

---

## Dependencies

- `__future__`
- `langchain_core.tools`
- `re`
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

- `Agent/tools/decompose_tool.py`

---

## Entry Points


---

## API / route hints

