# Module: Agent.tools.planning_tool

## Purpose

Инструменты плана: модель может создать план, обновлять статусы и продолжать выполнение.

---

## Responsibilities

- Инструменты плана: модель может создать план, обновлять статусы и продолжать выполнение.

---

## Public API

|name|description|
|---|---|
|save_plan|Сохраняет план (список шагов) для текущей задачи. Используй перед выполнением большой задачи.|
|load_plan|Загружает текущий план (если есть).|
|update_plan|Обновляет статус шага плана. status: pending | in_progress | completed | blocked.|
|clear_plan|Удаляет текущий план.|

---

## Dependencies

- `Agent.runtime_paths`
- `__future__`
- `datetime`
- `json`
- `langchain_core.tools`
- `pathlib`
- `typing`

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

- `Agent/tools/planning_tool.py`

---

## Entry Points


---

## API / route hints

