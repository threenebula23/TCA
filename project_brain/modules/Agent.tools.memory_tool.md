# Module: Agent.tools.memory_tool

## Purpose

Persistent in-session key-value memory for the agent.

---

## Responsibilities

- Persistent in-session key-value memory for the agent.

---

## Public API

|name|description|
|---|---|
|structured_memory|Сессионная KV-память: set/get/list/delete/clear.

action: set | get | list | delete | clear
key: ключ записи (для set/get/delete)
value: значение (для set)
namespace: пространство имён (по умолчанию 'default'; воркеры Creator используют wor…|

---

## Dependencies

- `__future__`
- `json`
- `langchain_core.tools`
- `pathlib`
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

- `Agent/tools/memory_tool.py`

---

## Entry Points


---

## API / route hints

