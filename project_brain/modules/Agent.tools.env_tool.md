# Module: Agent.tools.env_tool

## Purpose

Environment information tool.

---

## Responsibilities

- Environment information tool.

---

## Public API

|name|description|
|---|---|
|env_info|Информация об окружении: Python, пакеты, команды, ОС.

check_packages: ['django','numpy'] — проверить версии конкретных пакетов.
check_commands: ['git','node','npm'] — проверить наличие команд в PATH.
Вызывай в начале сессии или перед устан…|

---

## Dependencies

- `__future__`
- `importlib.metadata`
- `importlib.util`
- `langchain_core.tools`
- `os`
- `platform`
- `shutil`
- `sys`
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

- `Agent/tools/env_tool.py`

---

## Entry Points


---

## API / route hints

