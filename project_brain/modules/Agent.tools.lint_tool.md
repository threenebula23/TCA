# Module: Agent.tools.lint_tool

## Purpose

Run linters and return structured error reports.

---

## Responsibilities

- Run linters and return structured error reports.

---

## Public API

|name|description|
|---|---|
|lint_check|Запускает линтер на path и возвращает ошибки структурировано.

path: путь к файлу или директории
fix: если True — применить автоисправления (ruff --fix)
linter: auto | ruff | pylint | eslint | tsc
Всегда вызывай после правки кода перед тем …|

---

## Dependencies

- `__future__`
- `json`
- `langchain_core.tools`
- `pathlib`
- `subprocess`
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

- `Agent/tools/lint_tool.py`

---

## Entry Points


---

## API / route hints

