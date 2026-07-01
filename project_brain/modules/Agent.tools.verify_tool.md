# Module: Agent.tools.verify_tool

## Purpose

Post-edit result verification tool.

---

## Responsibilities

- Post-edit result verification tool.

---

## Public API

|name|description|
|---|---|
|verify_result|Проверка результата после правок: синтаксис, наличие строк, импорты, файлы.

checks: список проверок, каждая из которых — dict с ключом 'type':
  {"type": "syntax",       "path": "foo.py"}
  {"type": "contains",     "path": "foo.py",  "text…|

---

## Dependencies

- `__future__`
- `importlib.util`
- `langchain_core.tools`
- `pathlib`
- `py_compile`
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

- `Agent/tools/verify_tool.py`

---

## Entry Points


---

## API / route hints

