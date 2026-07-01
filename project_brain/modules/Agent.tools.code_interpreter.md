# Module: Agent.tools.code_interpreter

## Purpose

Code Interpreter tool for Lorne agent.

---

## Responsibilities

- Code Interpreter tool for Lorne agent.

---

## Public API

|name|description|
|---|---|
|code_interpreter|Выполняет произвольный Python-код и возвращает stdout/stderr.
Используй для вычислений, обработки данных, проверки алгоритмов.
Код запускается в отдельном процессе; stdin закрыт — input() даст EOF, не ожидай интерактивного ввода.
Доступны с…|

---

## Dependencies

- `__future__`
- `langchain_core.tools`
- `os`
- `subprocess`
- `sys`
- `tempfile`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/__init__.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/code_interpreter.py`

---

## Entry Points


---

## API / route hints

