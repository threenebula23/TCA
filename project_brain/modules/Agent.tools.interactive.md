# Module: Agent.tools.interactive

## Purpose

Интерактивные инструменты: запрос ввода у пользователя в терминале.

---

## Responsibilities

- Интерактивные инструменты: запрос ввода у пользователя в терминале.

---

## Public API

|name|description|
|---|---|
|ask_user|Спросить пользователя в терминале. Выводит question и возвращает ответ пользователя. Используй для подтверждения действий (например, запуск команды), выбора варианта или уточнения.|

---

## Dependencies

- `Interface.tui_bridge`
- `langchain_core.tools`
- `sys`
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

- `Agent/tools/interactive.py`

---

## Entry Points


---

## API / route hints

