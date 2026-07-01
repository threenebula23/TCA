# Module: Agent.creator_summary

## Purpose

Единый текст итога Creator Mode для TUI, classic CLI и истории сообщений.

---

## Responsibilities

- Единый текст итога Creator Mode для TUI, classic CLI и истории сообщений.

---

## Public API

|name|description|
|---|---|
|format_creator_summary_text|Markdown: статус, все воркеры, полный result (с защитными лимитами по длине).|

---

## Dependencies

- `__future__`
- `typing`

---

## Used By

- `Agent/agent/_impl_prepare.py`
- `Agent/command_router/_main.py`
- `Agent/command_router/_mixin_handlers.py`
- `Agent/deep_solver/legacy_loop.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/creator_summary.py`

---

## Entry Points


---

## API / route hints

