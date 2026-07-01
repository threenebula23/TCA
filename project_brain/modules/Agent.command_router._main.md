# Module: Agent.command_router._main

## Purpose

Маршрутизатор slash-команд (classic CLI и воркер TUI). Возврат: обработано / не команда / exit.

---

## Responsibilities

- Маршрутизатор slash-команд (classic CLI и воркер TUI). Возврат: обработано / не команда / exit.

---

## Public API



---

## Dependencies

- `Agent.agent._impl_prepare`
- `Agent.creator_summary`
- `Agent.llm_provider`
- `Agent.message_utils`
- `Agent.rag`
- `Agent.tool_registry`
- `Interface.panels.usage_calendar`
- `Interface.ui_prefs`
- `Interface.visualization`
- `Terminal.runner`
- `_mixin_handlers`
- `creator_summary`
- `json`
- `langchain_core.messages`
- `message_utils`
- `os`
- `pathlib`
- `rich`
- `rich.console`
- `rich.panel`
- `rich.rule`
- `rich.table`
- `rich.text`
- `simple_term_menu`
- `tool_registry`
- `typing`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_prepare.py`
- `Agent/agent/_impl_tui.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/command_router/_main.py`

---

## Entry Points


---

## API / route hints

