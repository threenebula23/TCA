# Module: Agent.agent._impl_tui

## Purpose

Точка входа TUI для агента Lorne.

---

## Responsibilities

- Точка входа TUI для агента Lorne.

---

## Public API

|name|description|
|---|---|
|run_tui_mode|Запуск полноэкранного TUI (Textual).|

---

## Dependencies

- `Agent.agent._impl_prepare`
- `Agent.command_router`
- `Agent.command_router._main`
- `Agent.deep_solver`
- `Agent.git_integration`
- `Agent.message_utils`
- `Agent.path_utils`
- `Agent.project_brain`
- `Agent.project_brain.agent_architecture`
- `Agent.prompts`
- `Agent.rag`
- `Agent.runtime_paths`
- `Agent.stream_chat_mode`
- `Interface.start_screen`
- `Interface.tui_app`
- `Interface.tui_bridge`
- `__future__`
- `_impl_classic`
- `_impl_prepare`
- `langchain_core.messages`
- `os`
- `sys`
- `threading`
- `traceback`

---

## Used By

- `Interface/panels/file_explorer.py`
- `lorne.py`
- `tests/test_ollama_provider.py`
- `tests/test_prompt_budget.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/agent/_impl_tui.py`

---

## Entry Points


---

## API / route hints

