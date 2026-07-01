# Module: Agent.git_integration

## Purpose

Git integration for Lorne — automatic snapshots and rollback.

---

## Responsibilities

- Git integration for Lorne — automatic snapshots and rollback.

---

## Public API

|name|description|
|---|---|
|get_git_manager|Get or create the singleton GitManager.|

---

## Dependencies

- `__future__`
- `git`
- `pathlib`
- `typing`

---

## Used By

- `Agent/agent/_impl_tui.py`
- `Agent/command_router/_mixin_handlers.py`
- `Agent/tools/file_ops.py`
- `Agent/tools/git_tool.py`
- `Interface/panels/version_control.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/git_integration.py`

---

## Entry Points


---

## API / route hints

