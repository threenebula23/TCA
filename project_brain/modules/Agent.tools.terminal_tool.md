# Module: Agent.tools.terminal_tool

## Purpose

Инструмент выполнения команд в терминале (Windows/Unix).

---

## Responsibilities

- Инструмент выполнения команд в терминале (Windows/Unix).

---

## Public API

|name|description|
|---|---|
|run_command|Shell; stdin закрыт (`-y`/pipe). cwd пусто = проект. background=True — отдельный процесс, лог в каталоге данных проекта (``.lorne`` / legacy ``.tca``) / ``background_cmd.log`` (dev-серверы).|

---

## Dependencies

- `Agent.runtime_paths`
- `Interface.graph_display`
- `Interface.tui_bridge`
- `Terminal.runner`
- `__future__`
- `contextlib`
- `langchain_core.tools`
- `os`
- `path_utils`
- `pathlib`
- `runtime_paths`
- `sys`
- `time`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/creator_mode.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/__init__.py`
- `tests/test_file_ops.py`
- `tests/test_latest_fixes.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/terminal_tool.py`

---

## Entry Points


---

## API / route hints

