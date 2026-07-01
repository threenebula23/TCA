# Module: Agent.tools.download_tool

## Purpose

`download_file` — pull an image / archive / dataset from an HTTP(S) URL.

---

## Responsibilities

- `download_file` — pull an image / archive / dataset from an HTTP(S) URL.

---

## Public API

|name|description|
|---|---|
|cancel_download|UI-facing: mark a download as cancelled. Returns True if a live
download was found, False otherwise.|
|download_file|HTTP(S) загрузка в workspace; dest пустой → ./downloads/<basename>; max_bytes 0 = 200 МиБ; прогресс/отмена в TUI.|

---

## Dependencies

- `Interface.branding`
- `Interface.tui_bridge`
- `__future__`
- `langchain_core.tools`
- `os`
- `path_utils`
- `pathlib`
- `threading`
- `time`
- `typing`
- `urllib.error`
- `urllib.parse`
- `urllib.request`
- `uuid`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/__init__.py`
- `Interface/panels/ai_chat/_mixin_events.py`
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

- `Agent/tools/download_tool.py`

---

## Entry Points


---

## API / route hints

