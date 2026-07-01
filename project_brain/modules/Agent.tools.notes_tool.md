# Module: Agent.tools.notes_tool

## Purpose

Free-form session notes and hypotheses scratch pad.

---

## Responsibilities

- Free-form session notes and hypotheses scratch pad.

---

## Public API

|name|description|
|---|---|
|session_notes|Свободные заметки сессии: append/read/clear/search.

action: append | read | clear | search
content: текст заметки (для append)
tag: категория ('bug', 'decision', 'todo', 'observation', 'hypothesis', 'fact')
Записывай сюда рассуждения, гипо…|

---

## Dependencies

- `__future__`
- `datetime`
- `langchain_core.tools`
- `pathlib`

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

- `Agent/tools/notes_tool.py`

---

## Entry Points


---

## API / route hints

