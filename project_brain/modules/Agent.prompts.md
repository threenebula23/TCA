# Module: Agent.prompts

## Purpose

Per-mode system prompt fragments for Lorne v1.0.

---

## Responsibilities

- Per-mode system prompt fragments for Lorne v1.0.

---

## Public API

|name|description|
|---|---|
|mode_prompt_addon|Return system prompt fragment for *mode* slug, or empty string.|

---

## Dependencies

- `__future__`
- `pathlib`
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

- `Agent/prompts/__init__.py`

---

## Entry Points


---

## API / route hints

