# Module: Agent.project_brain.build

## Purpose

Generate ``project_brain/`` via Relator templates (or Markdown fallback).

---

## Responsibilities

- Generate ``project_brain/`` via Relator templates (or Markdown fallback).

---

## Public API

|name|description|
|---|---|
|append_changelog_entry|Append a single changelog entry to the brain changelog JSONL file.|
|build_brain_markdown|Render brain tree under ``project_brain/``; return written file paths.

Uses Relator 1.3 with ``save_context=True`` so each render saves a
``*.relator-context.json`` file for incremental rebuilds.
Loads changelog entries from ``.lorne/brain…|

---

## Dependencies

- `__future__`
- `context_builder`
- `datetime`
- `hashlib`
- `json`
- `pathlib`
- `re`
- `relator`
- `typing`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_prepare.py`
- `Agent/agent/_impl_tui.py`
- `Agent/graph_runner.py`
- `Agent/tools/compact_tools.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/project_brain/build.py`

---

## Entry Points


---

## API / route hints

