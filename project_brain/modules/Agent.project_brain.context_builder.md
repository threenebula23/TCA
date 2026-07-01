# Module: Agent.project_brain.context_builder

## Purpose

Normalize ``scan_project`` output into a Relator-ready context (flat lists + nested dicts).

---

## Responsibilities

- Normalize ``scan_project`` output into a Relator-ready context (flat lists + nested dicts).

---

## Public API

|name|description|
|---|---|
|build_project_context|Build the JSON model consumed by Relator templates and ``rag_manifest``.|

---

## Dependencies

- `Agent.path_utils`
- `__future__`
- `collections`
- `datetime`
- `json`
- `pathlib`
- `re`
- `typing`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_prepare.py`
- `Agent/agent/_impl_tui.py`
- `Agent/graph_runner.py`
- `Agent/tools/compact_tools.py`
- `tests/test_ollama_provider.py`
- `tests/test_project_brain_tool.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/project_brain/context_builder.py`

---

## Entry Points


---

## API / route hints

