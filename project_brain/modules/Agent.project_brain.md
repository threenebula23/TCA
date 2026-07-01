# Module: Agent.project_brain

## Purpose

Project Brain: static scan + Markdown output for RAG (optional Relator).

---

## Responsibilities

- Project Brain: static scan + Markdown output for RAG (optional Relator).

---

## Public API

|name|description|
|---|---|
|refresh_project_brain|Scan ``root``, write ``project_brain/**``, return summary paths.|
|read_brain_context_summary|Read a short, prioritised excerpt of ``project_brain/*.md`` for prompt injection.

Originally only Deep Solver / Creator workers got this auto-injected
(see ``deep_solver/legacy_loop.py:_read_brain_context`` and
``creator_mode.py:_read_brai…|

---

## Dependencies

- `Agent.path_utils`
- `__future__`
- `agent_architecture`
- `build`
- `context_builder`
- `pathlib`
- `scanner`
- `typing`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_prepare.py`
- `Agent/agent/_impl_tui.py`
- `Agent/creator_mode.py`
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

- `Agent/project_brain/__init__.py`

---

## Entry Points


---

## API / route hints

