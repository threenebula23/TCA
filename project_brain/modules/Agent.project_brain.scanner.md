# Module: Agent.project_brain.scanner

## Purpose

Static project scan (AST, imports, paths) — no LLM. Output feeds ``build_project_context``.

---

## Responsibilities

- Static project scan (AST, imports, paths) — no LLM. Output feeds ``build_project_context``.

---

## Public API

|name|description|
|---|---|
|scan_project|Return a JSON-serialisable scan: modules, imports, entrypoints, readme, api hints.|

---

## Dependencies

- `__future__`
- `ast`
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

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/project_brain/scanner.py`

---

## Entry Points


---

## API / route hints

- L53: if re.search(r"\b(FastAPI|APIRouter|@router\.|@app\.(get|post|put|delete|patch)|add_url_rule)\b", s):
