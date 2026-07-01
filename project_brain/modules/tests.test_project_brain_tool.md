# Module: tests.test_project_brain_tool

## Purpose

Smoke test for ``project_brain_tool`` (writes under tmp project).

---

## Responsibilities

- Smoke test for ``project_brain_tool`` (writes under tmp project).

---

## Public API

|name|description|
|---|---|
|tmp_project||
|test_project_brain_tool_refresh_writes_and_indexes||
|test_project_brain_write_architecture_append||
|test_project_brain_write_architecture_requires_content||
|test_project_brain_write_brain_agent_path||
|test_project_brain_write_brain_root_notes||
|test_project_brain_write_brain_rejects_scanner_overview||
|test_project_brain_coerce_action_write_with_rel_path||
|test_agent_graph_includes_brain_sync_node||
|test_brain_refresh_respects_set_project_root|Brain must target the configured workspace root, not necessarily cwd.|
|test_build_project_context_is_workspace_agnostic||

---

## Dependencies

- `Agent.graph_runner`
- `Agent.path_utils`
- `Agent.project_brain.context_builder`
- `Agent.tool_schemas`
- `Agent.tools.compact_tools`
- `__future__`
- `os`
- `pathlib`
- `pytest`
- `unittest.mock`

---

## Used By


---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `tests/test_project_brain_tool.py`

---

## Entry Points


---

## API / route hints

