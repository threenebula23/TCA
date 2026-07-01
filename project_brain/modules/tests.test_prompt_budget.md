# Module: tests.test_prompt_budget

## Purpose

Regression guards for token budget of the system prompt + tool schemas.

---

## Responsibilities

- Regression guards for token budget of the system prompt + tool schemas.

---

## Public API

|name|description|
|---|---|
|test_system_prompt_text_under_budget|The hand-written system prompt must stay compact.|
|test_project_structure_fits_under_budget|project_structure must not silently exceed ~1800 chars.|
|test_full_session_system_prompt_budget|Composed system message (prompt + custom + project) under ~6k chars.|
|test_tool_schema_json_size_budget|Aggregate tool schemas stay under ~15.2k chars (headroom for small tools e.g. project_brain_tool).|

---

## Dependencies

- `Agent.agent`
- `Agent.system_promt`
- `Agent.tool_registry`
- `__future__`
- `json`
- `langchain_core.utils.function_calling`
- `pathlib`
- `pytest`

---

## Used By


---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `tests/test_prompt_budget.py`

---

## Entry Points


---

## API / route hints

