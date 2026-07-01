# Module: Agent.planner

## Purpose

Task planner for Lorne — generates structured plans for complex tasks.

---

## Responsibilities

- Task planner for Lorne — generates structured plans for complex tasks.

---

## Public API

|name|description|
|---|---|
|build_plan|Call the LLM in planning mode and return a list of steps.

Validates the response against a strict JSON-array-of-strings schema and
retries once with a corrective nudge before falling back to a naive
line-split (and finally a generic hardco…|
|build_creator_plan|Planner tuned for Creator: fewer, larger parallelizable subtasks.

Same validate-then-retry-once strategy as ``build_plan`` (see there).|

---

## Dependencies

- `Agent.llm_provider`
- `__future__`
- `json`
- `langchain_core.messages`
- `llm_provider`
- `pydantic`
- `typing`

---

## Used By

- `Agent/agent/_impl_prepare.py`
- `Agent/creator_mode.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/planner.py`

---

## Entry Points


---

## API / route hints

