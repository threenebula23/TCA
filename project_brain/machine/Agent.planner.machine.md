MODULE: Agent.planner

PURPOSE:
Task planner for Lorne — generates structured plans for complex tasks.

PUBLIC_API:
|name|description|
|---|---|
|build_plan|Call the LLM in planning mode and return a list of steps.

Validates the response against a strict JSON-array-of-strings schema and
retries once with a corrective nudge before falling back to a naive
line-split (and finally a generic hardco…|
|build_creator_plan|Planner tuned for Creator: fewer, larger parallelizable subtasks.

Same validate-then-retry-once strategy as ``build_plan`` (see there).|

DEPENDENCIES:
- Agent.llm_provider
- __future__
- json
- langchain_core.messages
- llm_provider
- pydantic
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_prepare.py
- Agent/creator_mode.py
- tests/test_ollama_provider.py

RISKS:
