# Module: Agent.deep_solver._impl_a

## Purpose

Deep Solver — long-running, local-only autonomous coding agent.

---

## Responsibilities

- Deep Solver — long-running, local-only autonomous coding agent.

---

## Public API

|name|description|
|---|---|
|is_running|True if a Deep Solver run is currently active in this process.|
|submit_user_message|Feed a mid-run user message into the live Deep loop.

Returns True if the message was queued (run is active), False otherwise
(caller should start a new run or behave normally).|
|register_checkpoint||
|get_checkpoint||
|clear_checkpoint||
|list_checkpoints|Return all live deep checkpoints (newest first).|

---

## Dependencies

- `Agent.checkpoint`
- `Agent.creator_provider`
- `Agent.llm_provider`
- `Agent.message_utils`
- `Agent.runtime_paths`
- `Agent.tool_registry`
- `Agent.tool_schemas`
- `__future__`
- `checkpoint`
- `creator_provider`
- `json`
- `langchain_core.messages`
- `langchain_core.tools`
- `llm_provider`
- `message_utils`
- `os`
- `runtime_paths`
- `threading`
- `time`
- `tool_registry`
- `tool_schemas`
- `typing`
- `uuid`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_tui.py`
- `tests/test_deep_solver.py`
- `tests/test_ollama_provider.py`
- `tests/test_package_imports.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/deep_solver/_impl_a.py`

---

## Entry Points


---

## API / route hints

