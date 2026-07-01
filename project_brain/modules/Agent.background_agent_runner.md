# Module: Agent.background_agent_runner

## Purpose

Short-lived background agent: LLM + tools in a worker thread (testing while main blocks).

---

## Responsibilities

- Short-lived background agent: LLM + tools in a worker thread (testing while main blocks).

---

## Public API

|name|description|
|---|---|
|run_short_agent_loop|Synchronous mini-loop; used from worker thread.|
|start_background_job||
|get_job_status||
|wait_for_job||

---

## Dependencies

- `Agent.message_utils`
- `Agent.tool_schemas`
- `Agent.tools`
- `Interface.tui_bridge`
- `__future__`
- `json`
- `langchain_core.messages`
- `os`
- `threading`
- `time`
- `typing`
- `uuid`

---

## Used By

- `Agent/tools/parallel_helper_tool.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/background_agent_runner.py`

---

## Entry Points


---

## API / route hints

