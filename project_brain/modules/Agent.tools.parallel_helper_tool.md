# Module: Agent.tools.parallel_helper_tool

## Purpose

Background helper: тот же пул тулов + отдельный LLM-цикл в потоке (тест, пока основной run_command ждёт).

---

## Responsibilities

- Background helper: тот же пул тулов + отдельный LLM-цикл в потоке (тест, пока основной run_command ждёт).

---

## Public API

|name|description|
|---|---|
|start_background_task|Фоновый микро-цикл LLM+тулы; вернёт job_id → get_background_result (параллельно с долгим run_command).|
|get_background_result|Статус или ожидание `start_background_task`. wait_seconds=0 — без ожидания.|

---

## Dependencies

- `Agent.background_agent_runner`
- `Agent.llm_provider`
- `Agent.tool_registry`
- `Interface.ui_prefs`
- `__future__`
- `background_agent_runner`
- `json`
- `langchain_core.tools`
- `llm_provider`
- `tool_registry`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/parallel_helper_tool.py`

---

## Entry Points


---

## API / route hints

