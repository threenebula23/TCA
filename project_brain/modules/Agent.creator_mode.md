# Module: Agent.creator_mode

## Purpose

Creator Mode — оркестратор параллельных агентов для Lorne.

---

## Responsibilities

- Creator Mode — оркестратор параллельных агентов для Lorne.

---

## Public API

|name|description|
|---|---|
|run_creator_mode|Запустить Creator Mode для задачи.

Args:
    task: Основная задача пользователя
    tools: Список инструментов доступных агентам
    project_context: Контекст проекта (структура, etc.)
    depth: Current recursion depth (0 = root)
    pare…|

---

## Dependencies

- `Agent.creator_orchestration`
- `Agent.creator_provider`
- `Agent.message_utils`
- `Agent.path_utils`
- `Agent.planner`
- `Agent.project_brain.agent_architecture`
- `Agent.system_prompt`
- `Agent.tool_registry`
- `Agent.tool_schemas`
- `Agent.tools.terminal_tool`
- `Interface.graph_display`
- `Interface.tui_bridge`
- `Interface.visualization`
- `__future__`
- `concurrent.futures`
- `creator_orchestration`
- `creator_provider`
- `json`
- `langchain_core.messages`
- `langchain_core.tools`
- `langchain_openai`
- `langgraph.graph`
- `message_utils`
- `pathlib`
- `planner`
- `system_prompt`
- `threading`
- `time`
- `tool_registry`
- `tool_schemas`
- `typing`

---

## Used By

- `Agent/agent/_impl_prepare.py`
- `Agent/deep_solver/legacy_loop.py`
- `tests/test_creator_mode_local_tools.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/creator_mode.py`

---

## Entry Points


---

## API / route hints

