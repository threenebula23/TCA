# Module: Agent.agent._impl_prepare

## Purpose

Агент Lorne: основной цикл (TUI и classic), LangGraph, сессии, откаты, режимы.

---

## Responsibilities

- Агент Lorne: основной цикл (TUI и classic), LangGraph, сессии, откаты, режимы.

---

## Public API

|name|description|
|---|---|
|analyze_project_structure|Build a compact tree of the project for the system prompt.

Designed to stay under ~600 tokens: depth-limited, no per-file sizes,
truncation of oversized directories and a hard character cap at the
end. The model can always recover detail v…|

---

## Dependencies

- `Agent.checkpoint`
- `Agent.command_router`
- `Agent.creator_mode`
- `Agent.creator_provider`
- `Agent.creator_summary`
- `Agent.graph_runner`
- `Agent.llm_provider`
- `Agent.message_utils`
- `Agent.path_utils`
- `Agent.planner`
- `Agent.project_brain`
- `Agent.prompts.project_brain_rules`
- `Agent.rag`
- `Agent.runtime_paths`
- `Agent.spinner`
- `Agent.system_prompt`
- `Agent.tool_registry`
- `Interface.ui_prefs`
- `Interface.visualization`
- `checkpoint`
- `command_router`
- `creator_mode`
- `creator_provider`
- `creator_summary`
- `dotenv`
- `graph_runner`
- `json`
- `langchain_core.messages`
- `llm_provider`
- `message_utils`
- `os`
- `path_utils`
- `pathlib`
- `planner`
- `rag`
- `rich`
- `rich.panel`
- `spinner`
- `sys`
- `system_prompt`
- `time`
- `tool_registry`
- `typing`

---

## Used By

- `Agent/agent/_impl_tui.py`
- `Agent/command_router/_main.py`
- `Interface/panels/file_explorer.py`
- `lorne.py`
- `tests/test_ollama_provider.py`
- `tests/test_prompt_budget.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/agent/_impl_prepare.py`

---

## Entry Points


---

## API / route hints

