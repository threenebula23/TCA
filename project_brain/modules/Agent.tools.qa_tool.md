# Module: Agent.tools.qa_tool

## Purpose

One-shot QA scripts (npm/pnpm) to catch build and framework errors before shipping.

---

## Responsibilities

- One-shot QA scripts (npm/pnpm) to catch build and framework errors before shipping.

---

## Public API

|name|description|
|---|---|
|run_package_script|npm|pnpm|yarn run <script> (default build) в cwd; для dev — run_command(background=True).|

---

## Dependencies

- `Agent.path_utils`
- `Terminal.runner`
- `__future__`
- `langchain_core.tools`
- `path_utils`
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

- `Agent/tools/qa_tool.py`

---

## Entry Points


---

## API / route hints

