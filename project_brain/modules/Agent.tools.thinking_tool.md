# Module: Agent.tools.thinking_tool

## Purpose

Thinking and analysis tools for the Lorne agent.

---

## Responsibilities

- Thinking and analysis tools for the Lorne agent.

---

## Public API

|name|description|
|---|---|
|think|Краткая запись рассуждения; отображается в панели Thoughts.|
|show_diff|Unified diff old_content vs new_content для path (визуализация перед edit_file).|
|analyze_code|Анализировать код в файле с использованием RAG.
Находит релевантные чанки кода и возвращает их с контекстом.
path — путь к файлу, query — что именно анализировать.|

---

## Dependencies

- `Agent.rag`
- `Agent.tools.file_ops`
- `Interface.tui_bridge`
- `__future__`
- `difflib`
- `langchain_core.tools`
- `typing`

---

## Used By

- `Agent/background_agent_runner.py`
- `Agent/deep_solver/legacy_loop.py`
- `Agent/tool_registry.py`
- `Agent/tools/compact_tools.py`
- `tests/test_file_ops.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- May perform I/O when executed

---

## Risks


---

## File Paths

- `Agent/tools/thinking_tool.py`

---

## Entry Points


---

## API / route hints

