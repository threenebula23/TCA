# Module: Agent.rag

## Purpose

RAG (Retrieval-Augmented Generation) for Lorne.

---

## Responsibilities

- RAG (Retrieval-Augmented Generation) for Lorne.

---

## Public API

|name|description|
|---|---|
|index_project_brain|Index ``<root>/project_brain/**/*.md`` and ``rag_manifest.json`` with ``source=brain``.|
|index_documents|Index files in root_path with incremental mtime-based caching.

Args:
    root_path: Root directory to index
    pattern: Glob pattern (if "*.py", uses env patterns instead)
    progress_callback: Optional callable(current, total) for progr…|
|query|Search chunks: **Project Brain first**, then code; lexical + IDF scoring.|
|get_index_stats|Return statistics about the current index.|
|get_rag_tool|Create a LangChain @tool for the agent to search indexed documents.|

---

## Dependencies

- `Agent.file_loading`
- `Agent.path_utils`
- `Agent.runtime_paths`
- `file_loading`
- `langchain_core.tools`
- `math`
- `pathlib`
- `re`
- `runtime_paths`
- `typing`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_prepare.py`
- `Agent/agent/_impl_tui.py`
- `Agent/command_router/_main.py`
- `Agent/command_router/_mixin_handlers.py`
- `Agent/project_brain/agent_architecture.py`
- `Agent/tool_registry.py`
- `Agent/tools/compact_tools.py`
- `Agent/tools/thinking_tool.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/rag/__init__.py`

---

## Entry Points


---

## API / route hints

