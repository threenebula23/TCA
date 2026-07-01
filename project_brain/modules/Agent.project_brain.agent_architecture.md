# Module: Agent.project_brain.agent_architecture

## Purpose

Model-authored Markdown under ``project_brain/`` (RAG-indexed, mostly not touched by refresh).

---

## Responsibilities

- Model-authored Markdown under ``project_brain/`` (RAG-indexed, mostly not touched by refresh).

---

## Public API

|name|description|
|---|---|
|write_brain_markdown|Записать или дополнить ``project_brain/<rel_path>`` (только .md, без выхода из каталога).

Разрешённые пути см. :func:`_is_allowed_brain_rel`. Запрещены корневые
``overview.md`` / ``architecture.md`` и деревья ``modules/`` … — для них
испол…|
|write_agent_architecture|Совместимость: то же, что ``write_brain_markdown(..., agent_architecture.md, ...)``.|
|reindex_brain_rag|Reload brain chunks from disk into the in-process RAG store.|
|run_brain_sync_if_enabled|Reindex brain docs after an agent turn (respect ``LORNE_SKIP_BRAIN_SYNC``).|

---

## Dependencies

- `Agent.path_utils`
- `Agent.rag`
- `__future__`
- `datetime`
- `os`
- `pathlib`
- `re`

---

## Used By

- `Agent/agent/_impl_classic.py`
- `Agent/agent/_impl_prepare.py`
- `Agent/agent/_impl_tui.py`
- `Agent/creator_mode.py`
- `Agent/graph_runner.py`
- `Agent/tools/compact_tools.py`
- `tests/test_ollama_provider.py`

---

## Side Effects

- Import-time side effects unknown

---

## Risks


---

## File Paths

- `Agent/project_brain/agent_architecture.py`

---

## Entry Points


---

## API / route hints

