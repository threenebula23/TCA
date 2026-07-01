MODULE: Agent.tools.notes_tool

PURPOSE:
Free-form session notes and hypotheses scratch pad.

PUBLIC_API:
|name|description|
|---|---|
|session_notes|Свободные заметки сессии: append/read/clear/search.

action: append | read | clear | search
content: текст заметки (для append)
tag: категория ('bug', 'decision', 'todo', 'observation', 'hypothesis', 'fact')
Записывай сюда рассуждения, гипо…|

DEPENDENCIES:
- __future__
- datetime
- langchain_core.tools
- pathlib

SIDE_EFFECTS:
- May perform I/O when executed

USED_BY:
- Agent/background_agent_runner.py
- Agent/deep_solver/legacy_loop.py
- Agent/tool_registry.py
- tests/test_file_ops.py
- tests/test_ollama_provider.py

RISKS:
