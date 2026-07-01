MODULE: Agent.project_brain.scanner

PURPOSE:
Static project scan (AST, imports, paths) — no LLM. Output feeds ``build_project_context``.

PUBLIC_API:
|name|description|
|---|---|
|scan_project|Return a JSON-serialisable scan: modules, imports, entrypoints, readme, api hints.|

DEPENDENCIES:
- __future__
- ast
- pathlib
- re
- typing

SIDE_EFFECTS:
- Import-time side effects unknown

USED_BY:
- Agent/agent/_impl_classic.py
- Agent/agent/_impl_prepare.py
- Agent/agent/_impl_tui.py
- Agent/graph_runner.py
- Agent/tools/compact_tools.py
- tests/test_ollama_provider.py

RISKS:
